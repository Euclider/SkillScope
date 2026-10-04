"""Eight-rank fixed native checkpoint/full-minibatch probe, no checkpoint writes.

Each mode runs in a fresh torchrun group. An optional-mode OOM cannot poison a
subsequent training CUDA context. Input IDs, masks and positions are reused
verbatim; only the output head and communication/offload flags change.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path


def main():
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy
    from skillnet_cohort.cpu_shard_init import verify_identical_cpu_model
    from skillnet_cohort.native_restore import restore_parameters
    from phase3.common import write_new
    from .runtime import probe_indices

    p = argparse.ArgumentParser()
    for name in ('batch', 'config', 'native', 'output', 'baseline'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--mode', choices=['baseline', 'suffix', 'reference_gpu_root', 'no_sync'], required=True)
    p.add_argument('--gpu-reference', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    assert dist.get_world_size() == 8
    mesh = init_device_mesh('cuda', (8,), mesh_dim_names=('fsdp',))
    cfg = OmegaConf.create(json.loads(args.config.read_text()))
    payload = torch.load(args.batch, map_location='cpu', weights_only=True, mmap=True)
    source = payload['tensors']
    indices = probe_indices(source['actual_loss_mask'], source['advantages'], 128)[rank * 16:(rank + 1) * 16]
    names = ('input_ids', 'attention_mask', 'position_ids', 'responses', 'old_log_probs', 'advantages')
    data = {k: source[k][indices].clone().cuda() for k in names}
    data['loss_mask'] = source['actual_loss_mask'][indices].clone().cuda()
    temperature = float(payload['temperature'])
    assert temperature == cfg.actor_rollout_ref.rollout.temperature
    del source, payload
    suffix = args.mode != 'baseline'
    gpu_ref = args.mode == 'reference_gpu_root' or args.gpu_reference
    nosync = args.mode == 'no_sync'
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

    def build(reference):
        dtype = torch.bfloat16 if reference else torch.float32
        model = AutoModelForCausalLM.from_pretrained(cfg.actor_rollout_ref.model.path,
                    dtype=dtype, attn_implementation='sdpa', local_files_only=True).to(dtype=dtype)
        if not reference:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        wrap = get_fsdp_wrap_policy(model, config={'disable': reference and gpu_ref, 'min_num_params': 0})
        verify_identical_cpu_model(model)
        model = FSDP(model, device_id=None, auto_wrap_policy=wrap, device_mesh=mesh,
            cpu_offload=CPUOffload(offload_params=True) if reference and not gpu_ref else None,
            mixed_precision=mp, use_orig_params=False, sync_module_states=False, forward_prefetch=False)
        if not reference or gpu_ref:
            model.to(torch.cuda.current_device())
        return model

    # Keep the actual Adam moments and exact native FP32 parameters resident.
    actor_model = build(False)
    native = args.native / 'actor'
    state = torch.load(native / f'model_world_size_8_rank_{rank}.pt', map_location='cpu', weights_only=False)
    restore_parameters(actor_model, state, rank, 8)
    del state
    actor_cfg = OmegaConf.create(OmegaConf.to_container(cfg.actor_rollout_ref.actor, resolve=True))
    actor_cfg.ppo_mini_batch_size = 16
    actor_cfg.response_logits_only = suffix
    actor_cfg.accumulate_no_sync = nosync
    actor_cfg.trim_common_padding = False
    optim = torch.optim.AdamW(actor_model.parameters(), lr=1e-6, weight_decay=.01)
    optim.load_state_dict(torch.load(native / f'optim_world_size_8_rank_{rank}.pt', map_location='cpu', weights_only=False))
    actor = DataParallelPPOActor(actor_cfg, actor_model, optim)
    reference = build(True)
    ref_cfg = OmegaConf.create(OmegaConf.to_container(cfg.actor_rollout_ref.ref, resolve=True))
    ref_cfg.response_logits_only = suffix
    ref_cfg.trim_common_padding = False
    ref = DataParallelPPOActor(ref_cfg, reference)
    score_proto = DataProto.from_dict(tensors=data, meta_info=dict(micro_batch_size=1,
                    temperature=temperature, use_dynamic_bsz=False))
    ref.compute_log_prob(score_proto[:1])
    dist.barrier(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    ref_lp, _ = ref.compute_log_prob(score_proto)
    torch.cuda.synchronize()
    reference_seconds = time.perf_counter() - started
    if gpu_ref:
        reference._handle.reshard(True)
    data['ref_log_prob'] = ref_lp.detach()
    valid = data['loss_mask'].bool()
    actor_lp, actor_entropy = [], []
    actor_model.eval()
    with torch.no_grad():
        for i in range(16):
            ent, lp = actor._forward_micro_batch({k: v[i:i+1] for k, v in data.items()}, temperature, True)
            actor_lp.append(lp.cpu()); actor_entropy.append(ent.cpu())
    scores = dict(logprob=torch.cat(actor_lp), entropy=torch.cat(actor_entropy), reference=ref_lp.cpu(),
                  indices=indices)
    base_scores = scores if args.mode == 'baseline' else torch.load(
        args.baseline / f'scores-rank{rank}.pt', map_location='cpu', weights_only=True)
    assert base_scores['indices'] == indices
    lp_error = max((scores[k] - base_scores[k])[valid.cpu()].abs().max().item() for k in ('logprob', 'reference'))
    entropy_error = (scores['entropy'] - base_scores['entropy'])[valid.cpu()].abs().max().item()
    captured = []

    def capture_step():
        captured.append([p.grad.detach().cpu().clone() for p in actor_model.parameters()])
        return actor_model.clip_grad_norm_(float(actor_cfg.grad_clip))

    actor._optimizer_step = capture_step  # Never updates/writes policy or Adam state.
    proto = DataProto.from_dict(tensors=data, meta_info=dict(temperature=temperature, multi_turn=True))
    actor.config.ppo_mini_batch_size = 1
    actor.update_policy(proto[:1]); captured.clear()
    actor.config.ppo_mini_batch_size = 16
    gc.collect(); torch.cuda.empty_cache()
    rng = torch.load(native / f'extra_state_world_size_8_rank_{rank}.pt', map_location='cpu', weights_only=False)
    from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager
    BaseCheckpointManager.load_rng_state(rng['rng'])
    dist.barrier(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    metrics = actor.update_policy(proto)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**30
    grads = captured.pop()
    finite = all(torch.isfinite(x).all().item() for x in grads)
    initial = grads if args.mode == 'baseline' else torch.load(
        args.baseline / f'gradient-rank{rank}.pt', map_location='cpu', weights_only=True, mmap=True)
    values = torch.zeros(4, dtype=torch.float64)
    assert len(initial) == len(grads)
    for a, b in zip(initial, grads):
        assert a.shape == b.shape
        for aa, bb in zip(a.flatten().split(1000000), b.flatten().split(1000000)):
            aa, bb = aa.double(), bb.double()
            values += torch.tensor([aa.square().sum().item(), bb.square().sum().item(),
                                   (aa * bb).sum().item(), (aa - bb).square().sum().item()], dtype=torch.float64)
    values = values.cuda(); dist.all_reduce(values)
    s0, s1, dot, diff = values.cpu().tolist()
    # Root-GPU reference must also coexist with native actor/Adam and the actual
    # registered vLLM engine. This is a memory/weight-sync smoke test, not a new
    # environment rollout, and its sampled output is never used for learning.
    generation_smoke = False
    if gpu_ref:
        from verl.workers.rollout.vllm_v1 import VLLMV1Rollout
        from verl.workers.sharding_manager.fsdp_vllm_v1 import FSDPVLLMV1ShardingManager
        tokenizer = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True)
        with torch.random.fork_rng(devices=[rank]):
            rollout = VLLMV1Rollout(cfg.actor_rollout_ref.model.path, cfg.actor_rollout_ref.rollout, tokenizer)
        manager = FSDPVLLMV1ShardingManager(actor_model, rollout.inference_engine, offload_param=False)
        prompt_len = data['input_ids'].shape[-1] - data['responses'].shape[-1]
        prompts = DataProto.from_dict(tensors={k: data[k][:, :prompt_len] for k in
            ('input_ids', 'attention_mask', 'position_ids')}, meta_info={
            'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.pad_token_id})
        with manager:
            generated = rollout.generate_sequences(prompts)
        assert len(generated) == 16
        manager.suspend()
        generation_smoke = True
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
    row = dict(passed=True, finite=finite, world_size=8, rows_per_rank=16,
        optimizer_boundaries=len(metrics['actor/grad_norm']), unclipped_grad_norm=metrics['actor/grad_norm'][0],
        max_logprob_error=lp_error, max_entropy_error=entropy_error,
        grad_relative_l2=(diff / max(s0, 1e-30))**.5, grad_cosine=dot / max((s0 * s1)**.5, 1e-30),
        seconds=seconds, reference_seconds=reference_seconds, peak_gib=peak,
        total_gib=torch.cuda.get_device_properties(rank).total_memory / 2**30, indices=indices,
        reference_gpu_root=gpu_ref, vllm_colocation_passed=generation_smoke)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'baseline':
        for name, value in [('scores', scores), ('gradient', grads)]:
            with (args.output / f'{name}-rank{rank}.pt').open('xb') as stream:
                torch.save(value, stream)
    write_new(args.output / f'rank{rank}.json', row)
    gathered = [None] * 8
    dist.all_gather_object(gathered, row)
    if rank == 0:
        write_new(args.output / 'result.json', {'mode': args.mode, 'ranks': gathered})
        print('PROBE_COMPLETE', args.mode, flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
