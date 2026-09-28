"""Real native-worker GPU/optimizer/restore test; synthetic tensors, no ALFWorld RL.

Invoke via torch.distributed.run with explicitly assigned GPUs. The test uses
one microbatch per rank, not a complete 128-row optimization minibatch. It tests
the full 4096+512 length boundary and leaves its own checkpoint for inspection.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--embedding-router-model', type=Path, help='Also hold the frozen CPU router during native GPU checks')
    p.add_argument('--embedding-router-backend', choices=['skillrl_embedding_state', 'skillrl_embedding_state_batch'],
                   default='skillrl_embedding_state')
    p.add_argument('--rollout-micro-batch-size', type=int, choices=[1, 2, 16], default=1)
    p.add_argument('--optimizer-offload', action='store_true', help='Native CPU state storage between GPU Adam updates')
    p.add_argument('--inference-profile', type=Path, help='Opt in to the frozen V1 bridge; historical defaults stay unchanged')
    p.add_argument('--generation-cap', type=int, choices=[8, 512], default=8)
    p.add_argument('--total-training-steps', type=int, choices=[5, 150], default=150)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        p.error('Explicit --execute required; this loads GPUs and creates a synthetic checkpoint')
    if os.environ.get('PHASE2_CPU_ADAM') == '1' or os.environ.get('PHASE2_ELASTIC_TRAINING') == '1':
        p.error('Legacy CPU Adam / elastic overrides are not part of this native preflight')
    # torchrun already assigns LOCAL_RANK; the Ray-specific hook would try to
    # auto-create a separate cluster in every rank. It must not run here.
    os.environ.pop('RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES', None)
    import torch
    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from scripts.inspect_skillrl_alignment import load_config
    from verl import DataProto
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from phase3.common import require, write_new
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    require(world in (4, 8), 'Preflight requires the planned four/eight ranks')
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    output = a.output.resolve()
    require(not (output / f'rank-{rank}.json').exists(), 'Do not overwrite preflight evidence')
    started = previous = time.monotonic()
    timings = {}

    def record_stage(name, **details):
        nonlocal previous
        torch.cuda.synchronize()
        now = time.monotonic()
        timings[name] = now - previous
        write_new(output / 'stages' / f'rank-{rank}-{name}.json', {
            'rank': rank, 'stage': name, 'elapsed_seconds': now - started,
            'stage_seconds': timings[name], 'synthetic_only': True,
            'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(), **details})
        previous = now

    cfg = load_config().actor_rollout_ref
    if a.inference_profile:
        from skillnet_cohort.inference import registration, apply_training
        require(a.optimizer_offload, 'V1 preflight must test two consecutive native optimizer updates')
        require(a.rollout_micro_batch_size == 16 and a.generation_cap == 512,
                'V1 preflight must exercise the registered 16-sequence / 512-token maximum')
        wrapper = OmegaConf.create({'actor_rollout_ref': cfg})
        apply_training(wrapper, registration(a.inference_profile))
        cfg = wrapper.actor_rollout_ref
    cfg.model.path = str(a.model.resolve())
    cfg.rollout.prompt_length = 4096
    cfg.rollout.response_length = 512
    cfg.rollout.micro_batch_size = a.rollout_micro_batch_size
    cfg.actor.optim.total_training_steps = a.total_training_steps
    cfg.actor.fsdp_config.optimizer_offload = a.optimizer_offload
    cfg.actor.fsdp_config.cpu_shard_init = True
    cfg.ref.fsdp_config.cpu_shard_init = True
    cfg.cohort_seed = 404
    OmegaConf.resolve(cfg)
    actor = ActorRolloutRefWorker(OmegaConf.create(OmegaConf.to_container(cfg)), 'actor_rollout')
    actor.init_model()
    ref = ActorRolloutRefWorker(OmegaConf.create(OmegaConf.to_container(cfg)), 'ref')
    ref.init_model()
    device = torch.cuda.current_device()
    print(f'PREFLIGHT rank={rank} models_initialized', flush=True)
    record_stage('models_initialized')
    embedding_router, embedding_records = None, []
    if a.embedding_router_model:
        if rank == 0:
            from agent_system.memory.skillnet_runtime import create_embedding_skillnet37_runtime
            if a.embedding_router_backend == 'skillrl_embedding_state_batch':
                from agent_system.memory.skillnet_runtime import create_batched_embedding_skillnet37_runtime
                create_embedding_skillnet37_runtime = create_batched_embedding_skillnet37_runtime
            from scripts.test_skillrl_embedding_router import SYNTHETIC_STATES
            memory, embedding_router = create_embedding_skillnet37_runtime(model_path=a.embedding_router_model,
                device='cpu', cache_path=output / 'synthetic-router.sqlite3', max_local_calls=2)
            selected = embedding_router.route(memory.retrieve(''), **SYNTHETIC_STATES[0])
            embedding_records.append(selected['skill_router_api'])
            print('PREFLIGHT rank=0 frozen_cpu_router_coexists_with_actor_ref', flush=True)
        torch.distributed.barrier()
        record_stage('cpu_router_coexistence')
    ids = torch.full((1, 4608), 100, dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    batch = DataProto(TensorDict({'input_ids':ids, 'attention_mask':mask,
        'position_ids':torch.arange(4608, device=device)[None], 'responses':ids[:, -512:].clone()}, batch_size=[1]),
        meta_info={'temperature':1., 'global_token_num':[4608]*world, 'multi_turn':False})
    batch = batch.union(actor.compute_log_prob(batch))
    batch = batch.union(ref.compute_ref_log_prob(batch))
    batch.batch['advantages'] = torch.ones((1,512), device=batch.batch['responses'].device)
    print(f'PREFLIGHT rank={rank} probability_forwards_ok', flush=True)
    record_stage('probability_forwards')
    update = actor.update_actor(batch)
    require(all(torch.isfinite(param).all().item() for param in actor.actor_module_fsdp.parameters()), 'Non-finite updated parameter')
    if a.optimizer_offload:
        require(bool(actor.actor_optimizer.state) and all(value.device.type == 'cpu'
            for state in actor.actor_optimizer.state.values() for value in state.values() if torch.is_tensor(value)),
            'Native optimizer states were not offloaded after the GPU update')
    print(f'PREFLIGHT rank={rank} backward_optimizer_ok', flush=True)
    record_stage('backward_optimizer', optimizer_offload=a.optimizer_offload)
    checkpoint = output / 'synthetic-checkpoint' / 'actor'
    actor.save_checkpoint(str(checkpoint), global_step=1, max_ckpt_to_keep=None)
    weights = [param.detach().cpu().clone() for param in actor.actor_module_fsdp.parameters()]
    optimizer = [{key: value.detach().cpu().clone() if torch.is_tensor(value) else value for key,value in state.items()}
                 for state in actor.actor_optimizer.state.values()]
    rng_expected = torch.rand(16, device=device).cpu()
    with torch.no_grad():
        for param in actor.actor_module_fsdp.parameters():
            param.flatten()[0] += 1.
    actor.load_checkpoint(str(checkpoint), del_local_after_load=False)
    rng_actual = torch.rand(16, device=device).cpu()
    require(torch.equal(rng_expected, rng_actual), 'Native RNG restore mismatch')
    require(all(torch.equal(before, after.detach().cpu()) for before,after in zip(weights, actor.actor_module_fsdp.parameters())), 'Native parameter restore mismatch')
    require(len(optimizer) == len(actor.actor_optimizer.state), 'Optimizer state count changed')
    for before, after in zip(optimizer, actor.actor_optimizer.state.values()):
        for key, value in before.items():
            require(torch.equal(value, after[key].cpu()) if torch.is_tensor(value) else value == after[key], 'Native optimizer restore mismatch')
    if a.optimizer_offload:
        require(all(value.device.type == 'cpu' for state in actor.actor_optimizer.state.values()
            for value in state.values() if torch.is_tensor(value)), 'Restored optimizer was not offloaded')
    record_stage('native_checkpoint_restore', optimizer_offload=a.optimizer_offload)
    # Batch-2 exercises full 4096-token prompts. The requested generation cap
    # and actual unpadded response tokens are recorded separately.
    count = a.rollout_micro_batch_size
    prompt_width = 4096 if count > 1 else 16
    prompt = DataProto(TensorDict({'input_ids':ids[:, :prompt_width].repeat(count, 1),
        'attention_mask':mask[:, :prompt_width].repeat(count, 1),
        'position_ids':torch.arange(prompt_width,device=device)[None].repeat(count, 1)},
        batch_size=[count]), meta_info={'response_length':a.generation_cap})
    generated = actor.generate_sequences(prompt)
    require(generated.batch['responses'].shape == (count,512), 'Rollout returned wrong padded shape')
    actual_response_tokens = generated.batch['attention_mask'][:, -512:].sum(dim=1).tolist()
    record_stage('vllm_generation' if a.inference_profile else 'hf_generation',
                 prompt_tokens=prompt_width, generation_cap=a.generation_cap,
                 actual_response_tokens=actual_response_tokens)
    print(f'PREFLIGHT rank={rank} generation_after_restore_ok tokens={actual_response_tokens}', flush=True)
    # One more synthetic update specifically exercises CPU->GPU optimizer
    # reloading. This is not a second experiment or an ALFWorld RL update.
    if a.optimizer_offload:
        actor.update_actor(batch)
        require(all(torch.isfinite(param).all().item() for param in actor.actor_module_fsdp.parameters()),
                'Non-finite parameter after reloaded optimizer update')
        require(all(value.device.type == 'cpu' for state in actor.actor_optimizer.state.values()
            for value in state.values() if torch.is_tensor(value)), 'Second optimizer offload failed')
        record_stage('reloaded_optimizer_update', optimizer_offload=True)
        print(f'PREFLIGHT rank={rank} optimizer_reload_update_ok', flush=True)
    if a.inference_profile:
        # A second decode must load NEW actor weights after the native update.
        bridge = actor.rollout_sharding_manager
        require(bridge.dirty and bridge.sync_count == 1, 'Actor update did not invalidate vLLM')
        actor.generate_sequences(prompt)
        require(not bridge.dirty and bridge.sync_count == 2, 'Post-update vLLM resync failed')
        actor.generate_sequences(prompt)
        require(bridge.sync_count == 2, 'Unchanged weights were unnecessarily retransferred per state')
        bridge.suspend()
        record_stage('vllm_post_update_resync', weight_version=bridge.weight_version,
                     sync_count=bridge.sync_count, last_sync=bridge.last_sync)
    if embedding_router is not None:
        selected = embedding_router.route(memory.retrieve(''), **SYNTHETIC_STATES[1])
        embedding_records.append(selected['skill_router_api'])
        require(all(item['api_calls_this_step'] == 0 for item in embedding_records), 'No paid router in preflight')
        write_new(output / 'embedding-router.json', {'status':'PASS', 'device':'cpu',
            'protocol_sha256':embedding_router.protocol_hash, 'decisions':embedding_records,
            'coexists_with_gpu_forward_backward_and_restore':True, 'synthetic_only':True})
        embedding_router.close()
    write_new(output / f'rank-{rank}.json', {'rank':rank, 'world_size':world, 'model':str(a.model.resolve()),
        'status':'PASS', 'full_length_forward_backward':True, 'native_weight_optimizer_rng_restore':True,
        'generation_after_restore':True, 'inference_backend':cfg.rollout.name,
        'synthetic_only':True, 'real_rl_started':False,
        'rollout_microbatch':count, 'generation_prompt_tokens':prompt_width, 'generation_cap_tested':a.generation_cap,
        'actual_response_tokens':actual_response_tokens, 'optimizer_offload':a.optimizer_offload,
        'gpu_adam_with_state_offload_not_cpu_adam':a.optimizer_offload,
        'synthetic_optimizer_steps':2 if a.optimizer_offload else 1,
        'optimizer_reload_update_validated':a.optimizer_offload,
        'stage_seconds':timings, 'elapsed_seconds':time.monotonic() - started,
        'max_allocated_bytes':torch.cuda.max_memory_allocated(), 'max_reserved_bytes':torch.cuda.max_memory_reserved()})
    torch.distributed.barrier()
    if rank == 0:
        write_new(output / 'complete.json', {'status':'PASS', 'world_size':world, 'synthetic_only':True,
            'frozen_cpu_embedding_router_coexistence':bool(a.embedding_router_model),
            'embedding_router_backend':a.embedding_router_backend, 'rollout_microbatch':count,
            'optimizer_offload':a.optimizer_offload, 'generation_cap_tested':a.generation_cap,
            'training_horizon_configured':a.total_training_steps,
            'not_verified':['Ray task orchestration', 'real ALFWorld full-prompt coverage', 'full-128-row optimizer batch',
                            'exact all-vocabulary recording storage', 'whole-pipeline 30h completion']})
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
