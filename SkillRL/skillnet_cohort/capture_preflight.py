"""Eight-GPU synthetic 128-row accumulation + live FP32 vocabulary I/O check.

No real RL iteration, no generation and no checkpoint. Each rank has16 rows at
the4096+512 tensor boundary, with16 valid response tokens per row. The padding
mask limits temporary disk volume (~4.1GB) without changing vocabulary or dtype.
The512-valid-token compute boundary was separately tested by gpu_preflight.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from .common import file_hash, read_json, write_new_json


def synthetic_batch(rank, world, device):
    import torch
    from tensordict import TensorDict
    from verl import DataProto
    if world != 8 or rank not in range(8):
        raise ValueError('This capture probe requires exactly eight ranks')
    rows, prompt, response, valid = 16, 4096, 512, 16
    ids = torch.full((rows, prompt + response), 100, dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    mask[:, prompt + valid:] = 0
    return DataProto(TensorDict({'input_ids': ids, 'attention_mask': mask,
        'position_ids': torch.arange(prompt + response, device=device)[None].repeat(rows, 1),
        'responses': ids[:, -response:].clone(),
        'phase2_row_index': torch.arange(rank * rows, (rank + 1) * rows, device=device)},
        batch_size=[rows]), meta_info={'temperature': 1., 'global_token_num': [prompt + valid] * 128,
                                      'multi_turn': False})


def inspect_row(path, rank, row):
    import torch
    item = torch.load(path, map_location='cpu', weights_only=False)
    lp = item['log_probs']
    if (item['rank'] != rank or item['row_index'] != row
            or lp.dtype != torch.float32 or lp.shape != (16, 248320)
            or not torch.isfinite(lp).all()
            or not torch.equal(item['token_positions'], torch.arange(16))
            or not torch.equal(item['token_ids'], torch.full((16,), 100, dtype=torch.long))):
        raise ValueError('Exact capture identity/shape/mask/dtype/content failed')
    chosen = lp.gather(-1, item['token_ids'][:, None]).squeeze(-1)
    if not torch.equal(chosen, item['chosen_log_probs']):
        raise ValueError('Saved chosen-token values do not match full vocabulary values')
    max_error = float((chosen - item['trainer_chosen_log_probs']).abs().max())
    normalization_error = float(torch.logsumexp(lp, dim=-1).abs().max())
    if max_error > 1e-5 or normalization_error > 1e-5:
        raise ValueError('Full-vocabulary capture differs from trainer or is not normalized')
    return {'row_index': row, 'bytes': path.stat().st_size, 'sha256': file_hash(path),
            'chosen_max_abs_error': max_error, 'normalization_max_abs_error': normalization_error}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-preflight', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        p.error('Explicit --execute required for the new synthetic capture probe')
    if os.environ.get('PHASE2_CPU_ADAM') == '1' or os.environ.get('PHASE2_ELASTIC_TRAINING') == '1':
        p.error('Legacy optimizer/world-size overrides are forbidden')
    os.environ.pop('RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES', None)
    from .throughput_preflight import require_native_pass
    proof = require_native_pass(a.gpu_preflight)
    import torch
    from omegaconf import OmegaConf
    from scripts.inspect_skillrl_alignment import load_config
    from .inference import PROFILE, registration, apply_training
    from phase2.capture import optimizer_counter
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 8:
        raise ValueError('Exactly eight assigned GPUs required')
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    root = a.output.resolve()
    if (root / f'rank-{rank}-start.json').exists():
        raise FileExistsError('Do not restart a failed capture probe')
    write_new_json(root / f'rank-{rank}-start.json', {'rank': rank, 'synthetic_only': True,
        'formal_rl_started': False, 'native_preflight': proof})
    started = previous = time.monotonic()
    timings = {}

    def stage(name):
        nonlocal previous
        torch.cuda.synchronize()
        now = time.monotonic()
        timings[name] = now - previous
        write_new_json(root / 'stages' / f'rank-{rank}-{name}.json',
            {'rank': rank, 'stage': name, 'seconds': timings[name], 'elapsed_seconds': now - started})
        previous = now
        print(f'CAPTURE_PREFLIGHT rank={rank} {name}', flush=True)

    wrapper = load_config()
    apply_training(wrapper, registration(PROFILE))
    cfg = wrapper.actor_rollout_ref
    cfg.model.path = str(a.model.resolve())
    cfg.rollout.prompt_length, cfg.rollout.response_length = 4096, 512
    cfg.actor.optim.total_training_steps = 5
    cfg.actor.fsdp_config.optimizer_offload = True
    cfg.actor.fsdp_config.cpu_shard_init = True
    cfg.ref.fsdp_config.cpu_shard_init = True
    cfg.cohort_seed = 404
    OmegaConf.resolve(cfg)
    actor = ActorRolloutRefWorker(OmegaConf.create(OmegaConf.to_container(cfg)), 'actor_rollout')
    actor.init_model()
    ref = ActorRolloutRefWorker(OmegaConf.create(OmegaConf.to_container(cfg)), 'ref')
    ref.init_model()
    if actor.config.actor.ppo_mini_batch_size != 16 or actor.config.actor.ppo_micro_batch_size_per_gpu != 1:
        raise ValueError('Full128-row native optimizer layout changed')
    stage('models_initialized')
    batch = synthetic_batch(rank, world, torch.cuda.current_device())
    batch.meta_info['phase2_capture'] = {'root': str(root), 'update': 1, 'stage': 'old'}
    batch = batch.union(actor.compute_log_prob(batch))
    stage('old_exact_forward_and_io')
    batch = batch.union(ref.compute_ref_log_prob(batch))
    stage('reference_forward')
    batch.batch['advantages'] = torch.ones_like(batch.batch['responses'], dtype=torch.float32)
    before = optimizer_counter(actor.actor_optimizer)
    actor.update_actor(batch)
    if (optimizer_counter(actor.actor_optimizer) != before + 1
            or actor.actor.gradient_accumulation != 16
            or not all(torch.isfinite(v).all().item() for v in actor.actor_module_fsdp.parameters())
            or not all(v.device.type == 'cpu' for state in actor.actor_optimizer.state.values()
                       for v in state.values() if torch.is_tensor(v))):
        raise ValueError('Full minibatch accumulation/update/offload failed')
    stage('full128_row_optimizer_update')
    batch.meta_info['phase2_capture'] = {'root': str(root), 'update': 1, 'stage': 'new'}
    actor.compute_log_prob(batch)
    stage('new_exact_forward_and_io')
    evidence = {which: [inspect_row(root / f'{which}_logprobs/u0001' / f'row-{row:06d}.pt', rank, row)
                        for row in range(rank * 16, (rank + 1) * 16)] for which in ('old', 'new')}
    stage('capture_roundtrip_verified')
    write_new_json(root / f'rank-{rank}.json', {'status': 'PASS', 'rank': rank, 'synthetic_only': True,
        'rows': 16, 'valid_response_tokens_per_row': 16, 'prompt_tokens': 4096, 'padded_response_width': 512,
        'gradient_accumulation': 16, 'native_optimizer_steps': 1, 'exact_capture': evidence,
        'stage_seconds': timings, 'formal_rl_started': False, 'elapsed_seconds': time.monotonic() - started})
    torch.distributed.barrier()
    if rank == 0:
        records = [read_json(root / f'rank-{i}.json') for i in range(8)]
        if not all(row['status'] == 'PASS' for row in records):
            raise ValueError('Missing capture rank completion')
        write_new_json(root / 'complete.json', {'status': 'PASS', 'synthetic_only': True,
            'full_global_minibatch_rows': 128, 'world_size': 8, 'valid_response_tokens': 2048,
            'old_and_new_row_files': 256, 'formal_rl_iterations': 0,
            'exact_file_bytes': sum(x['bytes'] for row in records for items in row['exact_capture'].values() for x in items),
            'not_verified': ['real Ray rollout orchestration', 'full pipeline time/storage sufficiency',
                             '512-valid-token per-row I/O peak (this test uses16 valid,512 padded)']})
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
