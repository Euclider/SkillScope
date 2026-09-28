"""Native five-update blocks with a preparation-bound global training horizon."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .common import file_hash, read_json, require_authorization, write_new_json
from .runtime import disk_gate, router_backend
from .training import configuration, reject_legacy_overrides


def block_config(preparation, root, permit, start):
    spec = read_json(Path(preparation).parent / 'spec.json')
    if type(start) is not int or start not in range(0, spec['training']['iterations'], 5):
        raise ValueError('Unregistered five-update block')
    cfg = configuration(preparation, root, permit.get('router_max_api_calls', 0), len(permit['gpu_ids']),
                        router_local_calls=permit.get('router_max_local_calls', 0))
    cfg.skillnet_cohort.segment_start = start
    cfg.skillnet_cohort.segment_end = start + 5
    # Outcome-blind common block schedule. Native Adam/policy/RNG are restored;
    # TextWorld workers are recreated, not claimed to be serializable.
    cfg.env.seed = int(spec['seed']) + 16 * (start // 5)
    cfg.data.seed = int(spec['seed'])
    cfg.ray_init.num_cpus = 32
    cfg.ray_init.object_store_memory = 8 * 2**30
    cfg.ray_init.include_dashboard = False
    if start:
        cfg.trainer.resume_mode = 'resume_path'
        cfg.trainer.resume_from_path = str(Path(root) / 'checkpoints' / f'global_step_{start}')
    return cfg


def execute(preparation, root, authorization, start):
    from omegaconf import OmegaConf
    from .assets import model_inventory
    permit = require_authorization(authorization, preparation, 'training')
    root = Path(root).resolve()
    if str(root) != permit.get('run_root') or len(permit['gpu_ids']) != 8:
        raise PermissionError('This local execution permit binds one new eight-GPU run')
    reject_legacy_overrides()
    backend = router_backend(read_json(Path(preparation).parent / 'spec.json'))
    if backend == 'external_llm' and not os.environ.get('SKILLNET_ROUTER_API_KEY'):
        raise PermissionError('Missing dedicated router environment credential')
    if backend == 'external_llm':
        profile = Path(permit['cost_profile'])
        if file_hash(profile) != permit['cost_profile_sha256']:
            raise PermissionError('Changed cost-cap profile')
        os.environ['SKILLNET_ROUTER_COST_PROFILE'] = str(profile)
    cfg = block_config(preparation, root, permit, start)
    recovery = permit.get('pre_optimizer_recovery')
    if recovery:
        if start != 0:
            raise PermissionError('Only the interrupted pre-optimizer first block can be recovered')
        from .rollout_recovery import verify_binding
        verify_binding(recovery, root, preparation)
        cfg.skillnet_cohort.pre_optimizer_recovery = recovery
        cfg.phase2.recovery_progress_root = recovery['attempt_dir']
    cfg.phase2.save_pre_forward_batch = True
    cfg.phase2.save_forward_outputs = True
    if model_inventory(cfg.alignment_run.model_path) != read_json(Path(preparation).parent / 'model.json'):
        raise ValueError('Changed frozen B0')
    if (root / 'checkpoints' / f'global_step_{start+5}').exists():
        raise FileExistsError('An endpoint exists; do not rerun the block')
    if start:
        from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
        if validate_full_checkpoint(root / 'checkpoints' / f'global_step_{start}')['world_size'] != 8:
            raise ValueError('No silent topology change/optimizer resharding')
    limits = permit['storage']
    disk_gate(root, limits['checkpoint_reserve_bytes'], **{
        name: limits[name] for name in ('minimum_free_bytes', 'maximum_run_bytes')})
    write_new_json(root / 'resource_limits.json', {**limits, 'vocab_size': 248320})
    segment_root = Path(recovery['attempt_dir']) if recovery else root
    write_new_json(segment_root / 'segments' / f'u{start:04d}-u{start+5:04d}.json', OmegaConf.to_container(cfg, resolve=True))
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(map(str, permit['gpu_ids'])),
                      ALFWORLD_DATA=cfg.skillnet_cohort.data_root,
                      PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    from verl.trainer.main_ppo import run_ppo
    import ray
    try:
        run_ppo(cfg)
    finally:
        ray.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('preparation', 'root', 'authorization'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        raise PermissionError('Explicit execution is required')
    execute(a.preparation, a.root, a.authorization, a.start)


if __name__ == '__main__':
    main()
