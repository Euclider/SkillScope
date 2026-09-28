"""Five-update blocks of the unchanged global GRPO recipe, with native resume."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .common import digest, require, strict_json, write_new
from .prepare import ARMS, load
from .routing import router_backend


def restrict_eval_games(base_env, allowed, data_root):
    root = Path(data_root).resolve()
    require(allowed and len(set(allowed)) == len(allowed), 'Invalid Seen evidence split')
    paths = [(root / item).resolve() for item in allowed]
    require(all(path.is_relative_to(root / 'json_2.1.1/valid_seen') for path in paths), 'Native monitor must use Seen evidence games')
    available = {Path(path).resolve(): path for path in base_env.game_files}
    require(all(path in available for path in paths), 'Monitor game missing from ALFWorld runtime')
    base_env.game_files = [available[path] for path in paths]
    base_env.num_games = len(paths)


def configuration(preparation, root, branch, bank_path, bank_sha256, start, resume_update=None):
    from omegaconf import OmegaConf
    from scripts.inspect_skillrl_alignment import load_config
    manifest, runtime = load(preparation)
    require(branch in ARMS and type(start) is int and start % 5 == 0
            and 0 <= start < runtime['optimizer_horizon_updates'], 'Invalid branch/block')
    from .bank import Bank
    bank = Bank.load(bank_path, bank_sha256)
    require(bank.branch_id == branch, 'Foreign training bank')
    root, assets = Path(root).resolve(), Path(preparation).resolve().parent
    cfg = load_config()
    seed = runtime['rl_seed']
    cfg.alignment_run = {'seed': seed, 'model_path': runtime['model_path'],
        'train_files': str(assets / 'datasets/train.parquet'), 'val_files': str(assets / 'datasets/seen-monitor.parquet'),
        'run_id': f'phase3-{branch}-s{seed}', 'output_dir': str(root / 'checkpoints'),
        'archive_dir': str(root / 'unused-phase1-archive'), 'router_cache': str(root / 'router.sqlite3'),
        'ray_temp_dir': f'/tmp/p3-{digest([str(root), start])[:12]}'}
    cfg.phase2 = {'enabled': False}
    cfg.env.phase1_archive.enabled = False
    cfg.phase3 = {'enabled': True, 'root': str(root), 'branch_id': branch, 'selector': ARMS[branch],
        'bank_sha256': bank_sha256, 'segment_start': start, 'segment_end': start + 5, 'data_root': runtime['data_root']}
    cfg.phase3.penultimate_recovery_checkpoint = True
    cfg.actor_rollout_ref.cohort_seed = seed
    cfg.actor_rollout_ref.rollout.seed = seed
    from skillnet_cohort.inference import apply_training, registration
    apply_training(cfg, registration(runtime['inference_profile']))
    cfg.actor_rollout_ref.actor.fsdp_config.cpu_shard_init = True
    cfg.actor_rollout_ref.ref.fsdp_config.cpu_shard_init = True
    cfg.skillnet_cohort = {'enabled': True, 'seed': seed, 'data_root': runtime['data_root']}
    # A registered common stream schedule, not the first five games replayed
    # at every restart. Policy/optimizer/RNG are restored from native shards.
    cfg.env.seed = seed + (start // 5) * 16
    cfg.data.seed = seed
    cfg.data.filter_overlong_prompts = False
    split = strict_json((assets / 'split.json').read_text())
    cfg.env.alfworld.allowed_eval_game_ids = [row['game_id'] for row in split['evidence_seen']]
    embedding = router_backend(runtime['router']) == 'skillrl_embedding_state'
    cfg.env.skills_only_memory.step_routing = {**runtime['router'], 'enabled': True,
        'backend': 'phase3_skillrl_embedding_state' if embedding else 'phase3_external_llm',
        'bank_path': str(Path(bank_path).resolve()), 'bank_sha256': bank_sha256,
        'cache_path': str(root / ('router-local.sqlite3' if embedding else 'router.sqlite3'))}
    cfg.trainer.n_gpus_per_node = len(runtime['gpu_ids'])
    cfg.trainer.total_training_steps = runtime['optimizer_horizon_updates']
    cfg.trainer.total_epochs = runtime['optimizer_horizon_updates']
    cfg.trainer.max_actor_ckpt_to_keep = 2
    cfg.trainer.max_critic_ckpt_to_keep = 2
    cfg.actor_rollout_ref.actor.optim.total_training_steps = runtime['optimizer_horizon_updates']
    # Bound Ray's idle worker/object-store footprint; no algorithm change.
    cfg.ray_init.num_cpus = 32
    cfg.ray_init.object_store_memory = 8 * 2**30
    cfg.ray_init.include_dashboard = False
    if resume_update is not None:
        require(type(resume_update) is int and start < resume_update < start + 5,
                'Recovery update must be inside the original five-update window')
        cfg.trainer.resume_mode = 'resume_path'
        cfg.trainer.resume_from_path = str(root / 'checkpoints' / f'global_step_{resume_update}')
    elif start:
        cfg.trainer.resume_mode = 'resume_path'
        cfg.trainer.resume_from_path = str(root / 'checkpoints' / f'global_step_{start}')
    OmegaConf.resolve(cfg)
    require(cfg.trainer.total_training_steps == runtime['optimizer_horizon_updates']
            and cfg.actor_rollout_ref.rollout.name == 'vllm_v1'
            and cfg.data.train_batch_size * cfg.env.rollout.n == 128,
            'Global recipe changed')
    return cfg


def execute(preparation, root, branch, bank_path, bank_sha256, start, approved, resume_update=None):
    require(approved, 'Use --execute only after reviewing the frozen preparation and budgets')
    _, runtime = load(preparation)
    from skillnet_cohort.runtime import disk_gate
    from skillnet_cohort.training import reject_legacy_overrides
    reject_legacy_overrides()
    if router_backend(runtime['router']) == 'external_llm':
        require(bool(os.environ.get('SKILLNET_ROUTER_API_KEY')), 'Missing SKILLNET_ROUTER_API_KEY')
    root = Path(root).resolve()
    disk_gate(root, runtime['storage']['checkpoint_reserve_bytes'],
              minimum_free_bytes=runtime['storage']['minimum_free_bytes'], maximum_run_bytes=runtime['storage']['maximum_run_bytes'])
    if start or resume_update is not None:
        from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
        checkpoint_update = start if resume_update is None else resume_update
        check = validate_full_checkpoint(root / 'checkpoints' / f'global_step_{checkpoint_update}')
        require(check['world_size'] == len(runtime['gpu_ids']), 'Cannot silently reshard optimizer state')
    require(not (root / 'checkpoints' / f'global_step_{start + 5}').exists(), 'Block already checkpointed; use runner resume')
    cfg = configuration(preparation, root, branch, bank_path, bank_sha256, start, resume_update)
    from omegaconf import OmegaConf
    segment_name = f'u{start:04d}-u{start+5:04d}'
    if resume_update is not None:
        segment_name += f'-resume-u{resume_update:04d}'
    write_new(root / 'segments' / f'{segment_name}.json', OmegaConf.to_container(cfg, resolve=True))
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(map(str, runtime['gpu_ids'])), ALFWORLD_DATA=runtime['data_root'],
                      PYTHONDONTWRITEBYTECODE='1')
    from verl.trainer.main_ppo import run_ppo
    import ray
    try:
        run_ppo(cfg)
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preparation', 'root', 'bank-path'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--branch', choices=ARMS, required=True)
    parser.add_argument('--bank-sha256', required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--resume-update', type=int)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    execute(args.preparation, args.root, args.branch, args.bank_path, args.bank_sha256,
            args.start, args.execute, args.resume_update)


if __name__ == '__main__':
    main()
