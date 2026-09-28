"""Offline, no-clobber registration of portable six-arm Phase3 inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from skillnet_cohort.assets import game_inventory, model_inventory, write_placeholders, TASKS
from skillnet_cohort.common import file_hash
from .api import APIConfig
from .bank import Bank
from .common import digest, finite, positive_int, require, strict_json, write_new
from .routing import router_backend

ROOT = Path(__file__).resolve().parents[1]
ARMS = {'readout_d': 'reward_sign_balance', 'skillrl_failure': 'failure_driven',
        'readout_magnitude': 'centered_magnitude', 'readout_gated_d': 'legacy_gated_d',
        'readout_p': 'negative_p', 'readout_c': 'positive_c'}


def validate_runtime(value):
    required = {'model_path', 'data_root', 'gpu_ids', 'rl_seed', 'optimizer_horizon_updates',
                'initial_stop_update', 'inference_profile',
                'router', 'editor', 'gate_games_per_task',
                'gate_tolerance_pp', 'eval_seeds', 'max_evidence_trajectories', 'storage', 'readout_parity_atol'}
    require(isinstance(value, dict) and set(value) == required, 'Runtime template contains missing/unknown fields')
    for name in ('model_path', 'data_root'):
        require(isinstance(value[name], str) and Path(value[name]).is_absolute(), f'Provide an absolute {name}')
    gpu = value['gpu_ids']
    require(isinstance(gpu, list) and len(gpu) in (4, 8) and len(set(gpu)) == len(gpu)
            and all(type(item) is int and item >= 0 for item in gpu), 'Explicit 4/8 GPU IDs required')
    positive_int(value['rl_seed'], 'independent RL seed', zero=True)
    require(value['rl_seed'] not in (404, 505, 606), 'Phase3 seed must be independent of score-selection seeds')
    require(type(value['optimizer_horizon_updates']) is int and value['optimizer_horizon_updates'] == 150,
            'The frozen optimizer horizon is 150 updates')
    require(type(value['initial_stop_update']) is int and value['initial_stop_update'] == 20,
            'The pre-registered first pause is U20')
    from skillnet_cohort.inference import PHASE3_PROFILE, PHASE3_MEMORY_PROFILE, PHASE3_MEMORY_PROFILE_V3, registration
    require(Path(value['inference_profile']).resolve() in
            {PHASE3_PROFILE.resolve(), PHASE3_MEMORY_PROFILE.resolve(), PHASE3_MEMORY_PROFILE_V3.resolve()},
            'Phase3 requires a registered vLLM profile')
    binding = registration(value['inference_profile'])
    require(binding['settings']['seed'] == value['rl_seed'], 'Inference/learning seed mismatch')
    embedding = router_backend(value['router']) == 'skillrl_embedding_state'
    require(embedding, 'This registered Phase3 protocol requires the frozen local embedding router')
    from .embedding_routing import batch_execution, validate_settings
    validate_settings(value['router'])
    execution = batch_execution(value['router'])
    if execution and value['router']['device'].startswith('cuda:'):
        require(execution['shared_gpu_physical_id'] in gpu,
                'Shared router GPU must be one of the registered Phase3 GPUs')
    for stage, model in [('editor', 'gpt-5.5')]:
        require(set(value[stage]) == {'max_input_tokens', 'max_completion_tokens', 'max_api_calls'}, 'Invalid API limits')
        APIConfig(stage=stage, model=model, **value[stage])
        positive_int(value[stage]['max_api_calls'], f'{stage} API calls')
    positive_int(value['gate_games_per_task'], 'gate games per task')
    tolerance = finite(value['gate_tolerance_pp'], 'gate tolerance')
    require(0 <= tolerance <= 100, 'Invalid gate tolerance')
    seeds = value['eval_seeds']
    require(isinstance(seeds, list) and seeds and len(set(seeds)) == len(seeds), 'Explicit unique evaluation seeds required')
    for seed in seeds:
        positive_int(seed, 'evaluation seed', zero=True)
    require(1 <= positive_int(value['max_evidence_trajectories'], 'evidence cap') <= 10, 'At most 10 evidence trajectories')
    require(set(value['storage']) == {'maximum_run_bytes', 'minimum_free_bytes', 'checkpoint_reserve_bytes'}, 'Invalid storage budget')
    for name, amount in value['storage'].items():
        positive_int(amount, name)
    require(finite(value['readout_parity_atol'], 'parity tolerance') > 0, 'Positive parity tolerance required')
    return value


def partition_seen(rows, per_task, seed=707):
    """Outcome-blind deterministic stratification; no Unseen input."""
    gate = []
    for task in sorted(TASKS):
        pool = sorted((row for row in rows if row['task_type'] == task),
                      key=lambda row: digest(['phase3-gate-v2', seed, row['game_id']]))
        require(len(pool) > per_task, 'Gate would exhaust an entire Seen task stratum')
        gate.extend(pool[:per_task])
    ids = {row['game_id'] for row in gate}
    evidence = [row for row in rows if row['game_id'] not in ids]
    return sorted(gate, key=lambda row: row['game_id']), sorted(evidence, key=lambda row: row['game_id'])


def prepare(runtime, output):
    runtime = validate_runtime(dict(runtime))
    output = Path(output).resolve()
    runtime = {**runtime, 'model_path': str(Path(runtime['model_path']).resolve()),
               'data_root': str(Path(runtime['data_root']).resolve())}
    embedding = router_backend(runtime['router']) == 'skillrl_embedding_state'
    if embedding:
        from agent_system.memory.skillrl_embedding_router import verify_snapshot
        from .embedding_routing import validate_settings
        _, files = validate_settings(runtime['router'])
        verify_snapshot(runtime['router']['model_path'], files)
    write_new(output / 'runtime.json', runtime)
    setting = strict_json((ROOT / 'configs/phase3_setting_embedding_v4.json').read_text())
    require(setting['schema_version'] == 'skillrl.phase3.setting.embedding.v4'
            and setting['initialization']['rl_seed'] == runtime['rl_seed']
            and setting['training']['optimizer_horizon_updates'] == runtime['optimizer_horizon_updates']
            and setting['training']['first_execution_stop_update'] == runtime['initial_stop_update']
            and setting['initialization']['execution_order'] == list(ARMS)
            and list(setting['arms']) == list(ARMS)
            and setting['readout']['skill_level_edit_threshold'] is None
            and {name: row['selector'] for name, row in setting['arms'].items()} == ARMS,
            'Runtime and registered Phase3 v3 setting disagree')
    write_new(output / 'setting.json', setting)
    inventory = game_inventory(runtime['data_root'])
    write_new(output / 'games.json', inventory)
    model = model_inventory(runtime['model_path'])
    # Weight identity does not depend on where a server mounted the same files.
    model['identity_sha256'] = digest(model['files'])
    write_new(output / 'model.json', model)
    gate, evidence = partition_seen(inventory['splits']['valid_seen']['games'], runtime['gate_games_per_task'], runtime['rl_seed'])
    write_new(output / 'split.json', {'gate': gate, 'evidence_seen': evidence, 'eval_seeds': runtime['eval_seeds'],
                                    'selection': 'hash-stratified-without-outcomes', 'unseen_used': False})
    write_placeholders(output / 'datasets/train.parquet', 3553, 'train')
    write_placeholders(output / 'datasets/seen-monitor.parquet', 64, 'valid_seen_evidence_only')
    banks = {}
    for branch in ARMS:
        bank = Bank.initial(branch)
        bank.save(output / 'banks')
        banks[branch] = bank.manifest_sha256
    assets = [{'path': path.relative_to(output).as_posix(), 'sha256': file_hash(path)}
              for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'manifest.json']
    manifest = {'schema_version': 'skillrl.phase3.preparation.v2', 'arms': ARMS, 'initial_banks': banks,
                'assets': assets, 'execution_started': False, 'external_api_calls': 0}
    write_new(output / 'manifest.json', manifest)
    return manifest


def load(path):
    path = Path(path).resolve()
    manifest = strict_json(path.read_text())
    require(manifest['schema_version'] == 'skillrl.phase3.preparation.v2' and manifest['arms'] == ARMS, 'Foreign preparation')
    for row in manifest['assets']:
        target = (path.parent / row['path']).resolve()
        require(target.is_relative_to(path.parent) and file_hash(target) == row['sha256'], 'Changed preparation asset')
    runtime = validate_runtime(strict_json((path.parent / 'runtime.json').read_text()))
    return manifest, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(strict_json(args.runtime.read_text()), args.output)
    print({'status': 'OFFLINE_PREPARED_NOT_RUN', 'arms': list(result['arms']), 'manifest': str(args.output / 'manifest.json')})


if __name__ == '__main__':
    main()
