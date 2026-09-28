"""Offline versioned preparation/admission from retained engineering evidence."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from .common import REPO, file_hash, load_preparation, read_json, write_new_bytes, write_new_json
from .day_budget import INDEPENDENT_PROFILE, apply_budget
from .lossless_tensor import shuffled_profile


def project(evidence_root):
    root = Path(evidence_root)
    names = {
        'native': 'vllm-gpu-preflight-20260918-v4/verification-audit.json',
        'native_complete': 'vllm-gpu-preflight-20260918-v4/complete.json',
        'capture': 'vllm-capture-preflight-20260918-v1/verification-audit.json',
        'capture_complete': 'vllm-capture-preflight-20260918-v1/complete.json',
        'throughput': 'vllm-throughput-preflight-20260918-v2/timing-report.json',
        'router': 'vllm-router-calibration-20260918-v1/result.json',
        'compression': 'lossless-capture-calibration-20260918-v1/complete.json',
    }
    evidence = {key: read_json(root / path) for key, path in names.items()}
    for key in ('native_complete', 'capture', 'capture_complete', 'router', 'compression'):
        if evidence[key].get('status') != 'PASS':
            raise ValueError(f'Incomplete engineering prerequisite: {key}')
    if (not evidence['capture']['all_256_row_hashes_match'] or evidence['native_complete']['world_size'] != 8
            or evidence['capture_complete']['full_global_minibatch_rows'] != 128
            or evidence['throughput']['status'] != 'EPISODES_COMPLETE'):
        raise ValueError('Missing exact recording / eight-GPU / ALFWorld evidence')
    natural = [row for row in evidence['compression']['records'] if row['kind'] == 'natural_retokenized']
    if len(natural) != 24 or not all(row['bitwise_roundtrip_verified'] for row in natural):
        raise ValueError('Natural compression calibration is incomplete')
    timing = evidence['capture']['stage_max_seconds']
    # Deliberately no assumed cache hit or early episode termination in this
    # conditional high workload scenario. Not a mathematical/stochastic bound.
    rows = 5 * 128 * 50
    step_groups = rows / 128
    start_tokens = math.ceil(128 * evidence['throughput']['split_statistics']['train']['mean_output_tokens_per_episode'])
    parts = {
        'optimizer_native_full_length': step_groups * timing['full128_row_optimizer_update'],
        'old_and_reference_native_full_length': step_groups * (timing['old_exact_forward_and_io'] + timing['reference_forward']),
        'training_router_no_cache': rows * evidence['router']['mean_query_encode_seconds_per_state'],
        'training_generation': 250 * evidence['native']['stage_summary']['vllm_post_update_resync']['max_seconds'] / 2
                               + evidence['native']['stage_summary']['vllm_generation']['max_seconds'],
        'evaluation_and_monitoring': (548 + 134 + 2592 + 64) / 8 * max(
            value['max_episode_seconds'] for value in evidence['throughput']['split_statistics'].values()),
        'start_capture_compression': start_tokens / 8 * max(row['encode_and_verify_seconds'] / row['tokens'] for row in natural),
        'readout_six_forwards_scoring_and_decode_allowance': 6400 / 8 * (6 * 1.0 + 1.0)
                         + start_tokens / 8 * max(row['read_and_verify_seconds'] / row['tokens'] for row in natural),
        'cold_initialization_export_checkpoint_report_allowance': 2400,
    }
    # 50% token margin and a 0.655-ish encoded/raw bound vs measured <=0.449.
    token_allowance = math.ceil(start_tokens * 1.5)
    storage = {'full_vocab_compression': shuffled_profile(),
        'full_vocab_max_encoded_bytes_per_token': 650000, 'full_vocab_max_encoded_overhead_bytes': 65536,
        'minimum_free_bytes': 100 * 2**30, 'maximum_run_bytes': 760 * 2**30,
        'checkpoint_reserve_bytes': 80 * 2**30}
    peak = token_allowance * 650000 + 6400 * 65536 + 110 * 2**30
    estimate = math.ceil(sum(parts.values()))
    if estimate > 97200:
        raise ValueError(f'Whole-seed conditional estimate {estimate}s does not fit 27h admission allowance')
    result = {'status': 'PASS', 'projection_is_conditional_not_completion_guarantee': True,
        'batch_router_validated': True, 'native_eight_gpu_microbatch_validated': True,
        'exact_capture_validated': True, 'sharded_evaluation_validated': True,
        'projected_total_upper_seconds': estimate, 'projected_peak_run_bytes': peak,
        'component_seconds': parts, 'start_old_token_projection': start_tokens,
        'capacity_token_allowance': token_allowance,
        'natural_compression_ratio_max': max(r['encoded_bytes']/r['plain_torch_save_bytes'] for r in natural),
        'sources': [{'path': str(root / path), 'sha256': file_hash(root / path)} for path in names.values()],
        'evidence_limits': ['16 real episodes, not a full formal seed; no formal RL completed',
            'native 128-row test had 16 valid tokens with 4096+512 padded shape',
            'readout time includes a 1s/forward + 1s/scoring allowance, not an end-to-end readout timing',
            'short natural responses may change after RL; exact runtime capacity gate can stop the run',
            'no guarantee of three seeds in 12h or 30h; queue re-estimates after each entire completed seed',
            'real Ray orchestration and natural live/replay parity are checked during formal execution']}
    return result, storage


def sampling_audit(source, seeds):
    """Replay the installed TextWorld game iterator, without env construction."""
    from collections import Counter
    import numpy as np
    from textworld.gym.envs.utils import shuffled_cycle
    inventory = read_json(Path(source).parent / 'games.json')
    data = Path(inventory['data_root'])
    lookup = {row['game_id']: row['task_type'] for row in inventory['splits']['train']['games']}
    ordered = []
    for parent, _, files in os.walk(data / 'json_2.1.1/train', topdown=False):
        candidate = (Path(parent) / 'game.tw-pddl').relative_to(data).as_posix()
        if candidate in lookup:
            ordered.append(candidate)
    if len(ordered) != len(lookup) or set(ordered) != set(lookup):
        raise ValueError('Training pool changed')
    result = []
    for seed in seeds:
        groups = []
        for group in range(16):
            rng = np.random.RandomState(seed + group)
            games = list(ordered)
            rng.shuffle(games)
            stream = shuffled_cycle(games, rng=rng)
            groups.append([next(stream) for _ in range(5)])
        for update in range(1, 6):
            games = [row[update-1] for row in groups]
            result.append({'seed': seed, 'update': update, 'unique_games': len(set(games)),
                           'game_ids': games, 'task_counts': dict(Counter(lookup[game] for game in games))})
    return {'environment_started': False, 'kind': 'read_only_sampling_projection', 'records': result,
            'caveat': 'Depends on current runtime os.walk ordering; verify actual rollout game IDs too.'}


def prepare(source, output, run_root, receipt, evidence_root):
    source, output, run_root = map(lambda p: Path(p).resolve(), (source, output, run_root))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Independent preparation must be new; never rewrite a frozen setting')
    load_preparation(source)
    projection, storage = project(evidence_root)
    jobs = []
    for seed in (404, 505, 606):
        target = output / f'preparation-s{seed}-8gpu'
        manifest = read_json(source)
        spec = read_json(source.parent / 'spec.json')
        spec['seed'] = seed
        spec = apply_budget(spec, INDEPENDENT_PROFILE)
        for row in manifest['assets']:
            if row['path'] == 'spec.json':
                write_new_json(target / row['path'], spec)
            else:
                write_new_bytes(target / row['path'], (source.parent / row['path']).read_bytes())
            row['sha256'] = file_hash(target / row['path'])
        prep = target / 'manifest.json'
        write_new_json(prep, manifest)
        load_preparation(prep)
        admission = output / f'admission-s{seed}.json'
        write_new_json(admission, {**projection, 'preparation_sha256': file_hash(prep),
                                  'router_profile_sha256': spec['router_profile_sha256']})
        jobs.append({'preparation': str(prep), 'preparation_sha256': file_hash(prep),
            'run_root': str(run_root / f'seed-{seed}'), 'admission': str(admission), 'admission_sha256': file_hash(admission)})
    write_new_json(output / 'sampling-precheck.json', sampling_audit(source, [404, 505, 606]))
    sources = sorted({path for directory in ('skillnet_cohort', 'phase2', 'agent_system/memory')
                      for path in (REPO / directory).glob('*.py')})
    sources += [REPO / name for name in ('verl/workers/rollout/vllm_v1.py', 'verl/workers/actor/dp_actor.py',
        'verl/trainer/ppo/ray_trainer.py', 'agent_system/multi_turn_rollout/rollout_loop.py',
        'agent_system/environments/env_manager.py', 'agent_system/environments/env_package/alfworld/envs.py',
        'verl/workers/fsdp_workers.py', 'verl/utils/model.py', 'verl/utils/checkpoint/fsdp_checkpoint_manager.py')]
    write_new_json(output / 'cohort.json', {'schema_version': 'skillnet.phase12.independent_queue.v1',
        'approved': True, 'user_authority': '登记404/505/606，预算内依次执行；总计12h目标、原30h硬上限',
        'seeds': [404, 505, 606], 'root': str(run_root), 'jobs': jobs,
        'target_seconds': 43200, 'hard_limit_seconds': 108000, 'finish_reserve_seconds': 10800,
        'storage': storage, 'permit_template': {'operations': ['training', 'evaluation', 'readout', 'exports'],
            'gpu_ids': list(range(8)), 'router_max_api_calls': 0, 'router_max_local_calls': 202100,
            'reclaim_regenerable_full_vocab': True, 'upload_receipt': str(Path(receipt).resolve()),
            'upload_receipt_sha256': file_hash(receipt)},
        'source_sha256': {str(path.relative_to(REPO)): file_hash(path) for path in sources}})
    return {'preparation': str(output), 'formal_rl_started': False,
            'conditional_first_seed_hours': projection['projected_total_upper_seconds'] / 3600,
            'projected_peak_seed_gib': projection['projected_peak_run_bytes'] / 2**30}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'output', 'run-root', 'receipt', 'evidence-root'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.output, args.run_root, args.receipt, args.evidence_root))


if __name__ == '__main__':
    main()
