"""Append-only FP64 readout correction on retained 404/505 update batches.

No RL, environment interaction, editing, report replacement or cache writes.
The original evaluator and every source-frozen file remain unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

from .common import REPO, file_hash, read_json, write_new_bytes, write_new_json

SCHEMA = 'skillnet.numerical_readout.v1'
COHORT = REPO/'artifacts/phase12/skillnet37-independent-s404-505-606-v4'
F = COHORT/'all-first-calls-v1'
SOURCES = {404: F/'seed-404', 505: F/'followup-s505/seed-505'}
PAUSE = F/'recovery-v3/pause-20260921T115708Z.json'
EXTRA_SOURCES = ('phase2/stable_direction.py', 'skillnet_cohort/numerical_readout.py',
    'skillnet_cohort/numerical_readout_report.py', 'skillnet_cohort/numerical_readout_run.py')
KEYS = ['decision_id', 'control', 'response_token_offset']
FEATURE_KEYS = ['control', 'skill_id', 'context_id', 'phase']
EXTRA_MEANS = ('P_int_centered', 'D_centered_contribution',
    'P_centering_abs_error', 'C_numerator_centering_abs_error',
    'P_direct_dot_abs_error', 'old_probability_mass_error',
    'normalized_probability_mass_error', 'direction_sum_residual')


def binding(path, seed=None):
    path = Path(path).resolve(); plan = read_json(path)
    if (plan.get('schema_version') != SCHEMA or plan.get('approved') is not True
            or plan.get('seeds') != [404, 505] or plan.get('automatic_retry') is not False
            or plan.get('root') != str(COHORT/'numerical-readout-v1')
            or path != Path(plan['root'])/'plan.json'
            or plan.get('resume_505_only_after_404_report') is not True
            or plan.get('rewrite_original_reports') is not False):
        raise PermissionError('Only the explicitly approved numerical correction is allowed')
    from .seed_queue import verify_sources
    verify_sources(plan)
    for item in plan['bindings']:
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Bound source changed: '+item['path'])
    if seed is not None and (seed not in SOURCES or plan['source_roots'][str(seed)] != str(SOURCES[seed])):
        raise PermissionError('Unregistered seed')
    return plan


def paused_tree(plan):
    """Read-only identity check; live suspended reservations are not failures."""
    from .first_calls_defer import process_identity, descendants
    receipt = read_json(plan['pause']['path'])
    if file_hash(plan['pause']['path']) != plan['pause']['sha256']:
        raise ValueError('Pause authorization identity changed')
    result = []
    for expected in receipt['after']:
        current = process_identity(expected['pid'])
        if (current is None or any(current[k] != expected[k] for k in ('pid', 'start_ticks', 'command_sha256'))
                or current['state'] not in ('T', 't')):
            raise ProcessLookupError('Paused seed505 process changed or resumed unexpectedly')
        result.append(current)
    children = descendants([receipt['supervisor_pid']])
    if {p['pid'] for p in children} != {p['pid'] for p in result if p['pid'] != receipt['supervisor_pid']}:
        raise ValueError('Paused process ownership changed')
    if plan.get('pause_verification'):
        progress = read_json(plan['pause_verification']['path'])
        if file_hash(plan['pause_verification']['path']) != plan['pause_verification']['sha256']:
            raise ValueError('Pause progress receipt changed')
        for row in progress['endpoints']['u0000']['shards']:
            if file_hash(row['path']) != row['sha256']:
                raise ValueError('505 utility advanced while it was required to stay paused')
    return result


def disk(plan):
    from .first_calls_storage import disk_gate
    limits = plan['storage']
    return disk_gate(SOURCES[404], limits['checkpoint_reserve_bytes']+plan['new_result_reserve_bytes'],
        minimum_free_bytes=limits['minimum_free_bytes'], maximum_run_bytes=limits['maximum_run_bytes'])


def prepare(output, test_report):
    import xml.etree.ElementTree as ET
    from .seed_queue import verify_sources
    from .first_calls_port_recovery import binding as port_binding
    from .assets import model_inventory
    output = Path(output).resolve()
    if output != COHORT/'numerical-readout-v1' or output.exists():
        raise FileExistsError('A fresh numerical-readout-v1 is required')
    suites = list(ET.parse(test_report).getroot().iter('testsuite'))
    if (not suites or sum(int(s.attrib.get('tests', 0)) for s in suites) < 30
            or any(int(s.attrib.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped'))):
        raise ValueError('Passing numerical and execution regression required')
    port, _ = port_binding(F/'recovery-v3/plan.json')
    verify_sources(port)
    pause = read_json(PAUSE)
    if pause['state'] != 'PAUSED_BY_USER_SIGSTOP' or pause['process_count'] != 17:
        raise PermissionError('This workflow requires the explicitly paused 505 tree')
    pause_progress = PAUSE.parent/'pause-verification-20260921T115708Z.json'
    inputs = [PAUSE, pause_progress, F/'recovery-v3/plan.json', Path(test_report).resolve()]
    models = {}
    for seed, root in SOURCES.items():
        cfg = read_json(root/'protocol.json')
        if any(cfg['signals'][k] != 0 for k in ('minimum_nonzero_advantage_decisions', 'minimum_training_games', 'minimum_training_trajectories')):
            raise ValueError('All computable skills must remain eligible')
        if cfg['signals']['epsilon'] != 1e-12 or cfg['signals']['tau_C'] != 0:
            raise ValueError('Reviewed epsilon and C gate changed')
        for rel in ('protocol.json', 'manifest.json', 'signals/calibration.json', 'resource_limits.json',
                'batches/u0001/manifest.json', 'batches/u0001/training_batch.pt',
                'support/manifest.json', 'support/coverage.csv',
                'window_signals/u0000-u0005/committed.json',
                'window_signals/u0000-u0005/token_signals.parquet',
                'window_signals/u0000-u0005/skill_context_features.parquet',
                'window_signals/u0000-u0005/parameter_delta.json'):
            inputs.append(root/rel)
        inputs.extend(sorted((root/'window_signals/u0000-u0005').glob('*-shard-*')))
        inputs.extend(sorted((root/'window_signals/u0000-u0005').glob('shard-*.json')))
        inputs.extend(sorted((root/'support').glob('anchors-*.json')))
        original_plan = read_json(root.parent/'plan.json')
        models[str(seed)] = {}
        for endpoint in ('u0000', 'u0005'):
            inventory = model_inventory(root/'models'/endpoint)
            if inventory != original_plan['model_inventory'][endpoint]:
                raise ValueError('Original policy inventory changed')
            models[str(seed)][endpoint] = inventory
        inputs.append(root.parent/'plan.json')
    # Existing 404 labels remain immutable and explicitly retrospective. No
    # target labels are read by the scoring code or used to choose parameters.
    for folder in ('reports', 'window_metrics'):
        inputs.extend(p for p in (SOURCES[404]/folder).iterdir() if p.is_file())
    inputs.extend([SOURCES[404]/'complete.json', SOURCES[404]/'sealed.json'])
    plan = {'schema_version': SCHEMA, 'approved': True, 'root': str(output), 'seeds': [404, 505],
        'created_unix': time.time(), 'source_roots': {str(k): str(v) for k, v in SOURCES.items()},
        'user_authority': '2026-09-21 数值稳定性修正＋原版对照；先重算404，完成后再启动505及同规则修正',
        'source_sha256': {**port['source_sha256'], **{p: file_hash(REPO/p) for p in EXTRA_SOURCES}},
        'bindings': [{'path': str(p), 'sha256': file_hash(p)} for p in sorted(set(inputs))],
        'pause': {'path': str(PAUSE), 'sha256': file_hash(PAUSE)}, 'models': models,
        'pause_verification': {'path': str(pause_progress), 'sha256': file_hash(pause_progress)},
        'storage': read_json(SOURCES[404]/'resource_limits.json'), 'new_result_reserve_bytes': 8*2**30,
        'automatic_retry': False, 'resume_505_only_after_404_report': True,
        'resume_505_after_stable_scores_committed': True, 'rewrite_original_reports': False,
        'legacy_recorded_preserved': True, 'legacy_same_forward_recomputed': True,
        'forward_legacy_comparison_required': 'bitwise_same_scalar_readouts_including_NaN_equal',
        'forward_dtype': 'bfloat16 unchanged', 'readout_dtype': 'float64',
        'old_and_new_models_on_separate_gpus': True, 'logical_shards': 8, 'parallel_workers': 4,
        'within_shard_scoring_order': 'largest_actual_action_token_count_first_then_restore_original_row_order',
        'per_worker_visible_gpus': 2, 'minimum_free_gpu_mib': 15000,
        'P_centering_tolerance_abs': 1e-9, 'P_centering_tolerance_rel': 1e-11,
        'thresholds_validity_definitions_and_weights_unchanged': True,
        'KL_JS_and_activation_baselines_unchanged': True,
        'prior_target_results_available': {'404': True, '505': 'legacy_protocol_results_exist'},
        'scientific_status': 'retrospective_numerical_correction_with_preserved_comparators',
        'training_reexecuted': False, 'utility_rollouts_reexecuted': False,
        'external_api_calls': 0, 'seed606_started': False, 'hard_limit_seconds': None}
    paused_tree(plan)
    plan['capacity_admission'] = disk(plan)
    write_new_json(output/'plan.json', plan)
    print(json.dumps({'state': 'PREPARED_NOT_STARTED', 'plan': str(output/'plan.json'),
        'files_bound': len(inputs), 'source_files': len(plan['source_sha256'])}), flush=True)
    return plan


@contextmanager
def scoring_adapter(config):
    """Reuse the original inputs, forward and per-decision bookkeeping exactly."""
    import torch
    from phase2 import measure as base
    from phase2.stable_direction import token_signals
    from . import first_calls_measure as original
    old_forward, old_signals = original.forward, original.token_signals
    records = []

    def forward(model, *args):
        device = next(model.parameters()).device
        with torch.cuda.device(device):
            probabilities, hidden = base.forward(model, *args)
        return probabilities.to('cuda:0'), hidden

    def score(*args, **kwargs):
        value = token_signals(*args, tau_c=config['signals']['tau_C'],
            epsilon=config['signals']['epsilon'], tau_delta=config['_frozen_tau_delta'], include_legacy=True)
        records.append(value)
        return value

    original.forward, original.token_signals = forward, score
    try:
        yield records
    finally:
        original.forward, original.token_signals = old_forward, old_signals


def assert_legacy_equal(fresh, recorded):
    import numpy as np
    def equal(a, b):
        numeric = np.issubdtype(a.dtype, np.inexact) and np.issubdtype(b.dtype, np.inexact)
        return np.array_equal(a, b, equal_nan=bool(numeric))
    left = recorded.sort_values(KEYS).reset_index(drop=True)
    right = fresh.sort_values(KEYS).reset_index(drop=True)
    if len(left) != len(right) or not left[KEYS].equals(right[KEYS]):
        raise ValueError('Changed decision/control/action-position alignment')
    for name in left:
        legacy = 'legacy_'+name
        if legacy in right:
            a, b = left[name].to_numpy(), right[legacy].to_numpy()
            if not equal(a, b):
                raise ValueError('Same-forward legacy differs from recorded '+name)
    for name in ('skill_id', 'trajectory_id', 'game_id', 'action_token_id', 'advantage'):
        column = 'legacy_'+name if 'legacy_'+name in right else name
        if not equal(left[name].to_numpy(), right[column].to_numpy()):
            raise ValueError('Original training identity/advantage changed: '+name)


def ordered_decisions(selected, recorded_by_id):
    """Front-load the largest memory case; keep original indices for output."""
    return sorted(enumerate(selected),
        key=lambda item: (-len(recorded_by_id[item[1][1]['decision_id']]), item[0]))


def write_new_tensor(path, value):
    """Use the existing race-safe .publish-* path, never a disappearing .partial."""
    import io
    import torch
    stream = io.BytesIO()
    torch.save(value, stream)
    write_new_bytes(path, stream.getvalue())


def measure(path, seed, shard):
    import os
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from . import first_calls_measure as original
    from phase2.protocol import controls_by_skill
    plan = binding(path, seed); paused_tree(plan)
    if shard not in range(8) or os.environ.get('CUDA_VISIBLE_DEVICES') != f'{2*(shard%4)},{2*(shard%4)+1}':
        raise PermissionError('Wrong logical shard or GPU pair')
    root = SOURCES[seed]; out = Path(plan['root'])/f'seed-{seed}'
    if (out/f'shard-{shard}.json').exists() or (out/f'attempt-shard-{shard}.json').exists():
        raise FileExistsError('Never implicitly retry an attempted numerical shard')
    if seed == 505 and not (Path(plan['root'])/'seed-404/complete.json').exists():
        raise PermissionError('Corrected 404 report must finish before 505 correction')
    torch.set_num_threads(1)
    for gpu in range(2):
        if torch.cuda.mem_get_info(gpu)[0] < plan['minimum_free_gpu_mib']*2**20:
            raise MemoryError('Insufficient free GPU memory alongside paused505; do not evict it')
    write_new_json(out/f'attempt-shard-{shard}.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'plan_sha256': file_hash(path), 'seed': seed, 'shard': shard, 'automatic_retry': False})
    config = read_json(root/'protocol.json')
    config['_frozen_tau_delta'] = read_json(root/'signals/calibration.json')['tau_delta']
    batch = torch.load(root/'batches/u0001/training_batch.pt', map_location='cpu', weights_only=False)
    controls = controls_by_skill(config, REPO)
    rows, seen = [], set()
    for row, meta in enumerate(batch['metadata']):
        if meta['decision_id'] in seen:
            continue
        seen.add(meta['decision_id'])
        if meta['info'].get('selected_skill_id') in controls:
            rows.append((row, meta))
    selected = rows[shard::8]
    old = pd.read_parquet(root/'window_signals/u0000-u0005/token_signals.parquet')
    by_id = {key: frame for key, frame in old.groupby('decision_id', sort=False)}
    tokenizer = AutoTokenizer.from_pretrained(root/'models/u0000', local_files_only=True)
    models = [AutoModelForCausalLM.from_pretrained(root/'models'/endpoint,
        dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to(f'cuda:{rank}').eval()
        for rank, endpoint in enumerate(('u0000', 'u0005'))]
    frames, decisions = {}, {}
    ordered = ordered_decisions(selected, by_id)
    for progress, (index, (row, meta)) in enumerate(ordered):
        with scoring_adapter(config) as calls:
            fresh, dec, witness, _ = original.score_decision((root, *models), tokenizer,
                batch['tensors'], row, meta, controls[meta['info']['selected_skill_id']],
                float(batch['meta_info']['temperature']), repeat=False)
        if len(calls) != 4:
            raise ValueError('Expected primary and matched-backend signals for both controls')
        for arm, call in zip(('placebo', 'null'), (calls[1], calls[3])):
            for target, name in (('P_int_matched_backend', 'P_int'), ('D_matched_backend', 'D_contribution'),
                    ('C_upd_matched_backend', 'C_upd'), ('delta_norm_matched_backend', 'delta_norm')):
                fresh.loc[fresh.control == arm, 'legacy_'+target] = call['legacy_'+name].numpy()
        assert_legacy_equal(fresh, by_id[meta['decision_id']])
        tolerance = plan['P_centering_tolerance_abs']+plan['P_centering_tolerance_rel']*fresh.P_int.abs()
        if not (fresh.P_centering_abs_error <= tolerance).all():
            raise ValueError('Stable P centering identity failed; do not auto-adjust tolerance')
        frames[index] = fresh; decisions[index] = dec
        if index == 0:
            write_new_tensor(out/f'witness-shard-{shard}.pt', witness)
        print(f'seed{seed} shard{shard} {progress+1}/{len(selected)} original_row={row} legacy_exact=True '
              f'P_identity_max={fresh.P_centering_abs_error.max():.3g}', flush=True)
    tokens = pd.concat([frames[i] for i in range(len(selected))], ignore_index=True)
    decision_table = pd.concat([decisions[i] for i in range(len(selected))], ignore_index=True)
    write_new_bytes(out/f'tokens-shard-{shard}.parquet', tokens.to_parquet(index=False))
    write_new_bytes(out/f'decisions-shard-{shard}.parquet', decision_table.to_parquet(index=False))
    write_new_json(out/f'shard-{shard}.json', {'seed': seed, 'shard': shard, 'decisions': len(selected),
        'tokens': len(tokens), 'tokens_sha256': file_hash(out/f'tokens-shard-{shard}.parquet'),
        'decisions_sha256': file_hash(out/f'decisions-shard-{shard}.parquet'),
        'witness_sha256': file_hash(out/f'witness-shard-{shard}.pt'),
        'all_legacy_scalar_signals_exact': True, 'endpoints_on_same_original_batch': True,
        'max_P_centering_error': float(tokens.P_centering_abs_error.max()),
        'finished_unix': time.time(), 'peak_gpu_allocated_bytes': [torch.cuda.max_memory_allocated(i) for i in range(2)]})


def aggregate(path, seed):
    import numpy as np
    import pandas as pd
    from phase2.aggregate import aggregate_features
    plan = binding(path, seed); root = SOURCES[seed]; out = Path(plan['root'])/f'seed-{seed}'
    if (out/'committed.json').exists():
        raise FileExistsError('No duplicate aggregation')
    frames, decisions = [], []
    for shard in range(8):
        receipt = read_json(out/f'shard-{shard}.json')
        for kind in ('tokens', 'decisions'):
            if file_hash(out/f'{kind}-shard-{shard}.parquet') != receipt[kind+'_sha256']:
                raise ValueError('Numerical shard changed')
        if not receipt['all_legacy_scalar_signals_exact']:
            raise ValueError('Original comparator was not reproduced')
        frames.append(pd.read_parquet(out/f'tokens-shard-{shard}.parquet'))
        decisions.append(pd.read_parquet(out/f'decisions-shard-{shard}.parquet'))
    tokens = pd.concat(frames, ignore_index=True); dec = pd.concat(decisions, ignore_index=True)
    old = pd.read_parquet(root/'window_signals/u0000-u0005/token_signals.parquet')
    assert_legacy_equal(tokens, old)
    if tokens.duplicated(KEYS).any():
        raise ValueError('Duplicate numerical token')
    config = read_json(root/'protocol.json'); tau = read_json(root/'signals/calibration.json')['tau_delta']
    features, tokens = aggregate_features(tokens, dec, config, 5, tau)
    # Original aggregator reconstructs the RAW gate. Preserve centered gates
    # separately, with the same epsilon/tau values and no result-dependent fit.
    for i, row in features.iterrows():
        q = tokens[(tokens.control == row.control) & (tokens.skill_id == row.skill_id)
                   & (tokens.context_id == row.context_id)]
        if row.phase != 'all':
            q = q[q.phase == row.phase]
        for name in EXTRA_MEANS:
            features.loc[i, name] = float(q[name].mean()) if len(q) else np.nan
        features.loc[i, 'gate_centered_coverage'] = float(q.gate_centered.mean()) if len(q) else np.nan
        features.loc[i, 'raw_centered_gate_changes'] = int((q.gate != q.gate_centered).sum())
        features.loc[i, 'legacy_direction_valid_changes'] = int((q.direction_valid != q.legacy_direction_valid).sum())
        features.loc[i, 'legacy_gate_changes'] = int((q.gate != q.legacy_gate).sum())
    features['start_update'] = 0; features['window_horizon'] = 5; features['direction_batch_update'] = 1
    old_f = pd.read_parquet(root/'window_signals/u0000-u0005/skill_context_features.parquet')
    left = old_f.sort_values(FEATURE_KEYS).reset_index(drop=True)
    right = features.sort_values(FEATURE_KEYS).reset_index(drop=True)
    if not left[FEATURE_KEYS+['supported', 'token_count', 'decision_count']].equals(right[FEATURE_KEYS+['supported', 'token_count', 'decision_count']]):
        raise ValueError('Numerical correction changed candidate coverage or batch membership')
    write_new_bytes(out/'token_signals.parquet', tokens.to_parquet(index=False))
    write_new_bytes(out/'skill_context_features.parquet', features.to_parquet(index=False))
    comparison = features.merge(old_f, on=FEATURE_KEYS, suffixes=('_stable', '_legacy'), validate='one_to_one')
    write_new_bytes(out/'skill_score_comparison.csv', comparison.to_csv(index=False).encode())
    write_new_json(out/'committed.json', {'seed': seed, 'created_unix': time.time(),
        'features_sha256': file_hash(out/'skill_context_features.parquet'),
        'token_signals_sha256': file_hash(out/'token_signals.parquet'),
        'plan_sha256': file_hash(path), 'all_legacy_scalar_signals_exact': True,
        'tokens_with_controls': len(tokens), 'unique_decisions': int(tokens.decision_id.nunique()),
        'original_candidate_coverage_unchanged': True, 'target_labels_used_for_scoring': False,
        'earlier_target_labels_exist': True, 'thresholds_unchanged': True,
        'max_P_centering_error': float(tokens.P_centering_abs_error.max()),
        'raw_centered_gate_changes': int((tokens.gate != tokens.gate_centered).sum()),
        'legacy_gate_changes': int((tokens.gate != tokens.legacy_gate).sum()),
        'legacy_direction_valid_changes': int((tokens.direction_valid != tokens.legacy_direction_valid).sum())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path); parser.add_argument('--prepare', type=Path)
    parser.add_argument('--test-report', type=Path); parser.add_argument('--seed', type=int, choices=(404, 505))
    mode = parser.add_mutually_exclusive_group(); mode.add_argument('--measure', action='store_true')
    mode.add_argument('--aggregate', action='store_true'); parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    if args.prepare:
        if args.plan or args.measure or args.aggregate or not args.test_report:
            parser.error('Preparation is a separate operation')
        prepare(args.prepare, args.test_report); return
    if not args.plan or args.seed is None:
        parser.error('Explicit plan and seed required')
    if args.measure and args.shard is not None:
        measure(args.plan, args.seed, args.shard)
    elif args.aggregate:
        aggregate(args.plan, args.seed)
    else:
        parser.error('Select one operation')


if __name__ == '__main__':
    main()
