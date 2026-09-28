"""Seed505 frozen 285-readout replication; GPU geometry, then label-blind scores."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .common import REPO, file_hash, read_json, write_new_bytes, write_new_json
from .first_calls_storage import disk_gate
from .reward_variant_analysis import add_references, audit_legacy
from .reward_variants import label_blind_scales, validate_tokens

COHORT = REPO/'artifacts/phase12/skillnet37-independent-s404-505-606-v4'
SOURCE = COHORT/'numerical-readout-release-v1/seed-505'
TRAJECTORY = COHORT/'all-first-calls-v1/followup-s505/seed-505'
FROZEN = COHORT/'factorized-reward-s404-v1'
OUTPUT = COHORT/'realized-and-scores-s505-v1'
MODULE = 'skillnet_cohort.seed505_readout'


def limits():
    return read_json(TRAJECTORY/'resource_limits.json')


def capacity(extra=0):
    cfg = limits()
    return disk_gate(COHORT, cfg['checkpoint_reserve_bytes']+extra,
        minimum_free_bytes=cfg['minimum_free_bytes'], maximum_run_bytes=cfg['maximum_run_bytes'])


def record(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_hash(path)}


def verify(rows):
    for item in rows:
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Frozen input changed: '+item['path'])


def scope(output):
    output = Path(output).resolve()
    if output.parent != COHORT or not output.name.startswith('realized-and-scores-s505-v'):
        raise PermissionError('Only a new scoped seed505 readout directory is allowed')
    return output


def prepare(output):
    from .assets import model_inventory
    output = scope(output)
    if output.exists():
        raise FileExistsError('Never overwrite an attempted readout')
    if read_json(SOURCE/'complete.json')['status'] != 'complete':
        raise ValueError('Seed505 numerical input is incomplete')
    if read_json(TRAJECTORY/'complete.json')['status'] != 'complete':
        raise ValueError('Seed505 O/P/N source is incomplete')
    if read_json(COHORT/'seed-505/complete.json')['status'] != 'complete':
        raise ValueError('Seed505 RL endpoint is incomplete')
    numerical = read_json(SOURCE/'complete.json')
    if numerical['provenance_sha256'] != file_hash(SOURCE/'report-provenance.json'):
        raise ValueError('Numerical source changed')
    t = pd.read_parquet(SOURCE/'token_signals.parquet')
    validate_tokens(t)
    ledgers = sorted((TRAJECTORY/'old_logprobs/u0001').glob('rank-*.jsonl'))
    old = {}
    for path in ledgers:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = int(row['row_index'])
            item = {'row_index': key, 'path': str(TRAJECTORY/'old_logprobs/u0001'/f'row-{key:06d}.pt'),
                    'sha256': row['compression']['encoded_sha256']}
            if key in old and old[key] != item:
                raise ValueError('Conflicting archived OLD probability row')
            old[key] = item
    if len(old) < t.decision_id.nunique() or not all(Path(v['path']).is_file() for v in old.values()):
        raise ValueError('Incomplete OLD probability archive')
    model_inventory_u0 = model_inventory(TRAJECTORY/'models/u0000')
    model_inventory_u5 = model_inventory(TRAJECTORY/'models/u0005')
    inputs = [SOURCE/n for n in ('complete.json', 'report-provenance.json',
        'committed.json', 'token_signals.parquet', 'skill_context_features.parquet',
        'ranking-snapshots.json', 'reports/scores_and_gold.csv')]
    inputs += [TRAJECTORY/n for n in ('complete.json', 'protocol.json',
        'resource_limits.json', 'signals/calibration.json', 'batches/u0001/training_batch.pt')]
    inputs += [FROZEN/n for n in ('complete.json', 'registry.csv')]
    inputs += ledgers
    sources = [REPO/p for p in ('skillnet_cohort/seed505_readout.py',
        'skillnet_cohort/realized_reward_measure.py', 'skillnet_cohort/realized_reward.py',
        'skillnet_cohort/realized_reward_analysis.py', 'skillnet_cohort/reward_variants.py',
        'skillnet_cohort/reward_variant_analysis.py',
        'skillnet_cohort/reward_variant_constant_fix.py', 'skillnet_cohort/factorized_reward.py')]
    plan = {'version': 'skillnet.seed505_readout_285.v1', 'seed': 505,
        'output': str(output), 'source': str(SOURCE), 'trajectory_root': str(TRAJECTORY),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'authority': 'User requested independent RL seed505, all D variants, -deltaM and abs(deltaM)',
        'expected_token_control_rows': len(t), 'expected_decisions': int(t.decision_id.nunique()),
        'score_registry_path': str(FROZEN/'registry.csv'), 'score_columns': 285,
        'new_training': False, 'new_environment_rollouts': 0, 'new_api_calls': 0,
        'forward_dtype': 'BF16_SDPA_original', 'geometry_dtype': 'FP64_16_token_chunks',
        'model_inventory': {'u0000': model_inventory_u0, 'u0005': model_inventory_u5},
        'old_tensor_files': [old[k] for k in sorted(old)],
        'inputs': [record(p) for p in inputs], 'sources': [record(p) for p in sources],
        'new_result_reserve_bytes': 8*2**30, 'automatic_retry': False,
        'cohort_storage_limits': limits()}
    plan['capacity_at_preparation'] = capacity(plan['new_result_reserve_bytes'])
    write_new_json(output/'plan.json', plan)
    print(json.dumps({'status': 'PREPARED_NOT_STARTED', 'output': str(output),
        'decisions': plan['expected_decisions'], 'tokens': len(t), 'score_columns': 285}), flush=True)


def binding(output, *, models=False):
    from .assets import model_inventory
    output = scope(output)
    plan = read_json(output/'plan.json')
    if (plan['version'] != 'skillnet.seed505_readout_285.v1' or plan['seed'] != 505
            or plan['output'] != str(output) or plan['source'] != str(SOURCE)
            or plan['trajectory_root'] != str(TRAJECTORY) or plan['score_columns'] != 285
            or plan['new_training'] or plan['new_environment_rollouts'] or plan['new_api_calls']
            or plan['automatic_retry']):
        raise PermissionError('Changed seed505 readout scope')
    verify(plan['inputs']+plan['sources'])
    if models:
        for endpoint, expected in plan['model_inventory'].items():
            if model_inventory(TRAJECTORY/'models'/endpoint) != expected:
                raise ValueError('Policy model changed: '+endpoint)
    return plan


def gpu_pair(shard):
    if shard not in range(8):
        raise ValueError('Exactly eight logical shards')
    return f'{2*(shard%4)},{2*(shard%4)+1}'


def worker(output, shard):
    """Adapt the seed404 scorer without changing the frozen scoring arithmetic."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from phase2.protocol import controls_by_skill
    from . import first_calls_measure as original
    from .numerical_readout import ordered_decisions, write_new_tensor
    from .realized_reward_measure import adapter
    from .realized_reward import FIELDS, KEYS, assert_same_scalars

    output = scope(output); plan = binding(output)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_pair(shard):
        raise PermissionError('Wrong registered GPU pair')
    if (output/f'attempt-shard-{shard}.json').exists():
        raise FileExistsError('No automatic retry of an attempted shard')
    capacity(plan['new_result_reserve_bytes'])
    torch.set_num_threads(1)
    for gpu in range(2):
        if torch.cuda.mem_get_info(gpu)[0] < 28000*2**20:
            raise MemoryError('Do not share a registered GPU with another job')
    write_new_json(output/f'attempt-shard-{shard}.json', {'pid': os.getpid(), 'shard': shard,
        'started_unix': time.time(), 'plan_sha256': file_hash(output/'plan.json')})
    config = read_json(TRAJECTORY/'protocol.json')
    config['_frozen_tau_delta'] = read_json(TRAJECTORY/'signals/calibration.json')['tau_delta']
    batch = torch.load(TRAJECTORY/'batches/u0001/training_batch.pt', map_location='cpu', weights_only=False)
    controls = controls_by_skill(config, REPO)
    rows, seen = [], set()
    for row, meta in enumerate(batch['metadata']):
        if meta['decision_id'] in seen:
            continue
        seen.add(meta['decision_id'])
        if meta['info'].get('selected_skill_id') in controls:
            rows.append((row, meta))
    selected = rows[shard::8]
    recorded = pd.read_parquet(SOURCE/'token_signals.parquet')
    by_id = {key: frame for key, frame in recorded.groupby('decision_id', sort=False)}
    if set(by_id) != {m['decision_id'] for _, m in rows}:
        raise ValueError('Actual batch support differs from committed readout')
    tokenizer = AutoTokenizer.from_pretrained(TRAJECTORY/'models/u0000', local_files_only=True)
    models = [AutoModelForCausalLM.from_pretrained(TRAJECTORY/'models'/endpoint,
        dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to(f'cuda:{rank}').eval()
        for rank, endpoint in enumerate(('u0000', 'u0005'))]
    expected_old = {r['row_index']: r['sha256'] for r in plan['old_tensor_files']}
    frames, checks = {}, []
    for count, (index, (row, meta)) in enumerate(ordered_decisions(selected, by_id)):
        old_hash = file_hash(TRAJECTORY/'old_logprobs/u0001'/f'row-{row:06d}.pt')
        if old_hash != expected_old[row]:
            raise ValueError('OLD probability row changed')
        with adapter(config) as calls:
            fresh, _, witness, _ = original.score_decision((TRAJECTORY, *models), tokenizer,
                batch['tensors'], row, meta, controls[meta['info']['selected_skill_id']],
                float(batch['meta_info']['temperature']), repeat=False)
        if len(calls) != 4:
            raise ValueError('Both primary and matched backend checks required')
        assert_same_scalars(fresh, by_id[meta['decision_id']])
        if not (fresh.P_centering_abs_error <= 1e-9+1e-11*fresh.P_int.abs()).all():
            raise ValueError('Stable P identity failed')
        frames[index] = fresh[KEYS+list(FIELDS)].copy()
        checks.append({'decision_id': meta['decision_id'], 'row_index': row,
            'old_tensor_sha256': old_hash, 'tokens': len(fresh), 'stable_scalars_exact': True})
        if index == 0:
            small = {k: (v[:16].clone() if isinstance(v, torch.Tensor) else v)
                     for k, v in witness.items()}
            write_new_tensor(output/f'witness-shard-{shard}.pt', small)
        print(f'seed505 realized shard{shard} {count+1}/{len(selected)} '
              f'original_row={row} tokens={len(fresh)} stable_exact=True', flush=True)
    result = pd.concat([frames[i] for i in range(len(selected))], ignore_index=True)
    write_new_bytes(output/f'tokens-shard-{shard}.parquet', result.to_parquet(index=False))
    write_new_json(output/f'input-audit-shard-{shard}.json', checks)
    write_new_json(output/f'shard-{shard}.json', {'seed': 505, 'shard': shard,
        'decisions': len(selected), 'tokens': len(result),
        'tokens_sha256': file_hash(output/f'tokens-shard-{shard}.parquet'),
        'input_audit_sha256': file_hash(output/f'input-audit-shard-{shard}.json'),
        'witness_sha256': file_hash(output/f'witness-shard-{shard}.pt'),
        'all_stable_scalar_signals_exact': True, 'target_labels_read': False,
        'finished_unix': time.time(), 'peak_gpu_allocated_bytes':
            [torch.cuda.max_memory_allocated(i) for i in range(2)]})


def finalize(output):
    from .realized_reward_analysis import joined_tokens
    from .realized_reward import aggregate_scores as realized_scores, registry as realized_registry
    from .factorized_reward import aggregate_scores as factor_scores, registry as factor_registry
    from . import reward_variant_constant_fix as corrected
    from . import reward_variant_analysis as base
    output = scope(output); plan = binding(output, models=True)
    t = joined_tokens(output, plan)
    validate_tokens(t)
    write_new_bytes(output/'token_signals.parquet', t.to_parquet(index=False))
    raw = pd.read_parquet(SOURCE/'token_signals.parquet')
    scales = {c: label_blind_scales(raw[raw.control == c]) for c in ('placebo', 'null')}
    write_new_json(output/'label-blind-scales.json', scales)
    base_scores, _ = corrected.correct_scores(raw, scales)
    audit = audit_legacy(base_scores, pd.read_parquet(SOURCE/'skill_context_features.parquet'))
    reference = pd.read_csv(SOURCE/'reports/scores_and_gold.csv',
                            keep_default_na=False, na_values=[''])
    base_scores = add_references(base_scores, [], reference)
    real = realized_scores(t)
    factor, components = factor_scores(t)
    scores = pd.concat([base_scores, real, factor], ignore_index=True)
    keys = ['control', 'context_id', 'phase', 'skill_id', 'score']
    if scores.duplicated(keys).any():
        raise ValueError('Duplicate readout identity')
    registry = pd.read_csv(FROZEN/'registry.csv', keep_default_na=False, na_values=[''])
    if (len(registry) != 285 or scores.score.nunique() != 285
            or set(scores.score) != set(registry.score)
            or set(real.score) != {m['score'] for m in realized_registry()}
            or set(factor.score) != {m['score'] for m in factor_registry()}):
        raise ValueError('The full frozen 285 readout registry was not reproduced')
    write_new_bytes(output/'registry.csv', (FROZEN/'registry.csv').read_bytes())
    write_new_bytes(output/'skill_scores.csv', scores.to_csv(index=False).encode())
    write_new_bytes(output/'factor-components.parquet', components.to_parquet(index=False))
    write_new_json(output/'score-commit.json', {'score_sha256': file_hash(output/'skill_scores.csv'),
        'registry_sha256': file_hash(output/'registry.csv'),
        'all_285_frozen_formulas': True, 'computed_before_new_gold': True,
        'new_gold_labels_read': False, 'seed505_old_labels_previously_visible': True,
        'stable_scalar_audit': audit, 'exact_constant_cells_corrected': len(corrected.FIX_ROWS)})
    paths = [output/n for n in ('plan.json', 'token_signals.parquet', 'skill_scores.csv',
        'registry.csv', 'factor-components.parquet', 'score-commit.json', 'label-blind-scales.json')]
    paths += [output/f'{kind}-shard-{i}.{extension}' for i in range(8)
              for kind, extension in (('tokens', 'parquet'), ('input-audit', 'json'), ('witness', 'pt'))]
    paths += [output/f'shard-{i}.json' for i in range(8)]
    write_new_json(output/'provenance.json', {'plan_sha256': file_hash(output/'plan.json'),
        'files': [{'path': str(p.relative_to(output)), 'sha256': file_hash(p)} for p in paths]})
    write_new_json(output/'complete.json', {'status': 'complete', 'seed': 505,
        'tokens': len(t), 'readout_columns': 285,
        'provenance_sha256': file_hash(output/'provenance.json'),
        'new_environment_rollouts': 0, 'new_api_calls': 0})


def child_environment(gpu=''):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTHONPATH=str(REPO)+os.pathsep+env.get('PYTHONPATH', ''))
    return env


def run(output):
    output = scope(output); plan = binding(output, models=True)
    if (output/'run-intent.json').exists():
        raise FileExistsError('No implicit retry')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise PermissionError('GPUs must be idle')
    capacity(plan['new_result_reserve_bytes'])
    write_new_json(output/'run-intent.json', {'pid': os.getpid(),
        'started_unix': time.time(), 'plan_sha256': file_hash(output/'plan.json')})
    try:
        for wave in range(2):
            children = []
            for shard in range(4*wave, 4*wave+4):
                log = output/'logs'/f'shard-{shard}.log'; log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('xb') as stream:
                    proc = subprocess.Popen([sys.executable, '-u', '-B', '-m', MODULE,
                        'worker', '--output', str(output), '--shard', str(shard)], cwd=REPO,
                        env=child_environment(gpu_pair(shard)), stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                children.append((shard, proc))
            write_new_json(output/f'wave-{wave}-launch.json', {'children':
                [{'shard': s, 'pid': p.pid} for s, p in children]})
            while any(p.poll() is None for _, p in children):
                if any(p.poll() not in (None, 0) for _, p in children):
                    raise RuntimeError('Readout worker failed; no retry')
                capacity(plan['new_result_reserve_bytes'])
                write_new_json(output/'heartbeats'/f'wave-{wave}-{int(time.time())}.json',
                    {'utc': datetime.now(timezone.utc).isoformat(),
                     'workers': [{'shard': s, 'pid': p.pid, 'exit_code': p.poll()} for s, p in children]})
                time.sleep(30)
            if any(p.returncode != 0 for _, p in children):
                raise RuntimeError('Readout wave failed; see shard logs')
        finalize(output)
        print('COMPLETE seed505 full frozen readout', flush=True)
    except BaseException as error:
        write_new_json(output/'stopped.json', {'stage': 'geometry_or_score',
            'error': repr(error), 'stopped_unix': time.time(), 'automatic_retry': False})
        raise
    finally:
        # Only children created by this run may be stopped on a failure.
        if 'children' in locals():
            for _, p in children:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)


def launch(output):
    output = scope(output); binding(output)
    if any((output/n).exists() for n in ('launch.json', 'workflow.log', 'run-intent.json')):
        raise FileExistsError('Only a fresh launch is allowed')
    with (output/'workflow.log').open('xb') as stream:
        proc = subprocess.Popen([sys.executable, '-u', '-B', '-m', MODULE, 'run',
            '--output', str(output)], cwd=REPO, env=child_environment(),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
    write_new_json(output/'launch.json', {'pid': proc.pid, 'started_unix': time.time(),
        'plan_sha256': file_hash(output/'plan.json'), 'automatic_retry': False})
    print(json.dumps({'status': 'LAUNCHED_NOT_COMPLETE', 'pid': proc.pid,
                      'output': str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch', 'run', 'worker', 'finalize'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    if args.mode == 'worker':
        if args.shard is None:
            parser.error('--shard is required for worker')
        worker(args.output, args.shard)
    else:
        {'prepare': prepare, 'launch': launch, 'run': run,
         'finalize': finalize}[args.mode](args.output)


if __name__ == '__main__':
    main()
