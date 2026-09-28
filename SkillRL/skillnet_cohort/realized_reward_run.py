"""Explicit seed404-only forward/analysis supervisor; immutable, no retries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from .common import REPO, file_hash, read_json, write_new_json
from .reward_variant_analysis import COHORT, SOURCE, UTILITY, check_origin, check_plan, verify_records
from .realized_reward import VERSION, registry

PRIOR = COHORT/'reward-variants-s404-v2'
DEFAULT_OUTPUT = COHORT/'realized-reward-s404-v2'
MODULE = 'skillnet_cohort.realized_reward_run'
NEW_FILES = ['skillnet_cohort/realized_reward.py', 'skillnet_cohort/realized_reward_measure.py',
    'skillnet_cohort/realized_reward_analysis.py', 'skillnet_cohort/realized_reward_run.py',
    'tests/skillnet_cohort/test_realized_reward.py',
    'docs/experiments/phase12-independent-v4/REALIZED-REWARD-SEED404-20260922-v1.md']


def gpu_pair(shard):
    if type(shard) is not int or shard not in range(8):
        raise ValueError('Exactly eight logical shards')
    return f'{2*(shard%4)},{2*(shard%4)+1}'


def storage_gate(plan):
    from .first_calls_storage import disk_gate
    limits = plan['storage']
    return disk_gate(UTILITY, limits['checkpoint_reserve_bytes']+plan['new_result_reserve_bytes'],
        minimum_free_bytes=limits['minimum_free_bytes'], maximum_run_bytes=limits['maximum_run_bytes'])


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_hash(path)}


def validate_scope(plan, output):
    if (plan.get('version') != VERSION or plan.get('approved') is not True
            or plan.get('seed') != 404 or plan.get('output') != str(output)
            or plan.get('source') != str(SOURCE) or plan.get('trajectory_root') != str(UTILITY)
            or plan.get('prior') != str(PRIOR) or plan.get('new_training') is not False
            or plan.get('new_environment_rollouts') != 0 or plan.get('new_api_calls') != 0
            or plan.get('other_seeds') != [] or plan.get('automatic_retry') is not False
            or plan.get('registry') != registry() or plan.get('hard_timeout_seconds') is not None
            or plan.get('minimum_free_gpu_mib') != 28000):
        raise PermissionError('Outside the approved seed404 realized-readout scope')


def binding(output, full=False):
    output = Path(output).resolve()
    if output.parent != COHORT or not output.name.startswith('realized-reward-s404-'):
        raise PermissionError('Scoped new experiment directory required')
    plan = read_json(output/'plan.json'); validate_scope(plan, output)
    verify_records(plan['inputs']+plan['sources'])
    if full:
        verify_records(plan['model_files'])
        verify_records([{'path': r['path'], 'sha256': r['sha256']} for r in plan['old_tensor_files']])
    return plan


def prepare(output, tests):
    import pandas as pd
    from .assets import model_inventory
    from phase2.protocol import anchor_sets
    output = Path(output).resolve()
    if output.exists() or output.parent != COHORT or not output.name.startswith('realized-reward-s404-'):
        raise FileExistsError('Preparation requires a NEW scoped directory')
    suites = list(ET.parse(tests).getroot().iter('testsuite'))
    if not suites or sum(int(s.attrib.get('tests', 0)) for s in suites) < 20 or any(
            int(s.attrib.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Passing CPU regression (at least20 tests, no skips) required')
    prior_plan = check_plan(PRIOR); check_origin()
    if read_json(PRIOR/'complete.json')['status'] != 'complete':
        raise ValueError('Prior v2 analysis incomplete')
    release = read_json(SOURCE.parent/'plan.json')
    model_files = []
    for endpoint in ('u0000', 'u0005'):
        inventory = model_inventory(UTILITY/'models'/endpoint)
        if inventory != release['models']['404'][endpoint]:
            raise ValueError('Retained model differs from the existing numerical evaluation')
        model_files.extend({'path': str(Path(inventory['model_path'])/f['path']), 'sha256': f['sha256']}
                           for f in inventory['files'])
    sources = {r['path']: r for r in prior_plan['preserved_runtime_sources']+prior_plan['analysis_sources']}
    sources.update({str(REPO/p): record(REPO/p) for p in NEW_FILES})
    # Capture archived OLD hashes from the original compressed-row ledger.
    old_rows = {}
    ledger_files = sorted((UTILITY/'old_logprobs/u0001').glob('rank-*.jsonl'))
    for path in ledger_files:
        for line in path.read_text().splitlines():
            value = json.loads(line); row = int(value['row_index'])
            entry = {'row_index': row, 'path': str(UTILITY/'old_logprobs/u0001'/f'row-{row:06d}.pt'),
                'sha256': value['compression']['encoded_sha256']}
            if row in old_rows and old_rows[row] != entry:
                raise ValueError('Conflicting original OLD probability archive')
            old_rows[row] = entry
    if len(old_rows) != 5512:
        raise ValueError('All archived OLD rows must remain present')
    config = read_json(UTILITY/'protocol.json')
    if config['signals']['epsilon'] != 1e-12:
        raise ValueError('Registered epsilon differs from original')
    inputs = [record(tests), record(SOURCE.parent/'plan.json')]
    inputs += prior_plan['inputs']
    inputs += [record(p) for p in sorted(PRIOR.rglob('*')) if p.is_file()]
    inputs += [record(p) for p in ledger_files]
    inputs += [record(UTILITY/f) for f in ('protocol.json', 'resource_limits.json', 'signals/calibration.json',
        'batches/u0001/training_batch.pt', 'batches/u0001/manifest.json')]
    for spec in anchor_sets(config, REPO):
        inputs += [record(REPO/spec[k]) for k in ('anchors_path', 'placebo_path')]
    table = pd.read_parquet(SOURCE/'token_signals.parquet')
    if len(table) != 155638 or table.decision_id.nunique() != 5511:
        raise ValueError('Existing actual-token cohort changed')
    plan = {'version': VERSION, 'approved': True, 'seed': 404, 'output': str(output),
        'user_authority': '2026-09-22: 按此前reward校准实际更新方案扩充并启动全量评估',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source': str(SOURCE),
        'trajectory_root': str(UTILITY), 'prior': str(PRIOR), 'registry': registry(),
        'new_training': False, 'new_model_forward': True, 'new_environment_rollouts': 0,
        'new_api_calls': 0, 'other_seeds': [], 'automatic_retry': False,
        'expected_token_control_rows': len(table), 'expected_decisions': int(table.decision_id.nunique()),
        'forward_dtype': 'bfloat16_SDPA_unchanged', 'vector_arithmetic': 'FP64_16token_chunks',
        'logical_shards': 8, 'parallel_workers': 4, 'gpus_per_worker': 2,
        'minimum_free_gpu_mib': 28000, 'longest_decision_first': True,
        'hard_timeout_seconds': None, 'monitor_interval_seconds': 30,
        'stall_suspected_after_checks': 3, 'stall_action': 'advisory_only_no_retry',
        'new_result_reserve_bytes': 8*2**30, 'storage': read_json(UTILITY/'resource_limits.json'),
        'primary': {'control': 'placebo', 'phase': 'all', 'aggregation': 'token', 'threshold': 0.,
                    'score': 'D_real::token::reward', 'metrics': ['auroc_decline_vs_increase', 'spearman']},
        'event_thresholds': [0., .05], 'bootstrap_draws': 2000, 'sign_null_draws': 512,
        'prior_labels_seen': True, 'scientific_status': 'exploratory_not_confirmatory',
        'inputs': list({r['path']: r for r in inputs}.values()), 'sources': list(sources.values()),
        'model_files': model_files, 'old_tensor_files': [old_rows[k] for k in sorted(old_rows)],
        'original_runtime_source_count': len(prior_plan['preserved_runtime_sources'])}
    validate_scope(plan, output)
    plan['capacity_admission'] = storage_gate(plan)
    verify_records(plan['inputs']+plan['sources'])
    # Row hashes are validated before use by each worker and again before sealing.
    for item in plan['old_tensor_files']:
        if not Path(item['path']).is_file():
            raise FileNotFoundError(item['path'])
    write_new_json(output/'plan.json', plan)
    print(json.dumps({'stage': 'PREPARED_NOT_STARTED', 'output': str(output),
        'plan_sha256': file_hash(output/'plan.json'), 'tokens': len(table),
        'free_GiB': plan['capacity_admission']['free_bytes']/2**30}), flush=True)


def worker_env(shard=None):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='' if shard is None else gpu_pair(shard),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(REPO)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else ''))
    return env


def command(mode, output, shard=None):
    if mode not in ('worker', 'analyze', 'run'):
        raise ValueError('Only fixed workflow commands allowed')
    argv = [sys.executable, '-B', '-m', MODULE, mode, '--output', str(output)]
    if mode == 'worker':
        gpu_pair(shard)
        argv += ['--shard', str(shard)]
    elif shard is not None:
        raise ValueError('No shard on a nonworker job')
    return argv


def wait_jobs(jobs, output, plan, wave):
    checks = 0; stalled = {name: 0 for name, _, _ in jobs}; sizes = dict(stalled)
    storage_error = None
    while True:
        active = []; states = []
        for name, proc, log in jobs:
            code = proc.poll(); size = log.stat().st_size
            stalled[name] = stalled[name]+1 if size == sizes[name] and code is None else 0
            sizes[name] = size
            states.append({'name': name, 'pid': proc.pid, 'exit_code': code, 'log_bytes': size,
                'stall_suspected_advisory': stalled[name] >= plan['stall_suspected_after_checks']})
            if code is None:
                active.append(proc)
        try:
            capacity = storage_gate(plan)
        except OSError as error:
            storage_error = repr(error); capacity = {'storage_alert': storage_error}
        event = {'utc': datetime.now(timezone.utc).isoformat(), 'wave': wave, 'checks': checks,
            'workers': states, 'capacity': capacity, 'automatic_retry': False}
        write_new_json(output/'heartbeats'/f'{wave}-{checks:06d}.json', event)
        print(json.dumps(event), flush=True)
        if not active:
            failed = [(name, proc.returncode) for name, proc, _ in jobs if proc.returncode != 0]
            if failed or storage_error:
                raise RuntimeError(f'No next stage/no retry: workers={failed}, storage={storage_error}')
            return
        # Advisory only: no killing unrelated or still-working shard processes.
        checks += 1; time.sleep(plan['monitor_interval_seconds'])


def run(output):
    output = Path(output).resolve(); plan = binding(output)
    if (output/'run-intent.json').exists():
        raise FileExistsError('Never automatically repeat a started experiment')
    # Refuse starting alongside unrelated GPU jobs, including another user task.
    gpu_users = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if gpu_users:
        raise PermissionError('GPUs are occupied; do not evict existing jobs')
    storage_gate(plan)
    write_new_json(output/'run-intent.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'plan_sha256': file_hash(output/'plan.json'), 'automatic_retry': False})
    (output/'logs').mkdir()
    try:
        for wave in range(2):
            binding(output); storage_gate(plan)
            jobs = []
            for shard in range(4*wave, 4*wave+4):
                log = output/'logs'/f'shard-{shard}.log'
                with log.open('xb') as stream:
                    proc = subprocess.Popen(command('worker', output, shard), cwd=REPO,
                        env=worker_env(shard), stdin=subprocess.DEVNULL, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                jobs.append((f'shard-{shard}', proc, log))
            write_new_json(output/f'wave-{wave}-launch.json', {'jobs': [
                {'name': name, 'pid': proc.pid, 'log': str(log)} for name, proc, log in jobs]})
            wait_jobs(jobs, output, plan, f'wave-{wave}')
        log = output/'logs'/'analysis.log'
        with log.open('xb') as stream:
            proc = subprocess.Popen(command('analyze', output), cwd=REPO, env=worker_env(),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        wait_jobs([('analysis', proc, log)], output, plan, 'analysis')
        if read_json(output/'complete.json')['status'] != 'complete':
            raise ValueError('Missing analysis completion receipt')
        print('PIPELINE_COMPLETE seed404 only; no subsequent seed queued', flush=True)
    except Exception as error:
        write_new_json(output/'failed.json', {'error': repr(error), 'time_unix': time.time(),
            'automatic_retry': False, 'other_seeds_started': False})
        raise


def launch(output):
    output = Path(output).resolve(); binding(output)
    if (output/'launch.json').exists() or (output/'workflow.log').exists() or (output/'run-intent.json').exists():
        raise FileExistsError('One explicit launch only')
    with (output/'workflow.log').open('xb') as stream:
        proc = subprocess.Popen(command('run', output), cwd=REPO, env=worker_env(),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'pid': proc.pid, 'started_unix': time.time(), 'command': command('run', output),
        'plan_sha256': file_hash(output/'plan.json'), 'automatic_retry': False}
    write_new_json(output/'launch.json', receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch', 'run', 'worker', 'analyze'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--tests', type=Path)
    parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output, args.tests)
    elif args.mode == 'worker':
        from .realized_reward_measure import measure
        measure(args.output, args.shard)
    elif args.mode == 'analyze':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise PermissionError('Statistics are explicitly CPU-only')
        from .realized_reward_analysis import analyze
        analyze(args.output)
    else:
        {'launch': launch, 'run': run}[args.mode](args.output)


if __name__ == '__main__':
    main()
