"""One explicitly authorized numerical recovery after releasing paused505.

The four source-frozen numerical modules and all equations are unchanged.
404 report -> 505 numerical commitment -> fresh505 utility workers -> 505 report.
Never resume dead processes, resample completed trajectories, or restart RL.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from .common import REPO, exclusive_writer, file_hash, read_json, write_new_bytes, write_new_json
from . import numerical_readout as nr
from . import numerical_readout_run as previous_run
from .release_paused_505 import OUTPUT, PREVIOUS, AUTHORITY, released
from .first_calls_defer import gpu_users
from .seed_queue import verify_sources

SCHEMA = 'skillnet.numerical_readout_released505.v1'
MODULE = 'skillnet_cohort.numerical_released_run'
EXTRA_SOURCES = ('skillnet_cohort/release_paused_505.py',
    'skillnet_cohort/numerical_released_run.py', 'skillnet_cohort/released_505_evaluation.py')


def binding(path, seed=None):
    path = Path(path).resolve(); plan = read_json(path)
    if (path != OUTPUT/'plan.json' or plan.get('schema_version') != SCHEMA
            or plan.get('root') != str(OUTPUT) or plan.get('approved') is not True
            or plan.get('seeds') != [404, 505] or plan.get('automatic_retry') is not False
            or plan.get('seed505_restart_mode') != 'fresh_processes_reuse_disk_results'
            or plan.get('minimum_free_gpu_mib') != 28000
            or plan.get('rewrite_original_reports') is not False
            or plan.get('external_api_calls') != 0 or plan.get('training_reexecuted') is not False
            or plan.get('seed606_started') is not False):
        raise PermissionError('Only the explicit released505 numerical recovery is allowed')
    if seed is not None and (seed not in (404, 505)
            or plan['source_roots'][str(seed)] != str(nr.SOURCES[seed])):
        raise PermissionError('Unregistered seed/source')
    verify_sources(plan)
    for item in [*plan['bindings'], *plan['dependency_sources']]:
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Frozen numerical input changed: '+item['path'])
    return plan


def released_guard(plan):
    receipt = read_json(plan['release_505']['path'])
    if (file_hash(plan['release_505']['path']) != plan['release_505']['sha256']
            or receipt['status'] != 'RELEASED_BY_USER' or receipt['disk_results_preserved'] is not True):
        raise PermissionError('Explicit process-release evidence is required')
    released(receipt)
    # Before a separately recorded505 restart, every original index stays fixed.
    if not (Path(plan['root'])/'seed505-restart-intent.json').exists():
        progress = read_json(plan['pause_verification']['path'])
        for row in progress['endpoints']['u0000']['shards']:
            if file_hash(row['path']) != row['sha256']:
                raise ValueError('505 advanced before404 completion and505 score commitment')


def prepare(test_report):
    from .assets import model_inventory
    from .explicit_router_resume import audit_cache, read_only, row_signature
    from .first_calls_port_recovery import binding as port_binding
    from .first_calls_recovery import audit_endpoint, verify_retained
    if (OUTPUT/'plan.json').exists() or (OUTPUT/'router-resume-authorization.json').exists():
        raise FileExistsError('Preparation is a single new explicit attempt')
    old = nr.binding(PREVIOUS)
    receipt = read_json(OUTPUT/'release-505.json'); released(receipt)
    if gpu_users():
        raise PermissionError('All GPUs must be released before preparation')
    suites = list(ET.parse(test_report).getroot().iter('testsuite'))
    if (not suites or sum(int(s.attrib.get('tests', 0)) for s in suites) < 30
            or any(int(s.attrib.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped'))):
        raise ValueError('Passing release/numerical/restart CPU regression required')
    root = nr.SOURCES[505]
    for rel in ('evaluations/u0005', 'window_signals/u0000-u0005/prediction.json',
                'analysis-complete.json', 'reports', 'complete.json'):
        if (root/rel).exists():
            raise ValueError('The approved505 partial-U0 boundary changed')
    port, original = port_binding(nr.F/'recovery-v3/plan.json')
    verify_retained(port['retained'])
    endpoint = audit_endpoint(root, 0)
    if (endpoint['completed'], endpoint['expected']) != (810, 3636):
        raise ValueError('Only the recorded810 retained505 trajectories may resume')
    for seed, source in nr.SOURCES.items():
        for name in ('u0000', 'u0005'):
            if model_inventory(source/'models'/name) != old['models'][str(seed)][name]:
                raise ValueError('Retained policy weights changed')
    snapshots = receipt['preserved_cache_files']
    if len(snapshots) != 1 or Path(snapshots[0]['path']) != root/'router.sqlite3':
        raise ValueError('A sidecar-free preserved SQLite snapshot is required')
    snapshot = OUTPUT/'router-before-resume.sqlite3'
    write_new_bytes(snapshot, Path(snapshots[0]['snapshot']).read_bytes())
    if file_hash(snapshot) != snapshots[0]['sha256'] or file_hash(root/'router.sqlite3') != file_hash(snapshot):
        raise ValueError('Router changed after process release')
    allowed = []
    with read_only(snapshot) as cache:
        protocol_hash, _ = cache.execute('SELECT hash,data FROM protocol WHERE id=1').fetchone()
        rows = cache.execute('SELECT a.id,a.key,a.started,a.status,a.data FROM attempts a LEFT JOIN decisions d ON a.key=d.key WHERE d.key IS NULL ORDER BY a.id').fetchall()
        if [r[0] for r in rows] != [5826, 5828, 5829, 5830, 5831] or any(r[3] != 'started' for r in rows):
            raise PermissionError('Only the five queries interrupted by this paused505 release may reissue')
        for row in rows:
            allowed.append({'id': row[0], 'key': row[1], 'started': row[2], 'status': row[3],
                            'row_sha256': row_signature(row)})
    config = read_json(root/'protocol.json')
    authorization = {'schema_version': 'skillnet.explicit_local_query_resume.v1', 'approved': True,
        'user_authority': AUTHORITY+'；仅对本次释放的5条未完成本地查询各续算一次',
        'recovery_root': str(OUTPUT), 'run_root': str(root), 'cache_path': str(root/'router.sqlite3'),
        'protocol_hash': protocol_hash, 'max_local_calls': config['runtime']['max_local_calls'],
        'evaluation_protocol_sha256': file_hash(root/'protocol.json'),
        'snapshot': {'path': str(snapshot), 'sha256': file_hash(snapshot)}, 'allowed': allowed,
        'external_api_calls': 0, 'max_reissues_per_key': 1, 'automatic_retry': False,
        'original_attempt_rows_immutable': True, 'successful_decisions_immutable': True}
    auth_path = OUTPUT/'router-resume-authorization.json'
    write_new_json(auth_path, authorization)
    cache_audit = audit_cache(root/'router.sqlite3', authorization)
    retained_files = {r['path']: r for r in port['retained']['files']}
    retained_files.update({r['path']: r for r in endpoint['files']})
    extra = [PREVIOUS, PREVIOUS.parent/'stopped.json', OUTPUT/'release-505.json',
        OUTPUT/'release-intent.json', Path(test_report).resolve(), auth_path, snapshot]
    plan = {**old, 'schema_version': SCHEMA, 'root': str(OUTPUT), 'created_unix': time.time(),
        'user_authority': AUTHORITY, 'seed505_restart_mode': 'fresh_processes_reuse_disk_results',
        'source_sha256': {**old['source_sha256'], **{p: file_hash(REPO/p) for p in EXTRA_SOURCES}},
        'bindings': old['bindings']+[{'path': str(p), 'sha256': file_hash(p)} for p in extra],
        'release_505': {'path': str(OUTPUT/'release-505.json'), 'sha256': file_hash(OUTPUT/'release-505.json')},
        'router_resume': {'path': str(auth_path), 'sha256': file_hash(auth_path)},
        'router_preflight': cache_audit, 'router_at_preparation_sha256': file_hash(snapshot),
        'minimum_free_gpu_mib': 28000, 'legacy_comparator_shape_unchanged': True,
        'assessment_plan': port['assessment_plan'], 'dependency_sources': port['dependency_sources'],
        'retained505': {'files': list(retained_files.values()), 'shards': endpoint['shards']},
        'endpoint505': {k: v for k, v in endpoint.items() if k != 'files'},
        'engineering_tests': {'path': str(Path(test_report).resolve()), 'sha256': file_hash(test_report)},
        'restore_in_memory_processes': False, 'completed_utility_trajectories_reexecuted': False,
        'missing505_utility_continuations_authorized': True, 'runtime_failure_stops_group_no_retry': True}
    released_guard(plan); plan['capacity_admission'] = nr.disk(plan)
    write_new_json(OUTPUT/'plan.json', plan)
    print({'state': 'PREPARED_NOT_STARTED', 'plan': str(OUTPUT/'plan.json'),
        'sources': len(plan['source_sha256']), 'retained505': endpoint['completed'],
        'explicit_local_reissues': len(allowed)}, flush=True)
    return plan


@contextmanager
def numerical_adapter():
    from . import numerical_readout_report as reporting
    saved = nr.binding, nr.paused_tree, reporting.binding
    nr.binding, nr.paused_tree, reporting.binding = binding, released_guard, binding
    try:
        yield
    finally:
        nr.binding, nr.paused_tree, reporting.binding = saved


def command_jobs(path, seed, mode, shards=()):
    if mode == 'utility':
        if seed != 505 or not shards or any(r not in range(8) for r in shards) or len(set(shards)) != len(shards):
            raise PermissionError('Only registered505 utility shards are allowed')
        raise ValueError('Use utility_jobs with an explicit endpoint')
    jobs = previous_run.command_jobs(path, seed, mode, shards)
    result = []
    for args, label, devices in jobs:
        args = [MODULE, *args[1:]]
        if mode == 'report':
            args.append('--report')
        result.append((args, label, devices))
    return result


def utility_jobs(path, update, shards):
    if update not in (0, 5) or not shards or any(r not in range(8) for r in shards) or len(set(shards)) != len(shards):
        raise PermissionError('Distinct registered505 utility shards and endpoints required')
    return [(['skillnet_cohort.released_505_evaluation', '--plan', str(path), '--evaluate',
        '--update', str(update), '--shard', str(r)], f'seed505-utility-u{update:04d}-shard{r}', str(r)) for r in shards]


def verify_restart_requirements(plan):
    root = Path(plan['root'])
    previous_run.verify_report_completion(root/'seed-404')
    commit = read_json(root/'seed-505/committed.json')
    if (commit['seed'] != 505 or commit['all_legacy_scalar_signals_exact'] is not True
            or file_hash(root/'seed-505/skill_context_features.parquet') != commit['features_sha256']
            or file_hash(root/'seed-505/token_signals.parquet') != commit['token_signals_sha256']):
        raise PermissionError('Corrected505 scores must be committed before utility restart')
    return commit


def commands(path, jobs):
    plan = binding(path); nr.disk(plan); released_guard(plan)
    allowed = []
    for seed in (404, 505):
        for r in range(8):
            allowed.extend(command_jobs(path, seed, 'measure', [r]))
        for mode in ('aggregate', 'report'):
            allowed.extend(command_jobs(path, seed, mode))
    for update in (0, 5):
        allowed.extend(utility_jobs(path, update, list(range(8))))
    if not jobs or any(job not in allowed for job in jobs) or len({j[1] for j in jobs}) != len(jobs):
        raise PermissionError('Command not in the explicit numerical/utility whitelist')
    devices = [device for _, _, gpu in jobs for device in gpu.split(',') if device]
    if len(set(devices)) != len(devices):
        raise PermissionError('Concurrent jobs must not overlap GPUs')
    if any('-measure-' in label for _, label, _ in jobs) and (OUTPUT/'seed505-restart-intent.json').exists():
        raise PermissionError('Do not recompute readout after505 utility has restarted')
    if any('utility-' in label for _, label, _ in jobs):
        verify_restart_requirements(plan)
        if not (OUTPUT/'seed505-restart-intent.json').exists():
            raise PermissionError('505 utility restart has not been explicitly admitted')
    children = []; next_check = 0
    try:
        for args, label, devices in jobs:
            log = OUTPUT/'logs'/(label+'.log'); log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open('x'); env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=devices, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                HF_HUB_OFFLINE='1', PYTHONPATH=str(REPO)+os.pathsep+env.get('PYTHONPATH', ''))
            env.pop('SKILLNET_PHASE12_RECOVERY_AUTHORIZATION', None)
            env.pop('SKILLSCOPE_EVAL_RENDEZVOUS_RECORD', None)
            try:
                child = subprocess.Popen([sys.executable, '-u', '-B', '-m', *args], cwd=REPO,
                    env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
            except BaseException:
                stream.close(); raise
            children.append((child, stream, label, log))
        while any(p.poll() is None for p, _, _, _ in children):
            if any(p.poll() not in (None, 0) for p, _, _, _ in children):
                raise RuntimeError('Recovery child failed; preserve evidence, no automatic retry')
            if time.time() >= next_check:
                nr.disk(plan); released_guard(plan)
                previous_run.status(plan, ','.join(c[2] for c in children), hard_limit_seconds=None,
                    children=[{'pid': p.pid, 'label': label, 'exit_code': p.poll(),
                               'log_bytes': log.stat().st_size} for p, _, label, log in children])
                next_check = time.time()+30
            time.sleep(1)
        if any(p.returncode != 0 for p, _, _, _ in children):
            raise RuntimeError('Recovery stage failed; no automatic retry')
    finally:
        for p, _, _, _ in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)  # Own newly started workers only.
        for p, stream, _, _ in children:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            stream.close()


def restart505(path):
    from .released_505_evaluation import run as evaluate_missing
    plan = binding(path); verify_restart_requirements(plan); released_guard(plan)
    if gpu_users():
        raise PermissionError('Numerical workers must release GPUs before505 evaluation')
    intent = OUTPUT/'seed505-restart-intent.json'
    if intent.exists():
        raise FileExistsError('No automatic505 restart repetition')
    write_new_json(intent, {'created_unix': time.time(), 'authority': AUTHORITY,
        'seed404_complete_sha256': file_hash(OUTPUT/'seed-404/complete.json'),
        'seed505_commit_sha256': file_hash(OUTPUT/'seed-505/committed.json'),
        'router_authorization_sha256': plan['router_resume']['sha256'],
        'restart_mode': 'fresh_processes_reuse_disk_results', 'automatic_retry': False})
    evaluate_missing(path)


def pipeline(path):
    plan = binding(path); released_guard(plan)
    for seed in (404, 505):
        if seed == 505:
            previous_run.verify_report_completion(OUTPUT/'seed-404')
        for shards in (list(range(4)), list(range(4, 8))):
            commands(path, command_jobs(path, seed, 'measure', shards))
        commands(path, command_jobs(path, seed, 'aggregate'))
        if seed == 505:
            restart505(path)
        commands(path, command_jobs(path, seed, 'report'))
        previous_run.verify_report_completion(OUTPUT/f'seed-{seed}')


def run(path):
    plan = binding(path); released_guard(plan); nr.disk(plan)
    if (OUTPUT/'launch.json').exists() or gpu_users():
        raise PermissionError('Only one new launch on idle GPUs is allowed')
    write_new_json(OUTPUT/'launch.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'plan_sha256': file_hash(path), 'automatic_retry': False, 'old505_released': True})
    try:
        with exclusive_writer(OUTPUT):
            pipeline(path)
        write_new_json(OUTPUT/'complete.json', {'status': 'complete', 'seeds': [404, 505],
            'original_reports_preserved': True, 'new_RL': False, 'seed606_started': False})
        previous_run.status(plan, 'COMPLETE_404_505_NUMERICAL_CORRECTION')
    except BaseException as error:
        write_new_json(OUTPUT/'stopped.json', {'error': f'{type(error).__name__}: {error}',
            'stopped_unix': time.time(), 'automatic_retry': False,
            'seed505_restart_intent_exists': (OUTPUT/'seed505-restart-intent.json').exists()})
        previous_run.status(plan, 'STOPPED', error=repr(error)); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path); p.add_argument('--prepare', action='store_true')
    p.add_argument('--test-report', type=Path); p.add_argument('--execute', action='store_true')
    p.add_argument('--detach', action='store_true'); p.add_argument('--seed', type=int, choices=(404, 505))
    p.add_argument('--shard', type=int)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--measure', action='store_true'); modes.add_argument('--aggregate', action='store_true')
    modes.add_argument('--report', action='store_true'); args = p.parse_args()
    if args.prepare:
        if args.execute or args.detach or args.plan or not args.test_report:
            p.error('Preparation is separate from execution')
        prepare(args.test_report); return
    if not args.plan:
        p.error('Explicit plan required')
    plan = binding(args.plan)
    if args.measure or args.aggregate or args.report:
        if args.execute or args.detach or args.seed is None:
            p.error('A numerical worker requires its seed and one mode only')
        with numerical_adapter():
            if args.measure:
                if args.shard is None or (OUTPUT/'seed505-restart-intent.json').exists():
                    p.error('Readout requires a shard before505 utility restart')
                nr.measure(args.plan, args.seed, args.shard)
            elif args.aggregate:
                nr.aggregate(args.plan, args.seed)
            else:
                from .numerical_readout_report import report
                report(args.plan, args.seed)
        return
    if not args.execute:
        p.error('Explicit --execute required')
    if args.detach:
        released_guard(plan)
        if gpu_users():
            raise PermissionError('All GPUs must be free before launch')
        with (OUTPUT/'workflow.log').open('x') as stream:
            env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                PYTHONPATH=str(REPO)+os.pathsep+env.get('PYTHONPATH', ''))
            child = subprocess.Popen([sys.executable, '-u', '-B', '-m', MODULE,
                '--plan', str(args.plan.resolve()), '--execute'], cwd=REPO, env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_new_json(OUTPUT/'supervisor.json', {'pid': child.pid, 'plan_sha256': file_hash(args.plan)})
        print({'state': 'STARTED', 'pid': child.pid}, flush=True); return
    def stop(signum, frame):
        raise InterruptedError('Explicitly stopped released505 numerical recovery')
    signal.signal(signal.SIGTERM, stop)
    run(args.plan)


if __name__ == '__main__':
    main()
