"""Explicit one-shot recovery of all-first-call assessment, without legacy edits.

Wait for the currently running legacy queue to finish naturally; resume seed404
at the missing U0 utility jobs, then extend successfully completed legacy seeds.
An unstarted seed blocked by storage remains pending, never bypasses admission.
No training, export, performance resampling, process pausing or automatic retry.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from .common import REPO, exclusive_writer, file_hash, read_json, write_new_json
from .first_calls_run import Assessment

EXTRA_SOURCES = ('skillnet_cohort/first_calls_storage.py', 'skillnet_cohort/first_calls_recovery.py')


def audit_endpoint(root, update):
    """Validate every retained episode and its exact partition before skipping it.

    Fail closed on orphan/partial files rather than letting the evaluator replace
    them. The authorized failure has neither; a different failure needs review.
    """
    from phase1.archive import stable_hash
    from phase2.protocol import evaluation_jobs, evaluation_identity
    from .first_calls_reuse import ANCHOR_SEMANTIC_FIELDS
    root = Path(root).resolve(); config = read_json(root/'protocol.json')
    shards = config['evaluation']['shards']; jobs = evaluation_jobs(config, REPO)
    expected = {}
    for position, job in enumerate(jobs):
        identity = evaluation_identity(config, update, job)
        tid = stable_hash(identity)[:24]
        if tid in expected:
            raise ValueError('Duplicate registered continuation identity')
        expected[tid] = (position % shards, identity, job[1])
    directory = root/'evaluations'/f'u{update:04d}'
    seen, files, counts, finished = set(), [], [], []
    for rank in range(shards):
        index = directory/f'shard-{rank}.jsonl'
        raw = index.read_bytes() if index.exists() else b''
        if raw and not raw.endswith(b'\n'):
            raise ValueError('Partial index tail; refuse automatic overwrite')
        rows = [json.loads(line) for line in raw.splitlines()]
        for row in rows:
            tid = row['trajectory_id']
            if tid in seen or tid not in expected or expected[tid][0] != rank:
                raise ValueError('Duplicate, unregistered or wrong-shard continuation')
            _, identity, anchor = expected[tid]
            target = directory/'trajectories'/identity['skill_id']/(tid+'.json')
            if row['trajectory_path'] != str(target) or target.is_symlink():
                raise ValueError('Continuation path changed')
            result = read_json(target)
            if (any(row.get(k) != v for k, v in identity.items())
                    or any(result.get(k) != v for k, v in row.items())
                    or not result['steps'] or result['prefix_replay_verified'] is not True
                    or result['actual_continuation_seed'] != identity['continuation_seed']
                    or any(result['original_anchor'].get(k) != anchor.get(k) for k in ANCHOR_SEMANTIC_FIELDS)):
                raise ValueError('Retained continuation identity/replay/content failed validation')
            seen.add(tid); files.append({'path': str(target), 'sha256': file_hash(target)})
        total = sum(v[0] == rank for v in expected.values())
        marker = directory/f'shard-{rank}-complete.json'
        if marker.exists():
            value = read_json(marker)
            if (len(rows) != total or value['jobs'] != total or value['shard'] != rank
                    or value['shards'] != shards or value['max_jobs'] is not None
                    or value['protocol_sha256'] != file_hash(root/'protocol.json')):
                raise ValueError('Invalid completed-shard receipt')
            finished.append(rank)
        counts.append({'shard': rank, 'completed': len(rows), 'expected': total,
            'index': str(index), 'index_bytes': len(raw), 'index_sha256': file_hash(index) if raw else None})
    if {str(p) for p in directory.glob('trajectories/*/*.json')} != {r['path'] for r in files}:
        raise ValueError('Orphan trajectory files; never overwrite or silently reexecute')
    if any(p.name not in {f'shard-{i}.jsonl' for i in range(shards)} for p in directory.glob('shard-*.jsonl')):
        raise ValueError('Unexpected evaluation shard')
    return {'update': update, 'expected': len(expected), 'completed': len(seen),
        'missing': len(expected)-len(seen), 'complete_shards': finished, 'shards': counts, 'files': files}


def verify_retained(record):
    for row in record['files']:
        if file_hash(row['path']) != row['sha256']:
            raise ValueError('Retained evidence changed: '+row['path'])
    # Indices are append-only; their pre-recovery prefix must stay byte-identical.
    import hashlib
    for row in record['shards']:
        if row['index_bytes']:
            with Path(row['index']).open('rb') as stream:
                actual = hashlib.sha256(stream.read(row['index_bytes'])).hexdigest()
            if actual != row['index_sha256']:
                raise ValueError('Retained index prefix changed')


def validate_readout(root):
    from phase2.window_evidence import validate_shards
    out = root/'window_signals/u0000-u0005'; value = read_json(out/'committed.json')
    for field, name in (('features_sha256', 'skill_context_features.parquet'),
                        ('token_signals_sha256', 'token_signals.parquet')):
        if value[field] != file_hash(out/name):
            raise ValueError('Committed readout changed')
    return validate_shards(root, 0, 5, 8)


def prepare(path, output, test_report):
    from .seed_queue import verify_sources
    from .first_calls_defer import process_identity
    path = Path(path).resolve(); output = Path(output).resolve(); plan = read_json(path)
    if output.parent != path.parent or output.name != 'recovery-v1' or output.exists():
        raise FileExistsError('This failure requires a fresh all-first-calls-v1/recovery-v1')
    verify_sources(plan)
    if file_hash(plan['legacy_queue_plan']) != plan['legacy_queue_plan_sha256']:
        raise ValueError('Original legacy queue plan changed')
    suites = list(ET.parse(test_report).getroot().iter('testsuite'))
    if (not suites or any(int(s.attrib.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped'))
            or sum(int(s.attrib.get('tests', 0)) for s in suites) < 20):
        raise ValueError('Passing recovery and coverage CPU tests required')
    root = Path(plan['jobs'][0]['run_root']); stopped = read_json(root/'stopped.json')
    if (not stopped['stage'].startswith('utility-u0000-shard')
            or 'FileNotFoundError' not in stopped['error'] or 'router.sqlite3-journal' not in stopped['error']):
        raise ValueError('Different failure needs a separately reviewed recovery')
    for receipt in ('assessment-launch.json', 'deferred-supervisor.json'):
        current = process_identity(read_json(path.parent/receipt)['pid'])
        if current and current['state'] != 'Z':
            raise RuntimeError('Previous assessment process still exists; refuse concurrent recovery')
    if ((root/'window_signals/u0000-u0005/prediction.json').exists()
            or (root/'evaluations/u0005').exists() or (root/'complete.json').exists()):
        raise ValueError('This explicit recovery starts only at partial U0 utility')
    record = audit_endpoint(root, 0)
    readout = validate_readout(root)
    # Freeze old failures, finished readout, support, and protocol. No model copy.
    files = []
    paths = [path, path.parent/'permit.json', root/'protocol.json', root/'manifest.json',
        root/'launch.json', root/'stopped.json', root/'reuse/u0000.json',
        path.parent/'assessment.log', path.parent/'deferred.log', path.parent/'deferred-stopped.json',
        path.parent/'coordinators-resumed.json']
    for subdir in ('window_signals/u0000-u0005', 'support', 'signals', 'logs'):
        paths.extend(p for p in (root/subdir).rglob('*') if p.is_file())
    for p in paths:
        files.append({'path': str(p), 'sha256': file_hash(p)})
    retained = {'files': files + record['files'], 'shards': record['shards']}
    value = {'schema_version': 'skillnet.all_first_calls_explicit_recovery.v1', 'approved': True,
        'user_authority': '2026-09-21 修复磁盘扫描竞态并复用已有结果恢复补评；不重跑RL',
        'root': str(output), 'original_plan': str(path), 'original_plan_sha256': file_hash(path),
        'source_sha256': {**plan['source_sha256'], **{p: file_hash(REPO/p) for p in EXTRA_SOURCES}},
        'seeds': [404, 505, 606], 'automatic_retry': False, 'pause_legacy_processes': False,
        'hard_limit_seconds': None, 'storage_limits_unchanged': True,
        'engineering_tests': {'path': str(Path(test_report).resolve()), 'sha256': file_hash(test_report)},
        'retained': retained, 'endpoint_summary': {k: v for k, v in record.items() if k != 'files'},
        'readout_audit': readout, 'scientific_protocol_unchanged': True}
    finished = queue_finished(plan)
    if finished:
        value['legacy_exit'] = {'path': str(finished), 'sha256': file_hash(finished),
            'record': read_json(finished)}
    assessment = RecoveryAssessment(path, output/'seed-404', value)
    value['capacity_admission'] = assessment.disk(plan['projected_recording_upper_bytes'])
    write_new_json(output/'plan.json', value)
    print(json.dumps({'prepared': str(output/'plan.json'), 'retained_U0': record['completed'],
        'missing_U0': record['missing'], 'readout_reexecuted': False}), flush=True)
    return value


class RecoveryAssessment(Assessment):
    def __init__(self, path, audit, recovery):
        super().__init__(path)
        self.audit = Path(audit); self.recovery = recovery

    def disk(self, additional=0):
        from .first_calls_storage import disk_gate
        limits = self.permit['storage']
        return disk_gate(self.root, limits['checkpoint_reserve_bytes']+additional,
            minimum_free_bytes=limits['minimum_free_bytes'], maximum_run_bytes=limits['maximum_run_bytes'])

    def commands(self, jobs):
        from .runtime_watch import StageWatch
        from .seed_queue import verify_sources
        if any(args[0] not in ('phase2.evaluate', 'phase2.aggregate', 'skillnet_cohort.first_calls_measure')
               for args, _, _ in jobs):
            raise PermissionError('Recovery only permits missing readout and utility evaluation commands')
        verify_sources(self.plan); verify_sources(self.recovery); self.disk()
        self.active_stage = ','.join(label for _, label, _ in jobs)
        watch = StageWatch(self.root, self.audit, self.active_stage, None)
        children = []; next_disk = 0
        try:
            for arguments, label, gpu in jobs:
                path = self.audit/'logs'/(label+'.log'); path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.open('x'); env = os.environ.copy()
                env.update(TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                    PYTHONPATH=str(REPO)+os.pathsep+env.get('PYTHONPATH', ''))
                env.pop('SKILLNET_PHASE12_RECOVERY_AUTHORIZATION', None)
                try:
                    child = subprocess.Popen([sys.executable, '-u', '-B', '-m', *map(str, arguments)],
                        cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    stream.close(); raise
                children.append((child, stream, label))
            watch.tick(children, event='started', force=True)
            while any(p.poll() is None for p, _, _ in children):
                if any(p.poll() not in (None, 0) for p, _, _ in children):
                    raise RuntimeError('Assessment child failed; preserve evidence, no automatic retry')
                if time.time() >= next_disk:
                    self.disk(); next_disk = time.time()+60
                watch.tick(children); time.sleep(1)
            if any(p.returncode != 0 for p, _, _ in children):
                raise RuntimeError('Assessment stage failed; see retained logs')
        finally:
            for p, _, _ in children:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)  # Own newly created sessions only.
            for p, stream, _ in children:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait()
                stream.close()
            watch.tick(children, event='finished' if all(p.returncode == 0 for p, _, _ in children) else 'stopped', force=True)

    def evaluate(self, update):
        before = audit_endpoint(self.root, update)
        ranks = [r for r in range(8) if r not in before['complete_shards']]
        if ranks:
            self.commands([(['phase2.evaluate', '--root', self.root, '--update', update,
                '--shards', 8, '--shard', rank], f'utility-u{update:04d}-shard{rank}', rank) for rank in ranks])
        after = audit_endpoint(self.root, update)
        verify_retained(before)
        if after['missing'] or after['complete_shards'] != list(range(8)):
            raise ValueError('Endpoint incomplete after evaluation')
        write_new_json(self.audit/f'endpoint-u{update:04d}.json',
            {k: v for k, v in after.items() if k != 'files'})

    def resume404(self):
        from .assets import model_inventory
        from .first_calls_run import verify_legacy_seal, lock_prediction
        from .first_calls_reuse import import_endpoint
        from .first_calls_report import report, publish
        from .window_storage import seal_window
        from .seed_queue import verify_sources
        verify_sources(self.plan); verify_sources(self.recovery)
        verify_retained(self.recovery['retained'])
        for update in (0, 5):
            if model_inventory(self.root/'models'/f'u{update:04d}') != self.plan['model_inventory'][f'u{update:04d}']:
                raise ValueError('Retained endpoint changed')
        verify_legacy_seal(self.plan); validate_readout(self.root)
        self.disk(self.plan['projected_recording_upper_bytes'])
        write_new_json(self.audit/'launch.json', {'started_unix': time.time(), 'explicit_recovery': True,
            'readout_reexecuted': False, 'training_reexecuted': False, 'automatic_retry': False})
        self.evaluate(0)
        lock_prediction(self.root)
        import_endpoint(self.root, 5)
        self.evaluate(5)
        self.active_stage = 'analysis_and_publication'
        report(self.root); seal_window(self.root, 0, 5)
        verify_retained(self.recovery['retained'])
        publish(self.root)
        write_new_json(self.root/'complete.json', {'status': 'complete', 'training_reexecuted': False,
            'reports_published': True, 'old_evidence_preserved': True, 'explicit_recovery': str(self.audit)})

    def fresh_completed_seed(self, finished):
        """Same amendment, scoped to one completed seed even if 606 was blocked.

        Do not forge the legacy queue's all-three-seeds completion receipt merely
        to pass the earlier runner's stricter scheduling (not scientific) gate.
        """
        from .assets import model_inventory
        from .first_calls_run import verify_legacy_seal, lock_prediction
        from .first_calls_reuse import import_endpoint
        from .first_calls_report import report, publish
        from .seed_queue import verify_sources
        from .window_storage import seal_window
        from .first_calls_defer import gpu_users
        from phase2.protocol import validate_extended
        seed = self.plan['jobs'][0]['seed']
        if seed not in read_json(finished)['completed_seeds'] or gpu_users():
            raise PermissionError('A successful retained seed and idle GPUs are required')
        if (self.root/'launch.json').exists():
            raise FileExistsError('No implicit followup retry')
        verify_sources(self.plan); verify_sources(self.recovery)
        registration = self.plan['policy_registration']
        if file_hash(registration) != self.plan['policy_registration_sha256']:
            raise ValueError('Earlier registered coverage rule changed')
        for update in (0, 5):
            if model_inventory(self.root/'models'/f'u{update:04d}') != self.plan['model_inventory'][f'u{update:04d}']:
                raise ValueError('Retained endpoint weights changed')
        verify_legacy_seal(self.plan); validate_extended(read_json(self.root/'protocol.json'), REPO)
        self.disk(self.plan['projected_recording_upper_bytes'])
        write_new_json(self.root/'launch.json', {'plan_sha256': file_hash(self.path),
            'started_unix': time.time(), 'training_executed': False, 'performance_resampled': False,
            'automatic_retry': False, 'legacy_exit': str(finished), 'recovery_root': str(self.audit)})
        self.commands([(['skillnet_cohort.first_calls_measure', '--root', self.root, '--shard', rank],
            f'readout-shard{rank}', rank) for rank in range(8)])
        self.commands([(['phase2.aggregate', '--root', self.root, '--update', 5,
            '--start-update', 0, '--shards', 8], 'readout-aggregate', None)])
        import_endpoint(self.root, 0); self.evaluate(0); lock_prediction(self.root)
        import_endpoint(self.root, 5); self.evaluate(5)
        self.active_stage = 'analysis_and_publication'
        report(self.root); seal_window(self.root, 0, 5); publish(self.root)
        write_new_json(self.root/'complete.json', {'status': 'complete', 'training_reexecuted': False,
            'reports_published': True, 'old_evidence_preserved': True})


def queue_finished(plan):
    """Wait-only handoff: never signal a legacy coordinator, RL or evaluator."""
    from .first_calls_defer import same_process, process_identity
    queue = plan['handoff_processes']['queue']
    if same_process(queue) and process_identity(queue['pid'])['state'] != 'Z':
        return None
    previous = read_json(plan['legacy_queue_plan'])
    path = Path(previous['recovery']['audit_root'])/'queue_finished.json'
    if not path.is_file():
        raise RuntimeError('Legacy queue exited without completion; do not restart failed RL')
    record = read_json(path)
    if record['status'] == 'complete' and record['completed_seeds'] == [404, 505, 606]:
        return path
    if (record['status'] == 'budget_stop' and record['completed_seeds'] == [404, 505]
            and record['not_completed_seeds'] == [606]):
        stop = read_json(path.parent/'not-started-606.json')
        if (stop['reason'] == 'remaining_shared_disk_budget' and stop['not_started_seeds'] == [606]
                and stop['no_files_deleted'] is True
                and not (Path(plan['cohort_root'])/'seed-606').exists()):
            return path
    raise RuntimeError('Legacy queue incomplete for an unreviewed reason; no automatic retry')


def run(path):
    from .first_calls_defer import gpu_users, status
    from .first_calls_protocol import prepare as prepare_followup
    from .first_calls_report import publish_cohort
    from .seed_queue import verify_sources
    path = Path(path).resolve(); recovery = read_json(path); output = Path(recovery['root'])
    original_path = Path(recovery['original_plan']); plan = read_json(original_path)
    if (original_path.parent != output.parent or output != path.parent
            or file_hash(original_path) != recovery['original_plan_sha256']
            or recovery.get('approved') is not True or recovery.get('seeds') != [404, 505, 606]):
        raise PermissionError('Recovery binding changed')
    if file_hash(plan['legacy_queue_plan']) != plan['legacy_queue_plan_sha256']:
        raise ValueError('Original legacy queue plan changed')
    verify_sources(recovery); verify_retained(recovery['retained'])
    if (output/'launch.json').exists():
        raise FileExistsError('Explicit attempt already started; no automatic retry')
    write_new_json(output/'launch.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'plan_sha256': file_hash(path), 'legacy_processes_signalled': False, 'automatic_retry': False})
    active = None
    try:
        while (finished := queue_finished(plan)) is None:
            status(output, 'WAITING_LEGACY_QUEUE', seeds=[404, 505, 606], paused_pids=[],
                recovery_registered=True, gpu_experiment_started=False)
            time.sleep(30)
        if recovery.get('legacy_exit') and file_hash(finished) != recovery['legacy_exit']['sha256']:
            raise ValueError('Registered legacy exit changed')
        for seed in recovery['seeds']:
            if seed not in read_json(finished)['completed_seeds']:
                write_new_json(output/'pending.json', {'pending_seeds': [seed],
                    'reason': 'legacy_seed_not_started_due_to_storage_guard', 'legacy_exit': str(finished),
                    'completed_assessments': [404, 505], 'no_storage_waiver': True, 'no_evidence_deleted': True})
                status(output, 'BLOCKED_LEGACY_STORAGE', pending_seed=seed, completed_assessments=[404, 505],
                    paused_pids=[], no_automatic_RL_launch=True)
                return
            verify_sources(recovery)
            while gpu_users():
                status(output, 'WAITING_IDLE_GPUS', next_seed=seed, paused_pids=[])
                time.sleep(30)
            next_plan = original_path
            if seed != 404:
                destination = original_path.parent/f'followup-s{seed}'
                prepare_followup(plan['legacy_queue_plan'], destination, seed=seed,
                    test_report=recovery['engineering_tests']['path'], policy_registration=original_path)
                next_plan = destination/'plan.json'
                write_new_json(destination/'handoff.json', {'status': 'ready', 'plan_sha256': file_hash(next_plan),
                    'boundary': {'exit_code': 0, 'legacy_queue_finished': str(finished), 'sha256': file_hash(finished)},
                    'paused_coordinators': [], 'gpu_pids': [], 'no_training_or_evaluator_child_signalled': True})
            active = RecoveryAssessment(next_plan, output/f'seed-{seed}', recovery)
            status(output, 'ASSESSMENT', seed=seed, paused_pids=[], readout_reused=seed == 404)
            with exclusive_writer(active.root):
                if seed == 404:
                    active.resume404()
                else:
                    active.fresh_completed_seed(finished)
            write_new_json(output/f'seed-{seed}'/'complete.json', {'status': 'complete', 'seed': seed})
        publish_cohort(original_path.parent)
        write_new_json(output/'complete.json', {'status': 'complete', 'seeds': [404, 505, 606],
            'same_coverage_rule': True, 'RL_never_interrupted_or_reexecuted': True})
        status(output, 'COMPLETE', seeds=[404, 505, 606], paused_pids=[])
    except BaseException as error:
        write_new_json(output/'stopped.json', {'error': f'{type(error).__name__}: {error}',
            'stage': active.active_stage if active else 'waiting_legacy_queue', 'automatic_retry': False})
        status(output, 'STOPPED', error=type(error).__name__, paused_pids=[])
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--prepare', type=Path); p.add_argument('--test-report', type=Path)
    p.add_argument('--execute', action='store_true'); p.add_argument('--detach', action='store_true')
    a = p.parse_args()
    if a.prepare:
        if a.execute or a.detach or not a.test_report:
            p.error('Preparation and execution are separate explicit operations')
        prepare(a.plan, a.prepare, a.test_report); return
    if not a.execute:
        print({'state': 'NOT_STARTED'}); return
    value = read_json(a.plan); output = Path(value['root'])
    if a.detach:
        with (output/'recovery.log').open('x') as stream:
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
            child = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'skillnet_cohort.first_calls_recovery',
                '--plan', str(a.plan.resolve()), '--execute'], cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_new_json(output/'supervisor.json', {'pid': child.pid, 'plan_sha256': file_hash(a.plan)})
        print({'state': 'DEFERRED', 'pid': child.pid}); return
    def stop(signum, frame):
        raise InterruptedError('Explicit recovery terminated; preserve evidence and do not retry')
    signal.signal(signal.SIGTERM, stop)
    with exclusive_writer(Path(value['original_plan']).parent), exclusive_writer(output):
        run(a.plan)


if __name__ == '__main__':
    main()
