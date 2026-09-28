"""New explicit attempt for the five approved interrupted local router queries.

Reuse the source-frozen recovery workflow via a process-local assessment class
adapter. No frozen files or raw ledger rows are patched. The original queue's
606 storage stop remains authoritative; only completed 404/505 are assessed.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from .common import REPO, exclusive_writer, file_hash, read_json, write_new_json
from . import first_calls_recovery as previous
from .explicit_router_resume import audit_cache, read_only, row_signature, load_authorization

EXTRA_SOURCES = ('skillnet_cohort/explicit_router_resume.py', 'skillnet_cohort/first_calls_router_recovery.py')


def prepare(previous_path, output, test_report):
    from .seed_queue import verify_sources
    from .first_calls_defer import process_identity, gpu_users
    previous_path = Path(previous_path).resolve(); old = read_json(previous_path)
    old_root = previous_path.parent; output = Path(output).resolve()
    if output != old_root.parent/'recovery-v2' or output.exists():
        raise FileExistsError('Use a fresh all-first-calls-v1/recovery-v2 only once')
    verify_sources(old); previous.verify_retained(old['retained'])
    if process_identity(read_json(old_root/'supervisor.json')['pid']) or gpu_users():
        raise PermissionError('Previous recovery and all GPU consumers must have exited')
    failure = read_json(old_root/'active-monitor-handoff.json')
    if failure['failure'] != 'RouterCacheError: Prior failed/incomplete local query; no automatic retry':
        raise ValueError('Different failure needs separate review')
    suites = list(ET.parse(test_report).getroot().iter('testsuite'))
    if (not suites or any(int(s.attrib.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped'))
            or sum(int(s.attrib.get('tests', 0)) for s in suites) < 20):
        raise ValueError('Passing explicit-router-resume CPU regression is required')
    original = read_json(old['original_plan']); root = Path(original['jobs'][0]['run_root'])
    if (root/'evaluations/u0005').exists() or (root/'window_signals/u0000-u0005/prediction.json').exists():
        raise ValueError('Only the reviewed partial-U0 boundary is authorized')
    if (old_root.parent/'followup-s505').exists():
        raise FileExistsError('Never silently resume a started followup')
    db = root/'router.sqlite3'
    if (db.is_symlink() or file_hash(db) != failure['router_cache']['sha256']
            or any(db.with_name(db.name+s).exists() for s in ('-journal', '-wal', '-shm'))):
        raise ValueError('Reviewed quiescent router ledger changed or has sidecars')
    allowed = []
    with read_only(db) as c:
        if c.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
            raise ValueError('Router SQLite integrity failed')
        rows = c.execute('SELECT a.id,a.key,a.started,a.status,a.data FROM attempts a LEFT JOIN decisions d ON a.key=d.key WHERE d.key IS NULL ORDER BY a.id').fetchall()
        declared = failure['router_cache']['incomplete_attempts']
        if len(rows) != 5 or len(declared) != 5:
            raise PermissionError('Exactly the five user-approved interrupted queries are authorized')
        for row, item in zip(rows, declared):
            if tuple(row[:4]) != tuple(item[k] for k in ('id', 'key', 'started', 'status')) or row[3] != 'started':
                raise ValueError('Interrupted query identity changed')
            allowed.append({**item, 'row_sha256': row_signature(row)})
    record = previous.audit_endpoint(root, 0); readout = previous.validate_readout(root)
    previous.queue_finished(original)  # Validate the known 606 storage-only stop.
    retained = copy.deepcopy(old['retained'])
    retained['shards'] = record['shards']
    files = {v['path']: v for v in retained['files']}
    files.update({v['path']: v for v in record['files']})
    for p in (old_root).rglob('*'):
        if p.is_file() and p.name != '.writer.lock':
            files[str(p)] = {'path': str(p), 'sha256': file_hash(p)}
    output.mkdir(parents=True)
    snapshot = output/'router-before-resume.sqlite3'
    with db.open('rb') as source, snapshot.open('xb') as target:
        shutil.copyfileobj(source, target, 1024*1024); target.flush(); os.fsync(target.fileno())
    expected = failure['router_cache']['sha256']
    if file_hash(snapshot) != expected or file_hash(db) != expected:
        raise ValueError('Router ledger changed while archiving; do not continue')
    files[str(snapshot)] = {'path': str(snapshot), 'sha256': expected}
    config = read_json(root/'protocol.json')
    authorization = {'schema_version': 'skillnet.explicit_local_query_resume.v1', 'approved': True,
        'user_authority': '2026-09-21 用户明确允许修复这5条未完成本地查询后继续404/505；保留原账本及成功缓存',
        'recovery_root': str(output), 'run_root': str(root), 'cache_path': str(db),
        'protocol_hash': failure['router_cache']['protocol_hash'], 'max_local_calls': config['runtime']['max_local_calls'],
        'evaluation_protocol_sha256': file_hash(root/'protocol.json'),
        'snapshot': {'path': str(snapshot), 'sha256': expected}, 'allowed': allowed,
        'external_api_calls': 0, 'max_reissues_per_key': 1, 'automatic_retry': False,
        'original_attempt_rows_immutable': True, 'successful_decisions_immutable': True}
    write_new_json(output/'router-resume-authorization.json', authorization)
    proof = audit_cache(db, authorization)
    retained['files'] = list(files.values())
    plan = {**old, 'schema_version': 'skillnet.explicit_router_recovery.v2', 'root': str(output),
        'previous_attempt': {'path': str(previous_path), 'sha256': file_hash(previous_path)},
        'user_authority': authorization['user_authority'], 'retained': retained, 'readout_audit': readout,
        'endpoint_summary': {k: v for k, v in record.items() if k != 'files'},
        'source_sha256': {**old['source_sha256'], **{p: file_hash(REPO/p) for p in EXTRA_SOURCES}},
        'router_resume': {'path': str(output/'router-resume-authorization.json'),
            'sha256': file_hash(output/'router-resume-authorization.json')},
        'router_preflight': proof,
        'engineering_tests': {'path': str(Path(test_report).resolve()), 'sha256': file_hash(test_report)}}
    instance = previous.RecoveryAssessment(old['original_plan'], output/'seed-404', plan)
    plan['capacity_admission'] = instance.disk(original['projected_recording_upper_bytes'])
    write_new_json(output/'plan.json', plan)
    print(json.dumps({'state': 'PREPARED_NOT_STARTED', 'plan': str(output/'plan.json'),
        'completed_U0': record['completed'], 'missing_U0': record['missing'], 'authorized_local_queries': 5}), flush=True)
    return plan


class RouterResumeAssessment(previous.RecoveryAssessment):
    def commands(self, jobs):
        from .runtime_watch import StageWatch
        from .seed_queue import verify_sources
        if any(args[0] not in ('phase2.evaluate', 'phase2.aggregate', 'skillnet_cohort.first_calls_measure') for args, _, _ in jobs):
            raise PermissionError('Only registered missing readout/utility commands are permitted')
        verify_sources(self.plan); verify_sources(self.recovery); self.disk()
        config = read_json(self.root/'protocol.json'); authorization = None
        if self.plan['jobs'][0]['seed'] == 404:
            authorization = load_authorization(self.recovery['router_resume']['path'])
        before = audit_cache(config['runtime']['cache_path'], authorization)
        self.active_stage = ','.join(label for _, label, _ in jobs)
        write_new_json(self.audit/'cache-admission'/(self.active_stage+'.json'), before)
        watch = StageWatch(self.root, self.audit, self.active_stage, None)
        children = []; next_disk = 0
        try:
            for arguments, label, gpu in jobs:
                arguments = list(arguments)
                if authorization is not None and arguments[0] == 'phase2.evaluate':
                    arguments = ['skillnet_cohort.explicit_router_resume',
                        '--router-resume-authorization', self.recovery['router_resume']['path'], *arguments[1:]]
                path = self.audit/'logs'/(label+'.log'); path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.open('x'); env = os.environ.copy()
                env.update(TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), PYTHONPATH=str(REPO)+os.pathsep+env.get('PYTHONPATH', ''))
                env.pop('SKILLNET_PHASE12_RECOVERY_AUTHORIZATION', None)
                try:
                    child = subprocess.Popen([sys.executable, '-u', '-B', '-m', *map(str, arguments)], cwd=REPO,
                        env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
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
                    os.killpg(p.pid, signal.SIGTERM)  # Only this attempt's new child sessions.
            for p, stream, _ in children:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait()
                stream.close()
            watch.tick(children, event='finished' if all(p.returncode == 0 for p, _, _ in children) else 'stopped', force=True)
        if authorization is not None:
            write_new_json(self.audit/'cache-after'/(self.active_stage+'.json'), audit_cache(config['runtime']['cache_path'], authorization))


def run(path):
    plan = read_json(path)
    from .seed_queue import verify_sources
    verify_sources(plan)
    binding = plan['previous_attempt']
    if file_hash(binding['path']) != binding['sha256']:
        raise ValueError('Previous failed attempt changed')
    authorization = load_authorization(plan['router_resume']['path'])
    audit_cache(authorization['cache_path'], authorization)
    # Explicit, process-local workflow adaptation; frozen source bytes are not
    # changed. Restore the class even on error. No legacy worker is running.
    original = previous.RecoveryAssessment
    previous.RecoveryAssessment = RouterResumeAssessment
    try:
        previous.run(path)
    finally:
        previous.RecoveryAssessment = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--prepare', type=Path)
    p.add_argument('--test-report', type=Path); p.add_argument('--execute', action='store_true')
    p.add_argument('--detach', action='store_true'); a = p.parse_args()
    if a.prepare:
        if a.execute or a.detach or not a.test_report:
            p.error('Prepare and execute are separate operations')
        prepare(a.plan, a.prepare, a.test_report); return
    if not a.execute:
        print({'state': 'NOT_STARTED'}); return
    plan = read_json(a.plan); output = Path(plan['root'])
    if a.detach:
        with (output/'recovery.log').open('x') as stream:
            env = os.environ.copy(); env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
            child = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'skillnet_cohort.first_calls_router_recovery',
                '--plan', str(a.plan.resolve()), '--execute'], cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_new_json(output/'supervisor.json', {'pid': child.pid, 'plan_sha256': file_hash(a.plan)})
        print({'state': 'STARTED', 'pid': child.pid}); return
    def stop(signum, frame):
        raise InterruptedError('Explicit router recovery stopped; no automatic retry')
    signal.signal(signal.SIGTERM, stop)
    with exclusive_writer(Path(plan['original_plan']).parent), exclusive_writer(output):
        run(a.plan)


if __name__ == '__main__':
    main()
