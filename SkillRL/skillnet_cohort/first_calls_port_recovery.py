"""Explicit seed505 utility-only continuation after the reviewed EADDRINUSE.

Source-frozen 404/505 training, completed readout and earlier evaluators are not
edited. A new attempt resumes existing first-call trajectories using a vLLM
UniProc subclass with per-engine FileStore rendezvous. No automatic retry, API,
new RL, full-performance resampling, seed404 republication or seed606 launch.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from .common import REPO, exclusive_writer, file_hash, read_json, write_new_json
from .explicit_router_resume import audit_cache
from .first_calls_recovery import (RecoveryAssessment, audit_endpoint,
    validate_readout, verify_retained, queue_finished)
from .seed_queue import verify_sources

EXECUTOR = 'skillnet_cohort.vllm_file_executor.FileStoreUniProcExecutor'
RECORD_ENV = 'SKILLSCOPE_EVAL_RENDEZVOUS_RECORD'
EXTRA_SOURCES = ('skillnet_cohort/vllm_file_executor.py',
                 'skillnet_cohort/first_calls_port_recovery.py')
SCHEMA = 'skillnet.seed505_port_recovery.v3'


def binding(path):
    path = Path(path).resolve(); plan = read_json(path)
    output = Path(plan['root']); source = Path(plan['assessment_plan']['path'])
    if (plan.get('schema_version') != SCHEMA or plan.get('approved') is not True
            or plan.get('seeds') != [505] or plan.get('automatic_retry') is not False
            or plan.get('external_api_calls') != 0 or plan.get('training_reexecuted') is not False
            or plan.get('seed606_storage_waiver') is not False
            or path != output/'plan.json' or output.name != 'recovery-v3'
            or source != output.parent/'followup-s505/plan.json'
            or plan.get('executor') != EXECUTOR):
        raise PermissionError('Only the explicit seed505 utility continuation is authorized')
    for field in ('assessment_plan', 'previous_attempt', 'engineering_tests'):
        item = plan[field]
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Changed recovery binding: '+field)
    original = read_json(source)
    if (len(original['jobs']) != 1 or original['jobs'][0]['seed'] != 505
            or Path(original['jobs'][0]['run_root']) != source.parent/'seed-505'
            or file_hash(source.parent/'seed-505/protocol.json') != plan['protocol_sha256']):
        raise PermissionError('Wrong seed, root or evaluation protocol')
    verify_sources(original); verify_sources(plan)
    for item in plan['dependency_sources']:
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Installed rendezvous implementation changed')
    return plan, original


def dependency_sources():
    names = {'vllm': ('vllm/v1/executor/uniproc_executor.py',
                     'vllm/v1/executor/abstract.py', 'vllm/distributed/parallel_state.py',
                     'vllm/utils/network_utils.py', 'vllm/config/parallel.py'),
             'torch': ('torch/distributed/rendezvous.py',)}
    return [{'path': str(importlib.metadata.distribution(package).locate_file(name)),
             'sha256': file_hash(importlib.metadata.distribution(package).locate_file(name))}
            for package, files in names.items() for name in files]


def prepare(previous_path, output, test_report):
    from .assets import model_inventory
    from .first_calls_run import verify_legacy_seal
    from .first_calls_defer import gpu_users, process_identity
    from phase2.protocol import validate_extended
    previous_path = Path(previous_path).resolve(); output = Path(output).resolve()
    if (previous_path.parent.name != 'recovery-v2'
            or output != previous_path.parent.parent/'recovery-v3' or output.exists()):
        raise FileExistsError('Use a fresh recovery-v3 after the reviewed recovery-v2 failure')
    old = read_json(previous_path); verify_sources(old)
    if process_identity(read_json(previous_path.parent/'supervisor.json')['pid']) or gpu_users():
        raise PermissionError('Old supervisor and GPU consumers must have exited')
    stopped = read_json(previous_path.parent/'stopped.json')
    log = previous_path.parent/'seed-505/logs/utility-u0000-shard7.log'
    if (not stopped['stage'].startswith('utility-u0000-shard')
            or 'EADDRINUSE' not in log.read_text() or 'port: 40669' not in log.read_text()):
        raise ValueError('Different failure requires separate review')
    suites = list(ET.parse(test_report).getroot().iter('testsuite'))
    if (not suites or sum(int(s.attrib.get('tests', 0)) for s in suites) < 20
            or any(int(s.attrib.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped'))):
        raise ValueError('Passing rendezvous/recovery CPU regression required')
    source = output.parent/'followup-s505/plan.json'; plan = read_json(source)
    verify_sources(plan)
    if len(plan['jobs']) != 1 or plan['jobs'][0]['seed'] != 505:
        raise PermissionError('Only the existing seed505 followup may resume')
    root = Path(plan['jobs'][0]['run_root'])
    if root != source.parent/'seed-505':
        raise PermissionError('Unexpected seed505 result root')
    for relative in ('evaluations/u0005', 'window_signals/u0000-u0005/prediction.json',
                     'complete.json', 'sealed.json', 'analysis-complete.json', 'reports'):
        if (root/relative).exists():
            raise FileExistsError('Only the reviewed pre-U0-completion boundary may resume')
    config = read_json(root/'protocol.json'); validate_extended(config, REPO)
    cache = audit_cache(root/'router.sqlite3')
    if cache['attempts'] != 0 or cache['successful_decisions'] != 0:
        raise ValueError('Reviewed pre-generation router ledger must still be empty')
    for update in (0, 5):
        if model_inventory(root/'models'/f'u{update:04d}') != plan['model_inventory'][f'u{update:04d}']:
            raise ValueError('Retained model weights changed')
    verify_legacy_seal(plan)
    endpoint = audit_endpoint(root, 0); readout = validate_readout(root)
    if (endpoint['completed'], endpoint['expected']) != (540, 3636):
        raise ValueError('Reviewed 540 retained / 3636 total boundary changed')
    registration = read_json(plan['policy_registration'])
    if file_hash(plan['policy_registration']) != plan['policy_registration_sha256']:
        raise ValueError('Earlier coverage registration changed')
    finished = queue_finished(registration)
    if finished is None:
        raise PermissionError('Legacy queue must have exited naturally')
    completed404 = output.parent/'seed-404'
    receipt404 = read_json(completed404/'complete.json')
    if receipt404['reports_published'] is not True or receipt404['status'] != 'complete':
        raise ValueError('Preserved seed404 must already be complete')
    files = {v['path']: v for v in endpoint['files']}
    # Bind every sealed seed404 artifact; do not recalculate or republish it.
    for row in read_json(completed404/'sealed.json')['files']:
        p = completed404/row['path']
        if not p.resolve().is_relative_to(completed404) or file_hash(p) != row['sha256']:
            raise ValueError('Completed seed404 seal changed')
        files[str(p)] = {'path': str(p), 'sha256': row['sha256']}
    for row in read_json(completed404/'report-publication.json')['files']:
        if file_hash(row['path']) != row['sha256']:
            raise ValueError('Published seed404 report changed')
        files[row['path']] = row
    paths = [source, source.parent/'permit.json', source.parent/'handoff.json',
        root/'protocol.json', root/'manifest.json', root/'launch.json', root/'readout-reuse.json',
        root/'reuse/u0000.json', completed404/'complete.json', completed404/'sealed.json',
        completed404/'report-publication.json', Path(finished)]
    for folder in (previous_path.parent, root/'support', root/'signals', root/'window_signals',
                   source.parent/'archived-reports', output.parent/'archived-reports'):
        paths.extend(p for p in folder.rglob('*') if p.is_file() and p.name != '.writer.lock')
    for p in paths:
        files[str(p)] = {'path': str(p), 'sha256': file_hash(p)}
    for row in plan['legacy_reports']:
        if file_hash(row['path']) != row['sha256'] or file_hash(row['archive']) != row['sha256']:
            raise ValueError('Original seed505 reports and backups must remain intact')
    value = {'schema_version': SCHEMA, 'approved': True, 'root': str(output), 'seeds': [505],
        'user_authority': '2026-09-21 用户明确要求修复505第7分片vLLM端口冲突并续跑505；不重跑RL或已完成轨迹',
        'previous_attempt': {'path': str(previous_path), 'sha256': file_hash(previous_path)},
        'assessment_plan': {'path': str(source), 'sha256': file_hash(source)},
        'source_sha256': {**old['source_sha256'], **plan['source_sha256'],
                         **{p: file_hash(REPO/p) for p in EXTRA_SOURCES}},
        'dependency_sources': dependency_sources(), 'executor': EXECUTOR,
        'protocol_sha256': file_hash(root/'protocol.json'),
        'retained': {'files': list(files.values()), 'shards': endpoint['shards']},
        'endpoint_summary': {k: v for k, v in endpoint.items() if k != 'files'},
        'readout_audit': readout, 'router_preflight': cache,
        'router_at_preparation_sha256': file_hash(root/'router.sqlite3'),
        'engineering_tests': {'path': str(Path(test_report).resolve()), 'sha256': file_hash(test_report)},
        'automatic_retry': False, 'external_api_calls': 0, 'training_reexecuted': False,
        'completed_readout_reexecuted': False, 'performance_resampled': False,
        'seed404_immutable': True, 'seed606_storage_waiver': False, 'hard_limit_seconds': None,
        'scientific_protocol_unchanged': True, 'scientific_verification_status': 'UNVERIFIED'}
    instance = RecoveryAssessment(source, output/'seed-505', value)
    value['capacity_admission'] = instance.disk(plan['projected_recording_upper_bytes'])
    write_new_json(output/'plan.json', value)
    print(json.dumps({'state': 'PREPARED_NOT_STARTED', 'plan': str(output/'plan.json'),
        'retained_U0': endpoint['completed'], 'missing_U0': endpoint['missing'],
        'source_files': len(value['source_sha256']), 'retained_files': len(files)}), flush=True)
    return value


def build_file_engine(model_path, profile):
    """Same frozen build_engine arguments except the explicit executor subclass."""
    from .inference import validate, require_versions
    from .vllm_backend import text_config
    validate({'inference_profile': profile}); settings = profile['settings']; require_versions(settings)
    from vllm import LLM, ModelRegistry
    ModelRegistry.register_model('SkillScopeQwen35Text',
                                'skillnet_cohort.vllm_qwen35:SkillScopeQwen35Text')
    options = {key: settings[key] for key in (
        'dtype', 'max_model_len', 'max_num_batched_tokens', 'max_num_seqs',
        'gpu_memory_utilization', 'enforce_eager', 'enable_prefix_caching', 'enable_chunked_prefill')}
    return LLM(model=str(model_path), tokenizer=str(model_path), tensor_parallel_size=1,
        distributed_executor_backend=EXECUTOR, hf_overrides=text_config, load_format='auto',
        enable_sleep_mode=False, seed=settings['seed'], generation_config='vllm',
        disable_log_stats=True, **options)


def evaluate(path, root, update, shard):
    plan, original = binding(path); root = Path(root).resolve()
    if (str(root) != original['jobs'][0]['run_root'] or update not in (0, 5)
            or shard not in range(8) or os.environ.get('CUDA_VISIBLE_DEVICES') != str(shard)):
        raise PermissionError('Wrong evaluation endpoint, seed root or GPU shard')
    runtime = read_json(root/'protocol.json')['runtime']
    if runtime['router_backend'] != 'skillrl_embedding_state_batch' or runtime['max_api_calls'] != 0:
        raise PermissionError('Only the unchanged local embedding router is permitted')
    audit = Path(plan['root'])/'seed-505/rendezvous'
    record = audit/f'u{update:04d}-shard{shard}.json'
    ready = audit/f'u{update:04d}-shard{shard}-ready.json'
    if record.exists() or ready.exists():
        raise FileExistsError('Shard already attempted; never retry implicitly')
    audit.mkdir(parents=True, exist_ok=True)
    os.environ[RECORD_ENV] = str(record)
    from . import vllm_backend
    original_policy = vllm_backend.VLLMPolicy
    class FilePolicy(original_policy):
        def __init__(self, checkpoint, profile):
            self.engine = build_file_engine(checkpoint, profile)
            self.tokenizer = self.engine.get_tokenizer()
            write_new_json(ready, {'state': 'ENGINE_READY', 'pid': os.getpid(),
                'plan_sha256': file_hash(path), 'rendezvous_sha256': file_hash(record),
                'update': update, 'shard': shard, 'sampling_implementation_unchanged': True})
    argv = sys.argv
    vllm_backend.VLLMPolicy = FilePolicy
    try:
        sys.argv = ['phase2.evaluate', '--root', str(root), '--update', str(update),
                    '--shards', '8', '--shard', str(shard)]
        from phase2.evaluate import main as evaluate_main
        evaluate_main()
    finally:
        vllm_backend.VLLMPolicy = original_policy
        sys.argv = argv


class PortRecoveryAssessment(RecoveryAssessment):
    def commands(self, jobs):
        from .runtime_watch import StageWatch
        expected = [(['phase2.evaluate', '--root', self.root, '--update', u,
            '--shards', 8, '--shard', rank], f'utility-u{u:04d}-shard{rank}', rank)
            for u in (0, 5) for rank in range(8)]
        if (not jobs or any(job not in expected for job in jobs)
                or len({label for _, label, _ in jobs}) != len(jobs)):
            raise PermissionError('Only missing seed505 utility shards may execute')
        binding(Path(self.recovery['root'])/'plan.json'); self.disk()
        before = audit_cache(self.root/'router.sqlite3')
        self.active_stage = ','.join(label for _, label, _ in jobs)
        write_new_json(self.audit/'cache-admission'/(self.active_stage+'.json'), before)
        watch = StageWatch(self.root, self.audit, self.active_stage, None)
        children = []; next_disk = 0
        try:
            for args, label, gpu in jobs:
                path = self.audit/'logs'/(label+'.log'); path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.open('x'); env = os.environ.copy()
                env.update(TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO)+os.pathsep+env.get('PYTHONPATH', ''))
                env.pop('SKILLNET_PHASE12_RECOVERY_AUTHORIZATION', None)
                env.pop(RECORD_ENV, None)
                command = [sys.executable, '-u', '-B', '-m', 'skillnet_cohort.first_calls_port_recovery',
                    '--plan', str(Path(self.recovery['root'])/'plan.json'), '--evaluate',
                    '--root', str(self.root), '--update', str(args[4]), '--shard', str(gpu)]
                try:
                    child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    stream.close(); raise
                children.append((child, stream, label))
            watch.tick(children, event='started', force=True)
            while any(p.poll() is None for p, _, _ in children):
                if any(p.poll() not in (None, 0) for p, _, _ in children):
                    raise RuntimeError('Utility child failed; preserve evidence, no automatic retry')
                if time.time() >= next_disk:
                    self.disk(); next_disk = time.time()+60
                watch.tick(children); time.sleep(1)
            if any(p.returncode != 0 for p, _, _ in children):
                raise RuntimeError('Utility stage failed; see preserved logs')
        finally:
            for p, _, _ in children:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)  # Only this attempt's own child sessions.
            for p, stream, _ in children:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait()
                stream.close()
            watch.tick(children, event='finished' if all(p.returncode == 0 for p, _, _ in children) else 'stopped', force=True)
        write_new_json(self.audit/'cache-after'/(self.active_stage+'.json'), audit_cache(self.root/'router.sqlite3'))

    def resume505(self):
        from .first_calls_run import lock_prediction
        from .first_calls_reuse import import_endpoint
        from .first_calls_report import report, publish
        from .window_storage import seal_window
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


def run(path):
    from .first_calls_defer import gpu_users, status
    plan, original = binding(path); output = Path(plan['root'])
    if (output/'launch.json').exists():
        raise FileExistsError('Recovery already attempted; no automatic retry')
    if gpu_users():
        raise PermissionError('GPUs must be idle before the explicit restart')
    verify_retained(plan['retained'])
    root = Path(original['jobs'][0]['run_root'])
    if file_hash(root/'router.sqlite3') != plan['router_at_preparation_sha256']:
        raise ValueError('Quiescent router changed after preparation')
    cache = audit_cache(root/'router.sqlite3')
    if cache['unresolved']:
        raise PermissionError('No new interrupted-query retries are authorized')
    validate_readout(root)
    endpoint = audit_endpoint(root, 0)
    if endpoint['completed'] != plan['endpoint_summary']['completed']:
        raise ValueError('Evaluation boundary changed after preparation')
    if (root/'evaluations/u0005').exists() or (root/'window_signals/u0000-u0005/prediction.json').exists():
        raise ValueError('Recovery must start before opening U5')
    active = PortRecoveryAssessment(plan['assessment_plan']['path'], output/'seed-505', plan)
    active.disk(original['projected_recording_upper_bytes'])
    write_new_json(output/'launch.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'plan_sha256': file_hash(path), 'training_reexecuted': False, 'automatic_retry': False})
    try:
        status(output, 'ASSESSMENT', seed=505, paused_pids=[], readout_reused=True)
        with exclusive_writer(root):
            active.resume505()
        write_new_json(output/'seed-505/complete.json', {'status': 'complete', 'seed': 505})
        write_new_json(output/'pending.json', {'pending_seeds': [606], 'completed_assessments': [404, 505],
            'reason': 'legacy_seed_not_started_due_to_storage_guard', 'no_storage_waiver': True,
            'no_evidence_deleted': True, 'no_automatic_RL_launch': True})
        status(output, 'BLOCKED_LEGACY_STORAGE', pending_seed=606, completed_assessments=[404, 505],
               paused_pids=[], no_automatic_RL_launch=True)
    except BaseException as error:
        write_new_json(output/'stopped.json', {'stage': active.active_stage,
            'error': f'{type(error).__name__}: {error}', 'automatic_retry': False})
        status(output, 'STOPPED', error=type(error).__name__, paused_pids=[])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--prepare', type=Path)
    parser.add_argument('--test-report', type=Path); parser.add_argument('--execute', action='store_true')
    parser.add_argument('--detach', action='store_true'); parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--root', type=Path); parser.add_argument('--update', type=int); parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    if args.prepare:
        if args.execute or args.detach or args.evaluate or not args.test_report:
            parser.error('Preparation and execution are separate operations')
        prepare(args.plan, args.prepare, args.test_report); return
    if args.evaluate:
        if args.execute or args.detach or args.root is None or args.update is None or args.shard is None:
            parser.error('Explicit evaluation requires root, endpoint and shard only')
        evaluate(args.plan, args.root, args.update, args.shard); return
    if not args.execute:
        print({'state': 'NOT_STARTED'}); return
    plan, _ = binding(args.plan); output = Path(plan['root'])
    if args.detach:
        with (output/'recovery.log').open('x') as stream:
            env = os.environ.copy(); env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='')
            env.pop('SKILLNET_PHASE12_RECOVERY_AUTHORIZATION', None)
            child = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'skillnet_cohort.first_calls_port_recovery',
                '--plan', str(args.plan.resolve()), '--execute'], cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_new_json(output/'supervisor.json', {'pid': child.pid, 'plan_sha256': file_hash(args.plan)})
        print({'state': 'STARTED', 'pid': child.pid}); return
    def stop(signum, frame):
        raise InterruptedError('Explicit seed505 recovery stopped; no automatic retry')
    signal.signal(signal.SIGTERM, stop)
    with exclusive_writer(output.parent), exclusive_writer(output):
        run(args.plan)


if __name__ == '__main__':
    main()
