"""One explicit 404-correction -> 505-correction -> 505-resume sequence."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import REPO, exclusive_writer, file_hash, read_json, write_new_json
from .numerical_readout import binding, paused_tree, disk, SOURCES, F


def status(plan, stage, **extra):
    from phase1.archive import atomic_write_json, utc_now
    value = {'updated_at': utc_now(), 'stage': stage, 'automatic_retry': False, **extra}
    atomic_write_json(Path(plan['root'])/'runtime-status.json', value)
    print(value, flush=True)


def command_jobs(path, seed, mode, shards=()):
    if seed not in (404, 505) or mode not in ('measure', 'aggregate', 'report'):
        raise PermissionError('No RL, export, environment evaluator or other command is permitted')
    if mode == 'measure':
        if not shards or any(r not in range(8) for r in shards) or len(set(shards)) != len(shards):
            raise ValueError('Distinct registered shards required')
        if len({r % 4 for r in shards}) != len(shards):
            raise ValueError('GPU pairs must not overlap')
        return [(['skillnet_cohort.numerical_readout', '--plan', str(path), '--seed', str(seed),
                  '--measure', '--shard', str(r)], f'seed{seed}-measure-shard{r}',
                 f'{2*(r%4)},{2*(r%4)+1}') for r in shards]
    if shards:
        raise ValueError('CPU stages cannot take GPU shards')
    module = 'skillnet_cohort.numerical_readout_report' if mode == 'report' else 'skillnet_cohort.numerical_readout'
    args = [module, '--plan', str(path), '--seed', str(seed)]
    if mode == 'aggregate':
        args.append('--aggregate')
    return [(args, f'seed{seed}-{mode}', '')]


def commands(path, jobs):
    plan = binding(path); disk(plan)
    allowed = []
    for seed in (404, 505):
        for r in range(8):
            allowed.extend(command_jobs(path, seed, 'measure', [r]))
        for mode in ('aggregate', 'report'):
            allowed.extend(command_jobs(path, seed, mode))
    if not jobs or any(j not in allowed for j in jobs) or len(set(j[1] for j in jobs)) != len(jobs):
        raise PermissionError('Only approved numerical stages may execute')
    children = []; next_check = 0
    try:
        for args, label, gpu in jobs:
            log = Path(plan['root'])/'logs'/(label+'.log'); log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open('x')
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                HF_HUB_OFFLINE='1', PYTHONPATH=str(REPO)+os.pathsep+env.get('PYTHONPATH', ''))
            env.pop('SKILLNET_PHASE12_RECOVERY_AUTHORIZATION', None)
            try:
                child = subprocess.Popen([sys.executable, '-u', '-B', '-m', *args], cwd=REPO,
                    env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
            except BaseException:
                stream.close(); raise
            children.append((child, stream, label, log))
        while any(p.poll() is None for p, _, _, _ in children):
            if any(p.poll() not in (None, 0) for p, _, _, _ in children):
                raise RuntimeError('Numerical child failed; preserve evidence and do not retry')
            if time.time() >= next_check:
                disk(plan)
                if not (Path(plan['root'])/'seed505-resumed.json').exists():
                    paused_tree(plan)
                status(plan, ','.join(x[2] for x in children), children=[
                    {'pid': p.pid, 'label': label, 'exit_code': p.poll(), 'log_bytes': log.stat().st_size}
                    for p, _, label, log in children])
                next_check = time.time()+30
            time.sleep(1)
        if any(p.returncode != 0 for p, _, _, _ in children):
            raise RuntimeError('Numerical stage failed; see its retained log')
    finally:
        for p, _, _, _ in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)  # Only newly created numerical workers.
        for p, stream, _, _ in children:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            stream.close()


def verify_report_completion(root):
    root = Path(root); complete = read_json(root/'complete.json')
    if complete['status'] != 'complete' or complete['all_comparators_present'] is not True:
        raise PermissionError('Numerical report is not complete')
    provenance = root/'report-provenance.json'
    if file_hash(provenance) != complete['provenance_sha256']:
        raise ValueError('Numerical report receipt changed')
    for row in read_json(provenance)['files']:
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != row['sha256']:
            raise ValueError('Numerical report output changed')
    return complete


def resume_505(path):
    plan = binding(path); root = Path(plan['root'])
    verify_report_completion(root/'seed-404')
    stable = root/'seed-505/committed.json'; commit = read_json(stable)
    if (commit['seed'] != 505 or commit['all_legacy_scalar_signals_exact'] is not True
            or file_hash(root/'seed-505/skill_context_features.parquet') != commit['features_sha256']):
        raise PermissionError('505 stable readout must be committed before utility resumes')
    if (root/'seed505-resume-intent.json').exists() or (root/'seed505-resumed.json').exists():
        raise FileExistsError('Never implicitly repeat a resume attempt')
    disk(plan); tree = paused_tree(plan)
    from .first_calls_port_recovery import binding as port_binding
    port_binding(F/'recovery-v3/plan.json')  # Original frozen code still executable.
    from .first_calls_defer import gpu_users
    pause = read_json(plan['pause']['path'])
    engines = {p['pid'] for p in tree if p['parent'] != pause['supervisor_pid'] and p['pid'] != pause['supervisor_pid']}
    if set(gpu_users()) != engines:
        raise PermissionError('Numerical GPUs must be released; no unrelated GPU consumers allowed')
    intent = {'created_unix': time.time(), 'processes': tree,
        'authority': plan['user_authority'], 'seed404_complete_sha256': file_hash(root/'seed-404/complete.json'),
        'seed505_stable_commit_sha256': file_hash(stable), 'only_signal': 'SIGCONT',
        'pause_seconds': time.time()-datetime.fromisoformat(pause['observed_at_utc']).timestamp(),
        'no_new_cache_query_permissions': True, 'no_new_environment_command': True}
    write_new_json(root/'seed505-resume-intent.json', intent)
    # Engines first, then their clients, then the stage coordinator. pidfds
    # bind signals to the checked processes instead of a recyclable PID name.
    order = ([p for p in tree if p['pid'] in engines]
             +[p for p in tree if p['parent'] == pause['supervisor_pid']]
             +[p for p in tree if p['pid'] == pause['supervisor_pid']])
    fds = []
    try:
        for item in order:
            fd = os.pidfd_open(item['pid']); fds.append(fd)
        paused_tree(plan)
        for fd in fds:
            signal.pidfd_send_signal(fd, signal.SIGCONT)
    finally:
        for fd in fds:
            os.close(fd)
    write_new_json(root/'seed505-resumed.json', {**intent, 'resumed_unix': time.time(), 'status': 'resumed'})
    status(plan, 'SEED505_EXISTING_UTILITY_RESUMED', supervisor_pid=pause['supervisor_pid'])


def wait_505(path):
    plan = binding(path); output = Path(plan['root']); previous = F/'recovery-v3'
    from .first_calls_defer import process_identity
    parent = read_json(plan['pause']['path'])['supervisor_pid']
    while True:
        if (previous/'stopped.json').exists():
            raise RuntimeError('Resumed505 failed; no automatic retry or interrupted-query reissue')
        if (previous/'seed-505/complete.json').exists():
            if read_json(previous/'seed-505/complete.json')['status'] != 'complete':
                raise ValueError('Invalid 505 completion')
            break
        current = process_identity(parent)
        expected = read_json(plan['pause']['path'])['after'][0]
        if (current is None or current['state'] in ('Z', 'T', 't')
                or current['start_ticks'] != expected['start_ticks']
                or current['command_sha256'] != expected['command_sha256']):
            raise ProcessLookupError('505 coordinator exited, paused or changed before completion')
        disk(plan)
        status(plan, 'WAITING_FOR_SEED505_UTILITY', supervisor_pid=parent,
            original_runtime_path=str(previous/'seed-505/runtime-status.json'))
        time.sleep(30)
    status(plan, 'SEED505_UTILITY_COMPLETE_CORRECTED_REPORT_PENDING')


def pipeline(path):
    plan = binding(path); paused_tree(plan)
    for seed in (404, 505):
        if seed == 505:
            verify_report_completion(Path(plan['root'])/'seed-404')
        for shards in (range(4), range(4, 8)):
            commands(path, command_jobs(path, seed, 'measure', shards))
        commands(path, command_jobs(path, seed, 'aggregate'))
        if seed == 404:
            commands(path, command_jobs(path, seed, 'report'))
            verify_report_completion(Path(plan['root'])/'seed-404')
        else:
            resume_505(path); wait_505(path)
            commands(path, command_jobs(path, seed, 'report'))
            verify_report_completion(Path(plan['root'])/'seed-505')


def run(path):
    plan = binding(path); output = Path(plan['root'])
    if (output/'launch.json').exists():
        raise FileExistsError('Workflow already attempted; no automatic retry')
    paused_tree(plan); disk(plan)
    write_new_json(output/'launch.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'plan_sha256': file_hash(path), 'automatic_retry': False})
    try:
        with exclusive_writer(output):
            pipeline(path)
        write_new_json(output/'complete.json', {'status': 'complete', 'seeds': [404, 505],
            'original_reports_preserved': True, 'new_RL': False, 'seed606_started': False})
        status(plan, 'COMPLETE_404_505_NUMERICAL_CORRECTION')
    except BaseException as error:
        write_new_json(output/'stopped.json', {'error': f'{type(error).__name__}: {error}',
            'stopped_unix': time.time(), 'automatic_retry': False,
            'seed505_resume_intent_exists': (output/'seed505-resume-intent.json').exists(),
            'seed505_resume_already_sent': (output/'seed505-resumed.json').exists()})
        status(plan, 'STOPPED', error=repr(error)); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--execute', action='store_true')
    p.add_argument('--detach', action='store_true'); a = p.parse_args()
    if not a.execute:
        p.error('Explicit --execute required')
    plan = binding(a.plan); output = Path(plan['root'])
    if a.detach:
        paused_tree(plan)
        with (output/'workflow.log').open('x') as stream:
            env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                PYTHONPATH=str(REPO)+os.pathsep+env.get('PYTHONPATH', ''))
            child = subprocess.Popen([sys.executable, '-u', '-B', '-m',
                'skillnet_cohort.numerical_readout_run', '--plan', str(a.plan.resolve()), '--execute'],
                cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        write_new_json(output/'supervisor.json', {'pid': child.pid, 'plan_sha256': file_hash(a.plan)})
        print({'state': 'STARTED', 'pid': child.pid}, flush=True); return
    def stop(signum, frame):
        raise InterruptedError('Explicitly stopped numerical workflow')
    signal.signal(signal.SIGTERM, stop)
    run(a.plan)


if __name__ == '__main__':
    main()
