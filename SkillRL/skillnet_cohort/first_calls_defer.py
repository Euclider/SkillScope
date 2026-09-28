"""Deferred, identity-bound GPU handoff AFTER seed505 RL exits successfully.

Only the two coordinator processes can be paused. Existing training/evaluation
children are never signalled. Paused coordinators are resumed even on failure.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import REPO, digest, exclusive_writer, file_hash, read_json, write_new_json


def process_identity(pid):
    try:
        root = Path('/proc')/str(int(pid))
        fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
        command = (root/'cmdline').read_bytes().split(b'\0')
        args = [s.decode(errors='replace') for s in command if s]
        module = args[args.index('-m')+1] if '-m' in args else None
        return {'pid': int(pid), 'parent': int(fields[1]), 'state': fields[0],
            'start_ticks': int(fields[19]), 'command_sha256': digest(args), 'module': module}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(binding):
    current = process_identity(binding['pid'])
    return current is not None and all(current[k] == binding[k]
        for k in ('pid', 'start_ticks', 'command_sha256', 'module'))


def bind_handoff(queue_pid, supervisor_pid, training_pid, cohort):
    pids = (queue_pid, supervisor_pid, training_pid)
    if len(set(pids)) != 3:
        raise ValueError('Three distinct coordinator/training identities required')
    identities = dict(zip(('queue', 'supervisor', 'training'), (process_identity(p) for p in pids)))
    modules = {'queue': 'skillnet_cohort.seed_queue', 'supervisor': 'skillnet_cohort.run',
               'training': 'skillnet_cohort.segmented_training'}
    for name, record in identities.items():
        if record is None or record['module'] != modules[name] or record['state'] in ('Z', 'T', 't'):
            raise ValueError('Handoff must be registered against the live, unpaused training queue')
        raw = (Path('/proc')/str(record['pid'])/'cmdline').read_bytes()
        required = str(cohort if name == 'queue' else Path(cohort)/'seed-505').encode()
        if required not in raw:
            raise ValueError('Handoff PID belongs to another cohort/seed')
    if identities['supervisor']['parent'] != queue_pid or identities['training']['parent'] != supervisor_pid:
        raise ValueError('Not the registered queue -> supervisor -> RL child tree')
    return identities


def training_boundary(plan):
    """Return None while RL runs; fail closed if RL exits without full success."""
    binding = plan['handoff_processes']['training']
    if same_process(binding) and process_identity(binding['pid'])['state'] != 'Z':
        return None
    root = Path(plan['cohort_root'])/'seed-505'
    legacy_plan = read_json(plan['legacy_queue_plan'])
    audit = Path(legacy_plan['recovery']['audit_root'])/'seed-505'
    path = root/'runtime-history.jsonl'
    # The ordinary seed supervisor writes telemetry under run_root even though
    # the queue's own stdout/launch receipt live under recovery-v4/seed-505.
    if not path.exists():
        path = audit/'runtime-history.jsonl'
    events = [json.loads(s) for s in path.read_text().splitlines() if s.endswith('}')]
    matching = [e for e in events if e.get('stage') == 'train-u0000-u0005'
        and any(c['pid'] == binding['pid'] for c in e.get('children', []))]
    if not matching or matching[-1]['event'] not in ('finished', 'stopped'):
        return None  # A successful child may exit one poll before the durable receipt.
    event = matching[-1]
    if event['event'] != 'finished' or any(c.get('exit_code') != 0 for c in event['children']):
        raise RuntimeError('seed505 RL failed; do not auto-retry or take its GPUs')
    from phase1.watch_qwen35_checkpoints import validate_full_checkpoint, read_committed_step
    checkpoint = root/'checkpoints/global_step_5'
    validation = validate_full_checkpoint(checkpoint)
    if validation['world_size'] != 8 or read_committed_step(root/'checkpoints') != 5:
        raise ValueError('seed505 must have a committed, complete eight-rank U5 checkpoint')
    metrics = [root/'metrics'/f'u{i:04d}.json' for i in range(1, 6)]
    if not all(p.is_file() for p in metrics):
        raise ValueError('seed505 has not completed all five registered updates')
    return {'training_child': binding, 'exit_code': 0, 'checkpoint': str(checkpoint),
        'checkpoint_structure': validation, 'metrics': [{'path': str(p), 'sha256': file_hash(p)} for p in metrics],
        'training_finish_event': event}


def descendants(pids):
    table = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                record = process_identity(int(path.name))
                if record is not None:
                    table[record['pid']] = record
            except (PermissionError, ProcessLookupError, ValueError):
                pass
    owned = set(pids)
    while True:
        expanded = owned | {pid for pid, p in table.items() if p['parent'] in owned}
        if expanded == owned:
            break
        owned = expanded
    return [table[p] for p in sorted(owned-set(pids)) if p in table and table[p]['state'] != 'Z']


def gpu_users():
    value = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=10)
    return sorted({int(s.strip()) for s in value.stdout.splitlines() if s.strip()})


def pause_coordinators(plan):
    if training_boundary(plan) is None:
        raise PermissionError('Never pause coordinators while RL is still running')
    paused = []
    try:
        for name in ('queue', 'supervisor'):
            binding = plan['handoff_processes'][name]
            if not same_process(binding) or process_identity(binding['pid'])['state'] in ('Z', 'T', 't'):
                raise ProcessLookupError('Coordinator changed/exited/already paused; refuse unrelated signals')
            paused.append(binding)
            os.kill(binding['pid'], signal.SIGSTOP)  # PID only, NEVER its process group.
        return paused
    except BaseException:
        resume_coordinators(paused)
        raise


def resume_coordinators(paused):
    for binding in reversed(paused):
        if same_process(binding):
            os.kill(binding['pid'], signal.SIGCONT)


def status(output, state, **extra):
    from phase1.archive import atomic_write_json, utc_now
    value = {'updated_at': utc_now(), 'state': state, 'automatic_retry': False, **extra}
    atomic_write_json(output/'deferred-status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run(path):
    path = Path(path).resolve(); plan = read_json(path); output = Path(plan['root'])
    from .seed_queue import verify_sources
    verify_sources(plan)
    if (output/'deferred-launch.json').exists():
        raise FileExistsError('This deferred attempt already ran; no automatic retry')
    write_new_json(output/'deferred-launch.json', {'pid': os.getpid(), 'plan_sha256': file_hash(path),
        'started_unix': time.time(), 'training_interruption_authorized': False})
    paused = []; child = None
    try:
        while (boundary := training_boundary(plan)) is None:
            supervisor = plan['handoff_processes']['supervisor']
            if not same_process(supervisor):
                raise RuntimeError('seed505 supervisor exited before a successful training boundary')
            status(output, 'WAITING_SEED505_RL', training_pid=plan['handoff_processes']['training']['pid'],
                paused_pids=[], gpu_experiment_started=False)
            time.sleep(30)
        paused = pause_coordinators(plan)
        write_new_json(output/'paused-coordinators.json', {'records': paused, 'boundary': boundary})
        while True:
            active = descendants([p['pid'] for p in paused]); users = gpu_users()
            if not active and not users:
                break
            # A CPU export or evaluation can start in the natural child-exit /
            # supervisor-poll gap. Let it finish without interruption.
            status(output, 'DRAINING_ALREADY_STARTED_POST_RL_CHILDREN', owned_children=active, gpu_pids=users)
            time.sleep(15)
        verify_sources(plan)
        write_new_json(output/'handoff.json', {'status': 'ready', 'plan_sha256': file_hash(path),
            'boundary': boundary, 'paused_coordinators': paused, 'gpu_pids': [],
            'no_training_or_evaluator_child_signalled': True})
        status(output, 'SEED404_ASSESSMENT', paused_pids=[p['pid'] for p in paused])
        with (output/'assessment.log').open('x') as stream:
            child = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'skillnet_cohort.first_calls_run',
                '--plan', str(path), '--execute'], cwd=REPO, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            write_new_json(output/'assessment-launch.json', {'pid': child.pid, 'plan_sha256': file_hash(path)})
            code = child.wait()
        if code:
            raise RuntimeError(f'Coverage assessment exited {code}; no automatic retry')
        write_new_json(output/'seed404-priority-complete.json', {'status': 'complete', 'seed404_reports_replaced': True})
    except BaseException as error:
        write_new_json(output/'deferred-stopped.json', {'error': f'{type(error).__name__}: {error}',
            'automatic_retry': False, 'training_not_restarted': True})
        raise
    finally:
        if child is not None and child.poll() is None:
            os.kill(child.pid, signal.SIGTERM)  # Own runner handles its own children.
            child.wait()
        resume_coordinators(paused)
        write_new_json(output/'coordinators-resumed.json', {'records': paused, 'resumed_unix': time.time(),
            'legacy_queue_continues_without_RL_restart': True})
        status(output, 'WAITING_LEGACY_QUEUE_FOR_FOLLOWUPS' if (output/'seed404-priority-complete.json').exists() else 'STOPPED',
            paused_pids=[], coordinators_resumed=bool(paused))


def run_followups(path):
    """Keep source-frozen RL untouched; later complete all seeds under one rule."""
    from .first_calls_protocol import prepare
    from .seed_queue import verify_sources
    from .first_calls_report import publish_cohort
    path = Path(path).resolve(); plan = read_json(path); output = Path(plan['root'])
    queue = plan['handoff_processes']['queue']; child = None
    try:
        while same_process(queue) and process_identity(queue['pid'])['state'] != 'Z':
            status(output, 'WAITING_LEGACY_QUEUE_FOR_FOLLOWUPS', paused_pids=[],
                registered_followup_seeds=plan['followup_seeds'])
            time.sleep(30)
        previous = read_json(plan['legacy_queue_plan'])
        audit = Path(previous['recovery']['audit_root'])
        finished = read_json(audit/'queue_finished.json')
        if finished['status'] != 'complete' or finished['completed_seeds'] != [404, 505, 606]:
            raise RuntimeError('Legacy queue incomplete; preserve evidence and do not restart failed RL')
        for seed in plan['followup_seeds']:
            verify_sources(plan)
            while gpu_users():
                status(output, 'WAITING_IDLE_GPUS', next_seed=seed, paused_pids=[])
                time.sleep(30)
            destination = output/f'followup-s{seed}'
            prepare(plan['legacy_queue_plan'], destination, seed=seed,
                test_report=plan['engineering_tests']['path'], policy_registration=path)
            next_plan = destination/'plan.json'
            write_new_json(destination/'handoff.json', {'status': 'ready', 'plan_sha256': file_hash(next_plan),
                'boundary': {'exit_code': 0, 'legacy_queue_finished': str(audit/'queue_finished.json'),
                    'sha256': file_hash(audit/'queue_finished.json')},
                'paused_coordinators': [], 'gpu_pids': [], 'no_training_or_evaluator_child_signalled': True})
            status(output, 'FOLLOWUP_ASSESSMENT', seed=seed, paused_pids=[])
            with (destination/'assessment.log').open('x') as stream:
                child = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'skillnet_cohort.first_calls_run',
                    '--plan', str(next_plan), '--execute'], cwd=REPO, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                write_new_json(destination/'assessment-launch.json', {'pid': child.pid, 'plan_sha256': file_hash(next_plan)})
                code = child.wait()
            if code:
                raise RuntimeError(f'seed{seed} coverage assessment failed; no automatic retry')
        publish_cohort(output)
        write_new_json(output/'deferred-complete.json', {'status': 'complete', 'seeds': [404, 505, 606],
            'same_coverage_rule': True, 'RL_never_interrupted_or_reexecuted': True})
        status(output, 'COMPLETE', seeds=[404, 505, 606], paused_pids=[])
    except BaseException as error:
        write_new_json(output/'followup-stopped.json', {'error': f'{type(error).__name__}: {error}',
            'automatic_retry': False})
        status(output, 'FOLLOWUP_STOPPED', paused_pids=[], error=type(error).__name__)
        raise
    finally:
        if child is not None and child.poll() is None:
            os.kill(child.pid, signal.SIGTERM); child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--detach', action='store_true')
    a = parser.parse_args(); plan = read_json(a.plan); output = Path(plan['root'])
    if not a.execute:
        print({'state': 'NOT_STARTED', 'wait_for_seed505_RL': True}); return
    if a.detach:
        with (output/'deferred.log').open('x') as stream:
            child = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'skillnet_cohort.first_calls_defer',
                '--plan', str(a.plan.resolve()), '--execute'], cwd=REPO, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_new_json(output/'deferred-supervisor.json', {'pid': child.pid, 'plan_sha256': file_hash(a.plan)})
        print({'state': 'DEFERRED_NOT_ASSESSING', 'pid': child.pid}); return
    def stop(signum, frame):
        raise InterruptedError('Deferred supervisor termination requested')
    signal.signal(signal.SIGTERM, stop)
    with exclusive_writer(output):
        run(a.plan)
        run_followups(a.plan)


if __name__ == '__main__':
    main()
