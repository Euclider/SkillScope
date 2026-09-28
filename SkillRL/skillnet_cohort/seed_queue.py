"""Explicitly authorized independent seed queue with shared storage protection.

Only timings/capacity from earlier completed seeds affect admission. Outcomes do
not affect seed order, methods, hyperparameters, or eligibility. No failed retry.
"""
from __future__ import annotations

import argparse
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

from .common import REPO, exclusive_writer, file_hash, load_preparation, read_json, write_new_json
from .runtime import disk_gate


SEEDS = [404, 505, 606]


def verify_sources(plan):
    for name, expected in plan.get('source_sha256', {}).items():
        source = (REPO / name).resolve()
        if not source.is_relative_to(REPO) or file_hash(source) != expected:
            raise ValueError('Registered runtime source changed; do not silently continue the queue')


def validate_plan(path):
    plan = read_json(path)
    verify_sources(plan)
    root = Path(plan['root']).resolve()
    if (plan.get('schema_version') != 'skillnet.phase12.independent_queue.v1'
            or plan.get('approved') is not True or plan.get('seeds') != SEEDS
            or (plan.get('hard_limit_seconds') != 108000 and not (
                'hard_limit_seconds' in plan and plan['hard_limit_seconds'] is None
                and plan.get('runtime_limit_override', {}).get('unlimited_wallclock_user_authorized') is True))
            or len(plan.get('jobs', [])) != 3):
        raise PermissionError('Explicit ordered three-seed and runtime-limit authorization required')
    if plan.get('recovery'):
        from .recover_queue import validate_recovery
        validate_recovery(plan)
    common = None
    for seed, job in zip(SEEDS, plan['jobs']):
        prep = Path(job['preparation'])
        load_preparation(prep)
        spec = read_json(prep.parent / 'spec.json')
        if (file_hash(prep) != job['preparation_sha256'] or spec['seed'] != seed
                or spec['training']['gpus'] != 8 or spec['training']['iterations'] != 5
                or spec['readout'].get('capture_scope') != 'window_start_old_only_v1'
                or Path(job['run_root']).resolve() != root / f'seed-{seed}'):
            raise ValueError('Changed seed preparation or unauthorized destination')
        identity = {k: spec[k] for k in ('bank_manifest_sha256', 'model_path', 'data_root',
                                       'router_profile_sha256', 'training', 'evaluation', 'readout', 'inference_profile')}
        if common is not None and identity != common:
            raise ValueError('Independent seed comparisons must share all other settings')
        common = identity
        evidence = read_json(job['admission'])
        if (file_hash(job['admission']) != job['admission_sha256']
                or evidence.get('preparation_sha256') != file_hash(prep) or evidence.get('status') != 'PASS'):
            raise ValueError('Missing measured seed admission')
    return plan


def next_estimate(initial, completed_seconds):
    # Recalibrate ONLY after an entire seed finishes; never on rewards or labels.
    return math.ceil(max(completed_seconds) * 1.25) if completed_seconds else math.ceil(initial)


def can_start(deadline, estimate, reserve, *, now=None):
    return deadline is None or deadline - (time.time() if now is None else now) >= estimate + reserve


def run(path):
    plan = validate_plan(path)
    root = Path(plan['root']).resolve()
    recovery = plan.get('recovery')
    audit_root = Path(recovery['audit_root']) if recovery else root
    if (root / 'queue_launch.json').exists() and not recovery:
        raise FileExistsError('Never implicitly resume/retry an existing seed queue')
    if (audit_root / 'queue_launch.json').exists():
        raise FileExistsError('This queue attempt has already started; no automatic retry')
    resumed_at = time.time()
    start = resumed_at - (recovery['previous_active_seconds'] if recovery else 0)
    deadline = None if plan['hard_limit_seconds'] is None else start + plan['hard_limit_seconds']
    write_new_json(audit_root / 'queue_launch.json', {'plan_sha256': file_hash(path), 'seeds': SEEDS,
        'started_unix': start, 'deadline_unix': deadline, 'pid': os.getpid(),
        'actual_attempt_started_unix': resumed_at, 'previous_active_seconds': resumed_at-start,
        'budget_scope': 'all three seeds together, previously consumed runtime retained', 'automatic_retry': False})
    durations, completed = [], []
    for seed, job in zip(SEEDS, plan['jobs']):
        verify_sources(plan)
        initial = read_json(job['admission'])
        estimate = next_estimate(initial['projected_total_upper_seconds'], durations)
        admission = {**initial, 'projected_total_upper_seconds': estimate,
            'initial_admission_sha256': job['admission_sha256'],
            'runtime_projection_basis': '1.25 x max completed seed walltime' if durations else 'component engineering measurements',
            'completed_seed_seconds': durations.copy(), 'outcome_metrics_consulted': False}
        if not can_start(deadline, estimate, plan['finish_reserve_seconds']):
            write_new_json(audit_root / f'not-started-{seed}.json', {'reason': 'remaining_shared_time_budget',
                'estimate_seconds': estimate, 'remaining_seconds': max(0, deadline-time.time()),
                'not_started_seeds': SEEDS[len(completed):], 'outcome_metrics_consulted': False})
            break
        limits = plan['storage']
        try:
            disk_gate(root, admission['projected_peak_run_bytes'] + limits['checkpoint_reserve_bytes'],
                      minimum_free_bytes=limits['minimum_free_bytes'], maximum_run_bytes=limits['maximum_run_bytes'])
        except OSError:
            write_new_json(audit_root / f'not-started-{seed}.json', {'reason': 'remaining_shared_disk_budget',
                'not_started_seeds': SEEDS[len(completed):], 'no_files_deleted': True})
            break
        actual_admission = audit_root / 'permits' / f'seed-{seed}-admission.json'
        write_new_json(actual_admission, admission)
        permit = {**plan['permit_template'], 'approved': True,
            'preparation_sha256': job['preparation_sha256'], 'run_root': job['run_root'],
            'router_cache_path': str(Path(job['run_root']) / 'router.sqlite3'),
            'storage': {**limits, 'cohort_storage_root': str(root)},
            'budget_admission': {'path': str(actual_admission), 'sha256': file_hash(actual_admission)},
            'budget_started_unix': start, 'budget_deadline_unix': deadline,
            'shared_queue_plan_sha256': file_hash(path)}
        if deadline is None:
            permit.update(unlimited_wallclock=True, shared_queue_plan=str(Path(path).resolve()))
        if recovery and seed == 404:
            if recovery.get('stage') == 'readout_aggregate':
                permit['readout_recovery'] = recovery['readout_binding']
                permit['operations'] = [op for op in permit['operations'] if op in ('evaluation', 'readout')]
            elif recovery.get('stage') == 'post_training_export':
                permit['post_training_recovery'] = recovery['endpoint_binding']
                # Defense in depth: the resumed seed has no authority to train.
                permit['operations'] = [op for op in permit['operations'] if op != 'training']
            else:
                permit['pre_optimizer_recovery'] = recovery['batch_binding']
        authorization = audit_root / 'permits' / f'seed-{seed}.json'
        write_new_json(authorization, permit)
        before = time.time()
        seed_root = Path(job['run_root'])
        seed_root.mkdir(parents=True, exist_ok=bool(recovery and seed == 404))
        seed_audit = audit_root / f'seed-{seed}' if recovery else seed_root
        seed_audit.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, '-u', '-B', '-m', 'skillnet_cohort.run', '--preparation', job['preparation'],
                '--root', job['run_root'], '--authorization', str(authorization), '--execute']
        with (seed_audit / 'supervisor.log').open('x') as stream:
            child = subprocess.Popen(args, cwd=REPO, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            write_new_json(seed_audit / 'supervisor_launch.json', {'pid': child.pid, 'started_unix': before,
                'queue_plan_sha256': file_hash(path), 'deadline_unix': deadline, 'command': args})
            while child.poll() is None:
                # Child enforces the same absolute deadline and stops its own job
                # process groups. This loop monitors it, without a separate retry.
                if deadline is not None and time.time() > deadline + 30:
                    print('Shared 30h hard timeout: terminating this queue\'s child supervisor.', flush=True)
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(1)
            child.wait()
            code = child.returncode
        elapsed = time.time() - before + (recovery['previous_active_seconds'] if recovery and seed == 404 else 0)
        write_new_json(audit_root / f'seed-{seed}-exit.json', {'seed': seed, 'exit_code': code,
            'elapsed_seconds': elapsed, 'completed': (seed_root / 'complete.json').is_file() and code == 0})
        if code != 0 or not (seed_root / 'complete.json').is_file():
            write_new_json(audit_root / 'queue_stopped.json', {'seed': seed, 'reason': 'seed_failed_or_incomplete',
                'completed_seeds': completed, 'automatic_retry': False, 'later_seeds_not_started': SEEDS[len(completed)+1:]})
            return 1
        completed.append(seed)
        durations.append(elapsed)
    from .reports import cohort_report
    cohort_report(root, plan, completed)
    write_new_json(audit_root / 'queue_finished.json', {'completed_seeds': completed,
        'not_completed_seeds': [seed for seed in SEEDS if seed not in completed],
        'status': 'complete' if completed == SEEDS else 'budget_stop', 'elapsed_seconds': time.time()-start,
        'no_evidence_deleted': True})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    plan = validate_plan(args.plan)
    if not args.execute:
        print({'status': 'validated_not_started', 'seeds': plan['seeds'],
               'shared_hours': None if plan['hard_limit_seconds'] is None else plan['hard_limit_seconds']/3600})
        return
    root = Path(plan['root'])
    if args.detach:
        root.mkdir(parents=True, exist_ok=bool(plan.get('recovery')))
        audit_root = Path(plan['recovery']['audit_root']) if plan.get('recovery') else root
        with (audit_root / 'queue.log').open('x') as stream:
            child = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'skillnet_cohort.seed_queue',
                '--plan', str(args.plan.resolve()), '--execute'], cwd=REPO, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_new_json(audit_root / 'queue_supervisor.json', {'pid': child.pid, 'plan_sha256': file_hash(args.plan)})
        print({'pid': child.pid, 'root': str(root), 'status': 'launched_not_complete'})
        return
    with exclusive_writer(root):
        raise SystemExit(run(args.plan))


if __name__ == '__main__':
    main()
