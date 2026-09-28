"""Prepare one explicitly authorized pre-optimizer recovery; never launch on import."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .common import REPO, file_hash, read_json, write_new_bytes, write_new_json


CHANGED_SOURCES = {'skillnet_cohort/runtime.py', 'skillnet_cohort/run.py',
    'skillnet_cohort/seed_queue.py', 'skillnet_cohort/segmented_training.py', 'skillnet_cohort/reports.py',
    'verl/trainer/ppo/ray_trainer.py', 'phase2/capture.py'}


def validate_recovery(plan):
    value = plan['recovery']
    if value.get('schema_version') == 'skillnet.phase12.explicit_recovery.v4':
        from .readout_recovery import validate
        return validate(plan)
    if value.get('schema_version') == 'skillnet.phase12.explicit_recovery.v3':
        from .post_training_recovery import validate
        return validate(plan)
    if value.get('schema_version') == 'skillnet.phase12.explicit_recovery.v2':
        from .recover_again import validate
        return validate(plan)
    root = Path(plan['root']).resolve()
    if (value.get('schema_version') != 'skillnet.phase12.explicit_recovery.v1'
            or value.get('user_authorized') is not True
            or value.get('exclude_failed_downtime') is not True
            or Path(value['audit_root']).resolve() != root/'recovery-v1'
            or file_hash(value['original_plan']) != value['original_plan_sha256']
            or file_hash(root/'queue_launch.json') != value['original_queue_launch_sha256']
            or file_hash(root/'queue_stopped.json') != value['original_queue_stopped_sha256']):
        raise PermissionError('Missing explicit recovery and cumulative-runtime authorization')
    original = read_json(value['original_plan'])
    if any(plan[k] != original[k] for k in ('root', 'seeds', 'storage', 'permit_template', 'hard_limit_seconds', 'finish_reserve_seconds')):
        raise ValueError('Recovery must not change experimental scope or resource limits')
    old_exit = read_json(root/'seed-404-exit.json')
    expected_spent = old_exit['elapsed_seconds'] + (
        read_json(root/'seed-404/supervisor_launch.json')['started_unix'] - read_json(root/'queue_launch.json')['started_unix'])
    if value['previous_active_seconds'] != expected_spent:
        raise ValueError('Previously consumed runtime may not be reset')
    if (root/'queue_finished.json').exists() or any((root/f'seed-{s}').exists() for s in (505, 606)):
        raise ValueError('Only the observed initial seed404 stop is covered by this recovery')
    from .rollout_recovery import verify_binding
    verify_binding(value['batch_binding'], root/'seed-404', plan['jobs'][0]['preparation'])
    for item in value['retained_files']:
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Previously retained recovery evidence changed')


def prepare(original_plan, manifest, output):
    original_plan, manifest, output = map(lambda x: Path(x).resolve(), (original_plan, manifest, output))
    original = read_json(original_plan)
    root = Path(original['root'])
    audit_root = root/'recovery-v1'
    if output != audit_root/'cohort-recovery.json' or output.exists():
        raise FileExistsError('Use the new recovery-v1/cohort-recovery.json only once')
    source = root/'seed-404'
    old_exit = read_json(root/'seed-404-exit.json')
    spent = old_exit['elapsed_seconds'] + (read_json(source/'supervisor_launch.json')['started_unix']
                                         - read_json(root/'queue_launch.json')['started_unix'])
    plan = copy.deepcopy(original)
    changed = []
    for name, old in original['source_sha256'].items():
        now = file_hash(REPO/name)
        if now != old:
            if name not in CHANGED_SOURCES:
                raise ValueError(f'Unrelated source modification is not authorized by this repair: {name}')
            changed.append({'path': name, 'before_sha256': old, 'after_sha256': now})
            plan['source_sha256'][name] = now
    for name in ('skillnet_cohort/recover_queue.py', 'skillnet_cohort/rollout_recovery.py'):
        plan['source_sha256'][name] = file_hash(REPO/name)
    # Subtract only the walltime of the already-completed, retained rollout,
    # using first-step publication -> complete summary as a conservative lower
    # bound. Initialization and partial OLD forwards are NOT discounted.
    saved_seconds = (source/'rollout_summaries/u0001-train.json').stat().st_mtime - (
        source/'rollout_progress/u0001/step-0000.json').stat().st_mtime
    if not 0 < saved_seconds < old_exit['elapsed_seconds']:
        raise ValueError('Invalid retained-stage timing evidence')
    admission = read_json(plan['jobs'][0]['admission'])
    admission.update(projected_total_upper_seconds=admission['projected_total_upper_seconds']-saved_seconds,
        previous_admission_sha256=plan['jobs'][0]['admission_sha256'],
        recovery_projection='original conditional projection minus completed retained rollout only',
        retained_completed_stage_seconds=saved_seconds, outcome_metrics_consulted=False)
    new_admission = audit_root/'remaining-s404-admission.json'
    write_new_json(new_admission, admission)
    plan['jobs'][0].update(admission=str(new_admission), admission_sha256=file_hash(new_admission))
    # Freeze evidence before continuation. The mutable ledger/progress files are
    # archived separately; original snapshots, logs, journal and old rows stay.
    preserved = [source/'stopped.json', source/'launch.json', source/'resource_limits.json',
        source/'segments/u0000-u0005.json', source/'logs/train-u0000-u0005.log',
        source/'logs/export-u0000.log', source/'supervisor.log']
    preserved += sorted((source/'models/u0000').rglob('*'))
    retained = [{'path': str(p), 'sha256': file_hash(p)} for p in preserved if p.is_file()]
    # The stopped ledger has no WAL and no writer; keep its exact pre-resume
    # bytes so future cache/stat updates cannot erase the failure-time snapshot.
    if any((source/name).exists() for name in ('router.sqlite3-wal', 'router.sqlite3-journal')):
        raise ValueError('Reconcile SQLite journal before taking an immutable stopped-ledger copy')
    snapshots = [source/'router.sqlite3', *sorted((source/'forward_progress').glob('*.json'))]
    for old_path in snapshots:
        snapshot = audit_root/'failure-snapshot'/old_path.relative_to(source)
        write_new_bytes(snapshot, old_path.read_bytes())
        retained.append({'path': str(snapshot), 'sha256': file_hash(snapshot)})
    plan['recovery'] = {'schema_version': 'skillnet.phase12.explicit_recovery.v1',
        'user_authorized': True, 'user_authority': '任务好像异常停止了，修复并继续跑；扣除故障停机时间，累计运行仍不超过30小时',
        'exclude_failed_downtime': True, 'previous_active_seconds': spent, 'audit_root': str(audit_root),
        'original_plan': str(original_plan), 'original_plan_sha256': file_hash(original_plan),
        'original_queue_launch_sha256': file_hash(root/'queue_launch.json'),
        'original_queue_stopped_sha256': file_hash(root/'queue_stopped.json'),
        'changed_sources': changed, 'retained_files': retained,
        'batch_binding': {'manifest': str(manifest), 'sha256': file_hash(manifest),
            'attempt_dir': str(audit_root/'seed-404'),
            'original_launch_sha256': file_hash(source/'launch.json'),
            'original_stopped_sha256': file_hash(source/'stopped.json')}}
    validate_recovery(plan)
    write_new_json(output, plan)
    print({'status': 'PREPARED_NOT_LAUNCHED', 'previous_active_seconds': spent,
           'remaining_seconds': 108000-spent, 'remaining_first_seed_estimate': admission['projected_total_upper_seconds'],
           'plan': str(output)}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original-plan', 'manifest', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.original_plan, a.manifest, a.output)
