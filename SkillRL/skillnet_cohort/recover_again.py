"""One user-authorized v2 continuation of the pre-optimizer bool failure."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .common import REPO, file_hash, read_json, write_new_bytes, write_new_json


CHANGED = {'skillnet_cohort/runtime.py', 'skillnet_cohort/run.py', 'skillnet_cohort/recover_queue.py',
    'skillnet_cohort/rollout_recovery.py', 'skillnet_cohort/segmented_training.py', 'skillnet_cohort/reports.py',
    'verl/trainer/ppo/ray_trainer.py', 'verl/trainer/ppo/metric_utils.py'}
NEW = {'skillnet_cohort/recover_again.py', 'skillnet_cohort/recovery_forward.py',
       'skillnet_cohort/runtime_watch.py', 'verl/trainer/ppo/metric_utils.py'}


def consumed_seconds(root):
    """v1 exit elapsed ALREADY contains original attempt runtime: never add twice."""
    prior = Path(root)/'recovery-v1'
    previous = read_json(prior/'seed-404-exit.json')
    if previous['exit_code'] != 1 or previous['completed'] is not False:
        raise ValueError('Expected the documented pre-optimizer failure')
    return previous['elapsed_seconds'] + (
        read_json(prior/'seed-404/supervisor_launch.json')['started_unix']
        - read_json(prior/'queue_launch.json')['actual_attempt_started_unix'])


def validate(plan):
    v = plan['recovery']
    root = Path(plan['root']).resolve()
    if (v.get('schema_version') != 'skillnet.phase12.explicit_recovery.v2'
            or v.get('user_authorized') is not True or v.get('exclude_failed_downtime') is not True
            or Path(v['audit_root']).resolve() != root/'recovery-v2'
            or file_hash(v['prior_plan']) != v['prior_plan_sha256']
            or v['previous_active_seconds'] != consumed_seconds(root)):
        raise PermissionError('Explicit second recovery and cumulative-runtime binding required')
    prior = read_json(v['prior_plan'])
    for key in ('root', 'seeds', 'storage', 'permit_template', 'hard_limit_seconds', 'finish_reserve_seconds'):
        if plan[key] != prior[key]:
            raise ValueError('Recovery may not change experimental scope or resource limits')
    for i, job in enumerate(plan['jobs']):
        before = prior['jobs'][i]
        if {k: x for k, x in job.items() if k not in ('admission', 'admission_sha256')} != {
                k: x for k, x in before.items() if k not in ('admission', 'admission_sha256')}:
            raise ValueError('Recovery changed a frozen scientific preparation')
        if i and job != before:
            raise ValueError('Unstarted seeds are unchanged')
    if any((root/f'seed-{s}').exists() for s in (505, 606)) or any(
            (root/name/'queue_finished.json').exists() for name in ('recovery-v1', 'recovery-v2')):
        raise ValueError('Continuation only covers the observed first seed stop')
    from .rollout_recovery import verify_binding
    manifest = verify_binding(v['batch_binding'], root/'seed-404', plan['jobs'][0]['preparation'])
    if not v['batch_binding'].get('reuse_complete_old') or manifest.get('complete_old_rows') != 5512:
        raise ValueError('Must reuse the fully verified original OLD batch')
    for item in v['retained_files']:
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Preserved failure evidence changed: '+item['path'])
    return plan


def prepare(prior_plan, manifest, output):
    prior_plan, manifest, output = (Path(p).resolve() for p in (prior_plan, manifest, output))
    prior = read_json(prior_plan)
    root = Path(prior['root'])
    audit = root/'recovery-v2'
    if (prior_plan != root/'recovery-v1/cohort-recovery.json'
            or output != audit/'cohort-recovery.json' or output.exists()):
        raise FileExistsError('Use a new explicitly authorized recovery-v2 plan only once')
    saved = read_json(manifest)
    if saved.get('complete_old_rows') != 5512 or saved.get('reuses_exact_trainer_chosen') is not True:
        raise ValueError('Complete native OLD verification must precede preparation')
    cpu = read_json(audit/'actual-batch-cpu-audit.json')
    from xml.etree import ElementTree
    suites = ElementTree.parse(audit/'offline-full-01.xml').getroot()
    if (cpu.get('status') != 'PASS' or cpu.get('rows') != 5512
            or cpu.get('source_manifest_sha256') != file_hash(manifest)
            or not list(suites.iter('testcase'))
            or suites.find('.//failure') is not None or suites.find('.//error') is not None):
        raise ValueError('Real-batch CPU and full regression must pass before GPU continuation')
    plan = copy.deepcopy(prior)
    changed = []
    for name, old in prior['source_sha256'].items():
        now = file_hash(REPO/name)
        if old != now:
            if name not in CHANGED:
                raise ValueError('Unrelated runtime modification: '+name)
            changed.append({'path': name, 'before_sha256': old, 'after_sha256': now})
            plan['source_sha256'][name] = now
    for name in NEW:
        plan['source_sha256'][name] = file_hash(REPO/name)
    admission = read_json(prior['jobs'][0]['admission'])
    old_forward_evidence = REPO.parent/'vllm-capture-preflight-20260918-v1/verification-audit.json'
    # The original projection used the maximum 6,400 rows / 128 per block.
    # Remove only its allocated U1 OLD forward and compression, not reference.
    old_forward_seconds = 50*read_json(old_forward_evidence)['stage_max_seconds']['old_exact_forward_and_io']
    compressed_seconds = admission['component_seconds']['start_capture_compression']
    witness_allowance = 60.
    admission.update(projected_total_upper_seconds=admission['projected_total_upper_seconds']
        - old_forward_seconds-compressed_seconds+witness_allowance,
        previous_admission_sha256=prior['jobs'][0]['admission_sha256'],
        recovery_projection='v1 remaining conditional estimate minus retained complete U1 OLD+compression; add 60s native witness',
        recovered_stage_projection_seconds={'old_forward': old_forward_seconds, 'compression': compressed_seconds},
        native_witness_allowance_seconds=witness_allowance, reference_forward_still_required=True,
        actual_batch_cpu_verified=True, outcome_metrics_consulted=False)
    admission['sources'] += [{'path': str(manifest), 'sha256': file_hash(manifest)}]
    new_admission = audit/'remaining-s404-admission.json'
    write_new_json(new_admission, admission)
    plan['jobs'][0].update(admission=str(new_admission), admission_sha256=file_hash(new_admission))
    source = root/'seed-404'
    preserved = [root/'recovery-v1/queue_launch.json', root/'recovery-v1/queue_stopped.json',
        root/'recovery-v1/seed-404-exit.json', root/'recovery-v1/seed-404/stopped.json',
        root/'recovery-v1/seed-404/segments/u0000-u0005.json',
        root/'recovery-v1/seed-404/logs/train-u0000-u0005.log',
        root/'recovery-v1/seed-404/supervisor.log', source/'pre_forward_batches/u0001.pt',
        Path(saved['batch_path']), Path(saved['old_probability_cache']), manifest,
        audit/'actual-batch-cpu-audit.json', audit/'offline-full-01.xml']
    preserved += sorted((root/'recovery-v1/seed-404/forward_progress').glob('*.json'))
    retained = prior['recovery']['retained_files'] + [{'path': str(p), 'sha256': file_hash(p)} for p in preserved]
    if any((source/name).exists() for name in ('router.sqlite3-wal', 'router.sqlite3-journal')):
        raise ValueError('Router ledger must be quiescent before failure snapshot')
    snapshot = audit/'failure-snapshot/router.sqlite3'
    write_new_bytes(snapshot, (source/'router.sqlite3').read_bytes())
    retained.append({'path': str(snapshot), 'sha256': file_hash(snapshot)})
    plan['recovery'] = {'schema_version': 'skillnet.phase12.explicit_recovery.v2',
        'user_authorized': True,
        'user_authority': '继续修复重启，老是报错中断你是不是得监控优化下；扣除故障停机时间，累计运行仍不超过30小时',
        'exclude_failed_downtime': True, 'previous_active_seconds': consumed_seconds(root),
        'audit_root': str(audit), 'prior_plan': str(prior_plan), 'prior_plan_sha256': file_hash(prior_plan),
        'changed_sources': changed, 'retained_files': retained,
        'batch_binding': {**prior['recovery']['batch_binding'], 'manifest': str(manifest),
            'sha256': file_hash(manifest), 'attempt_dir': str(audit/'seed-404'), 'reuse_complete_old': True}}
    validate(plan)
    write_new_json(output, plan)
    print({'status': 'PREPARED_NOT_LAUNCHED', 'previous_active_seconds': consumed_seconds(root),
        'remaining_seconds': 108000-consumed_seconds(root),
        'remaining_first_seed_conditional_estimate': admission['projected_total_upper_seconds'],
        'plan': str(output)}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prior-plan', 'manifest', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.prior_plan, a.manifest, a.output)
