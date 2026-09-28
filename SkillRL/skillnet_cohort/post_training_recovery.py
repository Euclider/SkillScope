"""Explicit v3 continuation after completed U5 training; never authorizes retraining."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from functools import lru_cache
import math
from pathlib import Path
from xml.etree import ElementTree

from .common import REPO, file_hash, read_json, write_new_bytes, write_new_json

CHANGED = {'phase2/export_model.py', 'skillnet_cohort/checkpoints.py', 'skillnet_cohort/run.py',
           'skillnet_cohort/seed_queue.py', 'skillnet_cohort/recover_queue.py',
           'skillnet_cohort/reports.py', 'skillnet_cohort/runtime_watch.py'}
NEW = {'skillnet_cohort/post_training_recovery.py', 'phase2/export_verify.py', 'scripts/model_merger.py'}


def consumed_seconds(root):
    prior = Path(root)/'recovery-v2'
    result = read_json(prior/'seed-404-exit.json')
    if result['exit_code'] != 1 or result['completed'] is not False:
        raise ValueError('Expected the documented post-training export failure')
    # This elapsed time already includes v0 and v1; do not double-charge them.
    return result['elapsed_seconds'] + (
        read_json(prior/'seed-404/supervisor_launch.json')['started_unix']
        - read_json(prior/'queue_launch.json')['actual_attempt_started_unix'])


def assert_training_complete(source):
    import json
    from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
    source = Path(source)
    if {p.name for p in (source/'metrics').glob('u*.json')} != {f'u{i:04d}.json' for i in range(1, 6)}:
        raise ValueError('All five retained RL iteration metrics are required')
    for update in range(1, 6):
        metrics = read_json(source/'metrics'/f'u{update:04d}.json')
        if metrics['global_update'] != update or any(isinstance(v, (int, float)) and not math.isfinite(v)
                                                   for v in metrics['metrics'].values()):
            raise ValueError('Invalid or non-finite retained training metrics')
        trajectories = read_json(source/'rollout_summaries'/f'u{update:04d}-train.json')['trajectories']
        if len(trajectories) != 128:
            raise ValueError('A completed iteration must have its original 16x8 trajectories')
    counts = []
    for rank in range(8):
        paths = sorted((source/'optimizer_steps').glob(f'u*-rank{rank}.jsonl'))
        if [p.name for p in paths] != [f'u{i:04d}-rank{rank}.jsonl' for i in range(1, 6)]:
            raise ValueError('Missing or extra optimizer iteration logs')
        rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
        if len(rows) != 204 or any(r['rank'] != rank or r['adam_step_before'] != i
                or r['adam_step_after'] != i+1 or not math.isfinite(r['grad_norm'])
                or r['learning_rate'] != 1e-6 for i, r in enumerate(rows)):
            raise ValueError('The retained 204-step native optimizer boundary differs')
        counts.append(len(rows))
    native = source/'checkpoints/global_step_5'
    metadata = validate_full_checkpoint(native)
    if metadata['world_size'] != 8 or (source/'checkpoints/latest_checkpointed_iteration.txt').read_text().strip() != '5':
        raise ValueError('U5 was not committed at the native eight-rank boundary')
    for prefix in ('model', 'optim', 'extra_state'):
        if {p.name for p in (native/'actor').glob(f'{prefix}_world_size_8_rank_*.pt')} != {
                f'{prefix}_world_size_8_rank_{rank}.pt' for rank in range(8)}:
            raise ValueError('Missing native checkpoint rank')
    return {'iterations': 5, 'trajectories': 640, 'optimizer_steps_each_rank': counts, 'checkpoint': metadata}


def audit_training(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    source, prior = root/'seed-404', root/'recovery-v2'
    if output != root/'recovery-v3/retained-training.json' or output.exists():
        raise FileExistsError('Use a fresh v3 retained-training manifest only once')
    summary = assert_training_complete(source)
    if read_json(prior/'seed-404/stopped.json')['stage'] != 'export-u0005':
        raise ValueError('This continuation is only for the documented U5 export failure')
    if any((source/name).exists() for name in ('evaluations', 'windows', 'reports', 'complete.json')):
        raise ValueError('Evaluation has already started; reconcile it instead of repeating it')
    old_partial = source/'models/u0005.partial'
    if (source/'models/u0005').exists() or not old_partial.is_dir() or list(old_partial.iterdir()):
        raise ValueError('Expected no U5 export and the retained empty failed staging directory')
    paths = []
    for directory in ('checkpoints/global_step_5', 'models/u0000', 'metrics', 'optimizer_steps',
                      'rollout_summaries', 'rollout_progress', 'batches/u0001'):
        paths += [p for p in (source/directory).rglob('*') if p.is_file()]
    paths += [source/'checkpoints/latest_checkpointed_iteration.txt', source/'launch.json',
              source/'stopped.json', source/'resource_limits.json', source/'segments/u0000-u0005.json']
    paths += [prior/'cohort-recovery.json', prior/'queue_launch.json', prior/'queue_stopped.json',
        prior/'seed-404-exit.json', prior/'seed-404/supervisor_launch.json', prior/'seed-404/stopped.json',
        prior/'seed-404/logs/train-u0000-u0005.log', prior/'seed-404/logs/export-u0005.log',
        prior/'seed-404/runtime-status.json', prior/'seed-404/runtime-history.jsonl', prior/'seed-404/supervisor.log']
    def item(path):
        return {'path': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size}
    with ThreadPoolExecutor(max_workers=4) as pool:
        retained = list(pool.map(item, sorted(set(paths))))
    if any((source/name).exists() for name in ('router.sqlite3-wal', 'router.sqlite3-journal')):
        raise ValueError('Router ledger must be quiescent before snapshot')
    snapshot = output.parent/'failure-snapshot/router.sqlite3'
    write_new_bytes(snapshot, (source/'router.sqlite3').read_bytes())
    retained.append(item(snapshot))
    value = {'schema_version': 'skillnet.phase12.retained_u5.v1', 'source_root': str(source),
        'preparation_sha256': read_json(prior/'cohort-recovery.json')['jobs'][0]['preparation_sha256'],
        'summary': summary, 'retained_files': retained, 'old_partial': str(old_partial),
        'old_partial_contents': [], 'scientific_outcomes_consulted_for_design': False}
    write_new_json(output, value)
    print({'status': 'TRAINING_PRESERVED', 'files': len(retained), 'summary': summary,
           'manifest': str(output)}, flush=True)


@lru_cache(maxsize=2)
def verified_manifest(path, expected):
    if file_hash(path) != expected:
        raise ValueError('Retained-training manifest changed')
    value = read_json(path)
    for row in value['retained_files']:
        if file_hash(row['path']) != row['sha256']:
            raise ValueError('Retained training/failure evidence changed: '+row['path'])
    return value


def verify_binding(binding, source, preparation):
    source = Path(source).resolve()
    value = verified_manifest(binding['manifest'], binding['sha256'])
    audit = source.parent/'recovery-v3'
    if (value['schema_version'] != 'skillnet.phase12.retained_u5.v1'
            or Path(value['source_root']) != source or source.name != 'seed-404'
            or value['preparation_sha256'] != file_hash(preparation)
            or Path(binding['attempt_dir']).resolve() != audit/'seed-404'
            or Path(binding['export_staging']).resolve() != audit/'seed-404/u0005.partial'
            or binding.get('completed_through_update') != 5
            or binding.get('skip_training') is not True):
        raise PermissionError('Post-training continuation must bind the retained U5 and a fresh attempt')
    old_partial = Path(value['old_partial'])
    if not old_partial.is_dir() or list(old_partial.iterdir()):
        raise ValueError('Preserve the previous failed export directory unchanged')
    assert_training_complete(source)
    return value


def validate(plan):
    v, root = plan['recovery'], Path(plan['root']).resolve()
    if (v.get('schema_version') != 'skillnet.phase12.explicit_recovery.v3'
            or v.get('stage') != 'post_training_export' or v.get('user_authorized') is not True
            or v.get('exclude_failed_downtime') is not True
            or Path(v['audit_root']).resolve() != root/'recovery-v3'
            or Path(v['prior_plan']).resolve() != root/'recovery-v2/cohort-recovery.json'
            or file_hash(v['prior_plan']) != v['prior_plan_sha256']
            or v['previous_active_seconds'] != consumed_seconds(root)):
        raise PermissionError('Explicit post-training continuation and cumulative budget required')
    prior = read_json(v['prior_plan'])
    for key in ('root', 'seeds', 'storage', 'permit_template', 'hard_limit_seconds', 'finish_reserve_seconds'):
        if plan[key] != prior[key]:
            raise ValueError('Do not change scientific scope or resource limits on recovery')
    for i, job in enumerate(plan['jobs']):
        allowed = ('admission', 'admission_sha256') if i == 0 else ()
        if {k: x for k, x in job.items() if k not in allowed} != {
                k: x for k, x in prior['jobs'][i].items() if k not in allowed}:
            raise ValueError('Frozen seed preparations must be unchanged')
    if any((root/f'seed-{s}').exists() for s in (505, 606)) or (root/'recovery-v3/queue_finished.json').exists():
        raise ValueError('Only the documented seed404 export stop may be resumed')
    verify_binding(v['endpoint_binding'], root/'seed-404', plan['jobs'][0]['preparation'])
    return plan


def prepare(prior_plan, manifest, output):
    prior_plan, manifest, output = (Path(p).resolve() for p in (prior_plan, manifest, output))
    prior = read_json(prior_plan)
    root = Path(prior['root']); audit = root/'recovery-v3'
    if prior_plan != root/'recovery-v2/cohort-recovery.json' or output != audit/'cohort-recovery.json' or output.exists():
        raise FileExistsError('Create only a new authorized recovery-v3 plan')
    tests = ElementTree.parse(audit/'offline-full-01.xml').getroot()
    if not list(tests.iter('testcase')) or tests.find('.//failure') is not None or tests.find('.//error') is not None:
        raise ValueError('CPU regression must pass first')
    plan = copy.deepcopy(prior); changed = []
    for name, before in prior['source_sha256'].items():
        after = file_hash(REPO/name)
        if before != after:
            if name not in CHANGED:
                raise ValueError('Unrelated runtime modification: '+name)
            changed.append({'path': name, 'before_sha256': before, 'after_sha256': after})
            plan['source_sha256'][name] = after
    for name in NEW:
        plan['source_sha256'][name] = file_hash(REPO/name)
    admission = read_json(prior['jobs'][0]['admission'])
    # Keep all original post-training allocations, including native validation,
    # already-completed cold initialization and checkpoint overhead: conservative.
    components = admission['component_seconds']
    remaining = {k: components[k] for k in ('evaluation_and_monitoring',
        'readout_six_forwards_scoring_and_decode_allowance', 'cold_initialization_export_checkpoint_report_allowance')}
    admission.update(projected_total_upper_seconds=sum(remaining.values()),
        previous_admission_sha256=prior['jobs'][0]['admission_sha256'],
        recovery_projection='all original post-training components retained; completed training not repeated',
        remaining_component_seconds=remaining, complete_training_retained=True,
        outcome_metrics_consulted=False)
    admission['sources'] += [{'path': str(manifest), 'sha256': file_hash(manifest)}]
    target = audit/'remaining-s404-admission.json'; write_new_json(target, admission)
    plan['jobs'][0].update(admission=str(target), admission_sha256=file_hash(target))
    plan['recovery'] = {'schema_version': 'skillnet.phase12.explicit_recovery.v3',
        'stage': 'post_training_export', 'user_authorized': True,
        'user_authority': '分析错误并继续运行；seed404的5轮RL已完成，卡在训练后的模型导出；累计运行30h扣除故障停机',
        'exclude_failed_downtime': True, 'previous_active_seconds': consumed_seconds(root),
        'audit_root': str(audit), 'prior_plan': str(prior_plan), 'prior_plan_sha256': file_hash(prior_plan),
        'changed_sources': changed,
        'endpoint_binding': {'manifest': str(manifest), 'sha256': file_hash(manifest),
            'attempt_dir': str(audit/'seed-404'), 'export_staging': str(audit/'seed-404/u0005.partial'),
            'completed_through_update': 5, 'skip_training': True}}
    validate(plan); write_new_json(output, plan)
    print({'status': 'PREPARED_NOT_LAUNCHED', 'previous_active_seconds': consumed_seconds(root),
        'remaining_seconds': 108000-consumed_seconds(root), 'remaining_first_seed_estimate': sum(remaining.values()),
        'plan': str(output)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    audit = sub.add_parser('audit'); audit.add_argument('--root', required=True, type=Path); audit.add_argument('--output', required=True, type=Path)
    prep = sub.add_parser('prepare')
    for name in ('prior-plan', 'manifest', 'output'):
        prep.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    if args.operation == 'audit':
        audit_training(args.root, args.output)
    else:
        prepare(args.prior_plan, args.manifest, args.output)
