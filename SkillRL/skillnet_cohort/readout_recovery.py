"""One explicit OLD-only readout continuation; completed work is immutable."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree

from .common import REPO, digest, file_hash, read_json, write_new_bytes, write_new_json
from .post_training_recovery import assert_training_complete

CHANGED = {'phase2/aggregate.py', 'skillnet_cohort/common.py', 'skillnet_cohort/day_budget.py',
           'skillnet_cohort/run.py', 'skillnet_cohort/seed_queue.py', 'skillnet_cohort/recover_queue.py',
           'skillnet_cohort/reports.py'}
NEW = {'phase2/window_evidence.py', 'skillnet_cohort/parameter_delta.py',
       'skillnet_cohort/readout_recovery.py'}


def consumed_seconds(root):
    prior = Path(root)/'recovery-v3'
    result = read_json(prior/'seed-404-exit.json')
    if result['exit_code'] != 1 or result['completed'] is not False:
        raise ValueError('Expected the documented aggregation failure')
    # The seed exit already includes all previous active attempts.
    return result['elapsed_seconds'] + (
        read_json(prior/'seed-404/supervisor_launch.json')['started_unix']
        - read_json(prior/'queue_launch.json')['actual_attempt_started_unix'])


def audit_completed(root, output):
    from .assets import model_inventory
    from .evaluate import collect_results, job_plan, partition_plan
    from phase2.utilities import read_evaluations
    from phase2.window_evidence import validate_shards
    root, output = Path(root).resolve(), Path(output).resolve()
    source, prior = root/'seed-404', root/'recovery-v3'
    if output != root/'recovery-v4/retained-readout.json' or output.exists():
        raise FileExistsError('Create a fresh retained-readout manifest once')
    plan = read_json(prior/'cohort-recovery.json')
    preparation = Path(plan['jobs'][0]['preparation'])
    summary = {'training': assert_training_complete(source)}
    if read_json(prior/'seed-404/stopped.json')['stage'] != 'u0000-u0005-valid_unseen-aggregate':
        raise ValueError('Only the documented aggregate stop is covered')
    window = source/'windows/u0000-u0005-valid_unseen'
    config = read_json(window/'protocol.json')
    original = Path(config['runtime']['authorization_path']).resolve()
    if original != prior/'permits/seed-404.json':
        raise ValueError('Unexpected original immutable window authorization')
    if file_hash(window/'protocol.json') != read_json(window/'manifest.json')['registered_protocol_sha256']:
        raise ValueError('Original registered protocol changed')
    forbidden = [source/'complete.json', source/'reports', window/'signals/calibration.json',
                 window/'reports', window/'evaluations/u0005']
    out = window/'window_signals/u0000-u0005'
    forbidden += [out/name for name in ('committed.json', 'prediction.json', 'parameter_delta.json',
                  'skill_context_features.parquet', 'token_signals.parquet', 'endpoint_replay_audit.json')]
    forbidden += list(source.glob('evaluations/u0005-*'))
    forbidden += [root/f'seed-{s}' for s in (505, 606)]
    if any(p.exists() for p in forbidden):
        raise ValueError('Downstream work already exists; reconcile instead of repeating it')
    export = read_json(source/'models/u0005/phase2_export.json')
    if (export['parent'] != str(source/'checkpoints/global_step_5') or export['dtype'] != 'float32'
            or export['native_weight_parity']['status'] != 'PASS'
            or export['native_weight_parity']['all_parameters_bitwise_equal'] is not True):
        raise ValueError('Missing verified retained U5 export')
    summary['completed_evaluation_sets'] = []
    checkpoint_identity = model_inventory(source/'models/u0000')
    for split, purpose, count in (('valid_seen', 'performance', 140),
            ('valid_unseen', 'performance', 134), ('valid_unseen', 'anchors', 134)):
        directory = source/'evaluations'/f'u0000-{split}-{purpose}'
        base = job_plan(preparation, source/'models/u0000', 0, split, purpose)
        base.update(checkpoint_identity=checkpoint_identity, prediction_lock=None)
        full = {**base, 'shard_count': 8}
        if read_json(directory/'plan.json') != full:
            raise ValueError('Completed evaluator plan differs from frozen game inventory')
        for rank in range(8):
            part = partition_plan(base, rank, 8)
            sub = directory/'shards'/f'{rank:02d}'
            done = read_json(sub/'completion.json')
            if read_json(sub/'plan.json') != part or done['plan_sha256'] != digest(part):
                raise ValueError('Completed evaluator shard identity changed')
        rows, missing = collect_results(full, directory)
        done = read_json(directory/'completion.json')
        if missing or len(rows) != count or done['episodes'] != count or done['plan_sha256'] != digest(full):
            raise ValueError('Completed evaluator traversal is not exact')
        summary['completed_evaluation_sets'].append({'split': split, 'purpose': purpose, 'episodes': count})
    utility = read_evaluations(window)
    if len(utility) != 540 or set(utility['update']) != {0}:
        raise ValueError('Expected the complete 540 U0 continuations and no target labels')
    summary['old_utility_continuations'] = len(utility)
    summary['readout'] = validate_shards(window, 0, 5, 8)
    paths = []
    for directory in ('evaluations', 'support', 'windows', 'models/u0005'):
        # Path.rglob does not traverse directory symlinks: native checkpoints,
        # all OLD rows and batches stay at their original, already-audited paths.
        paths += [p for p in (source/directory).rglob('*') if p.is_file() and not p.is_symlink()
                  and p.name != '.writer.lock']
    paths += [original, prior/'cohort-recovery.json', prior/'retained-training.json',
              prior/'seed-404-exit.json', prior/'queue_launch.json', prior/'queue_stopped.json',
              prior/'seed-404/supervisor_launch.json', prior/'seed-404/stopped.json',
              prior/'seed-404/logs/u0000-u0005-valid_unseen-aggregate.log']
    if any((source/name).exists() for name in ('router.sqlite3-wal', 'router.sqlite3-journal')):
        raise ValueError('Router ledger must be quiescent for its failure snapshot')
    snapshot = output.parent/'failure-snapshot/router.sqlite3'
    write_new_bytes(snapshot, (source/'router.sqlite3').read_bytes())
    paths.append(snapshot)
    def item(path):
        return {'path': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size}
    with ThreadPoolExecutor(max_workers=4) as pool:
        retained = list(pool.map(item, sorted(set(paths))))
    value = {'schema_version': 'skillnet.phase12.retained_readout.v1', 'source_root': str(source),
        'preparation_sha256': file_hash(preparation), 'summary': summary,
        'original_authorization': str(original), 'original_authorization_sha256': file_hash(original),
        'windows': [str(window)], 'retained_files': retained,
        'target_gold_read': False, 'scientific_outcomes_consulted_for_design': False}
    write_new_json(output, value)
    print({'status': 'COMPLETED_WORK_PRESERVED', 'files': len(retained), 'summary': summary}, flush=True)


@lru_cache(maxsize=2)
def verified_manifest(path, expected):
    if file_hash(path) != expected:
        raise ValueError('Retained-readout manifest changed')
    value = read_json(path)
    def check(row):
        if file_hash(row['path']) != row['sha256']:
            raise ValueError('Retained completed work changed: '+row['path'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check, value['retained_files']))
    return value


def verify_binding(binding, source, preparation):
    source = Path(source).resolve()
    value = verified_manifest(binding['manifest'], binding['sha256'])
    if (value['schema_version'] != 'skillnet.phase12.retained_readout.v1'
            or value['source_root'] != str(source) or source.name != 'seed-404'
            or value['preparation_sha256'] != file_hash(preparation)
            or Path(binding['attempt_dir']).resolve() != source.parent/'recovery-v4/seed-404'
            or binding.get('skip_training') is not True or binding.get('skip_measurement') is not True
            or binding.get('completed_through_update') != 5
            or any(binding.get(k) != value[k] for k in ('windows', 'source_root',
                'original_authorization', 'original_authorization_sha256'))):
        raise PermissionError('Readout continuation must bind completed U5, U0 episodes and exact shards')
    assert_training_complete(source)
    return value


def validate(plan):
    v, root = plan['recovery'], Path(plan['root']).resolve()
    if (v.get('schema_version') != 'skillnet.phase12.explicit_recovery.v4'
            or v.get('stage') != 'readout_aggregate' or v.get('user_authorized') is not True
            or v.get('exclude_failed_downtime') is not True
            or Path(v['audit_root']).resolve() != root/'recovery-v4'
            or Path(v['prior_plan']).resolve() != root/'recovery-v3/cohort-recovery.json'
            or file_hash(v['prior_plan']) != v['prior_plan_sha256']
            or v['previous_active_seconds'] != consumed_seconds(root)):
        raise PermissionError('Explicit aggregate continuation required')
    prior = read_json(v['prior_plan'])
    for key in ('root', 'seeds', 'storage', 'permit_template', 'finish_reserve_seconds', 'jobs'):
        if plan[key] != prior[key]:
            raise ValueError('Readout recovery must preserve all science and disk settings')
    override = plan.get('runtime_limit_override', {})
    if (plan['hard_limit_seconds'] is not None
            or override.get('unlimited_wallclock_user_authorized') is not True
            or override.get('storage_limits_unchanged') is not True):
        raise PermissionError('This continuation requires the explicit time-limit removal')
    if any((root/f'seed-{s}').exists() for s in (505, 606)) or (root/'recovery-v4/queue_finished.json').exists():
        raise ValueError('Only the documented seed404 readout stop may be resumed')
    verify_binding(v['readout_binding'], root/'seed-404', plan['jobs'][0]['preparation'])
    return plan


def prepare(prior_plan, manifest, output):
    prior_plan, manifest, output = (Path(p).resolve() for p in (prior_plan, manifest, output))
    prior = read_json(prior_plan)
    root = Path(prior['root']); audit = root/'recovery-v4'
    if prior_plan != root/'recovery-v3/cohort-recovery.json' or output != audit/'cohort-recovery.json' or output.exists():
        raise FileExistsError('Create only a new authorized recovery-v4 plan')
    tests = ElementTree.parse(audit/'offline-full-01.xml').getroot()
    if not list(tests.iter('testcase')) or tests.find('.//failure') is not None or tests.find('.//error') is not None:
        raise ValueError('CPU regression must pass first')
    for name in ('cpu-initial-parameter-audit.json', 'offline-continuation-audit.json'):
        if read_json(audit/name).get('status') != 'PASS':
            raise ValueError('Actual-artifact CPU continuation validation must pass')
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
    value = read_json(manifest)
    binding = {k: value[k] for k in ('windows', 'source_root', 'original_authorization', 'original_authorization_sha256')}
    binding.update(manifest=str(manifest), sha256=file_hash(manifest),
        attempt_dir=str(audit/'seed-404'), completed_through_update=5, skip_training=True, skip_measurement=True)
    plan.update(hard_limit_seconds=None, target_seconds=None,
        runtime_limit_override={'unlimited_wallclock_user_authorized': True,
            'user_authority': '不再设置预算限制，先跑（针对累计运行时间上限的明确答复）',
            'previous_hard_limit_seconds': prior['hard_limit_seconds'],
            'previous_target_seconds': prior['target_seconds'], 'storage_limits_unchanged': True,
            'scientific_scope_unchanged': True, 'retain_active_runtime_accounting': True})
    plan['recovery'] = {'schema_version': 'skillnet.phase12.explicit_recovery.v4',
        'stage': 'readout_aggregate', 'user_authorized': True,
        'user_authority': '修复，评估完后开始跑后续seed；不再设置预算限制，先跑',
        'exclude_failed_downtime': True, 'previous_active_seconds': consumed_seconds(root),
        'audit_root': str(audit), 'prior_plan': str(prior_plan), 'prior_plan_sha256': file_hash(prior_plan),
        'changed_sources': changed, 'readout_binding': binding,
        'cpu_regression_sha256': file_hash(audit/'offline-full-01.xml'),
        'offline_continuation_audit_sha256': file_hash(audit/'offline-continuation-audit.json')}
    validate(plan); write_new_json(output, plan)
    print({'status': 'PREPARED_NOT_LAUNCHED', 'time_limit_seconds': None,
        'previous_active_seconds': consumed_seconds(root), 'plan': str(output)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--root', required=True, type=Path)
    audit.add_argument('--output', required=True, type=Path)
    prep = sub.add_parser('prepare')
    for name in ('prior-plan', 'manifest', 'output'):
        prep.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    if args.operation == 'audit':
        audit_completed(args.root, args.output)
    else:
        prepare(args.prior_plan, args.manifest, args.output)
