"""Append-only seed404 utility precision amendment; no training/readout changes."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from .common import REPO, file_hash, read_json, write_new_bytes, write_new_json

COHORT = REPO/'artifacts/phase12/skillnet37-independent-s404-505-606-v4'
SOURCE = COHORT/'all-first-calls-v1/seed-404'
NUMERICAL = COHORT/'numerical-readout-release-v1/seed-404'
SCORES = COHORT/'factorized-reward-s404-v1'
DEFAULT_OUTPUT = COHORT/'utility-precision-s404-v1'
VERSION = 'skillnet.utility_precision_seed404.v1'
OLD_GOLD = (63011, 63021)
# run_branch uses base_seed + absolute_step. New bases are >=100 apart so
# their 50-step seed ranges do not overlap. Historical repeats stay untouched.
NEW_GOLD = (404, 404100, 404200, 404300, 404400, 404500)
GOLD = OLD_GOLD + NEW_GOLD
DOC = REPO/'docs/experiments/phase12-independent-v4/UTILITY-PRECISION-SEED404-20260922-v1.md'
NEW_FILES = [Path(__file__), REPO/'skillnet_cohort/utility_precision_run.py',
             REPO/'skillnet_cohort/utility_precision_analysis.py',
             REPO/'tests/skillnet_cohort/test_utility_precision.py', DOC]


def record(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_hash(path)}


def verify(records):
    for item in records:
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Immutable input changed: '+item['path'])


def scope(output):
    output = Path(output).resolve()
    if output.parent != COHORT or not output.name.startswith('utility-precision-s404-v'):
        raise PermissionError('Only a new seed404 utility-precision directory is authorized')
    return output


def expanded_config(old, output):
    if tuple(old['evaluation']['gold_seeds']) != OLD_GOLD:
        raise ValueError('Unexpected original gold seeds')
    result = copy.deepcopy(old)
    result['root'] = str(output)
    result['evaluation']['gold_seeds'] = list(GOLD)
    runtime = result['runtime']
    runtime['cache_path'] = str(output/'router.sqlite3')
    runtime['authorization_path'] = str(output/'permit.json')
    runtime['max_local_calls'] = old['evaluation']['anchor_count']*2*3*len(NEW_GOLD)*old['evaluation']['max_steps']
    return result


def validate_expansion(old, config, output):
    if config != expanded_config(old, output):
        raise ValueError('Only gold repeat count and owned runtime paths/budget may change')
    from phase2.protocol import seed_streams
    seed_streams(config)
    ev = config['evaluation']
    if (ev['max_steps'] != 50 or ev['shards'] != 8
            or config['runtime']['router_backend'] != 'skillrl_embedding_state_batch'
            or config['runtime']['max_api_calls'] != 0):
        raise PermissionError('Keep the frozen horizon, eight shards and local router')
    for seed in NEW_GOLD:
        if any(seed != other and abs(seed-other) < ev['max_steps'] for other in GOLD):
            raise ValueError('New decoding seed ranges overlap')


def jobs_with_ids(config, update):
    from phase1.archive import stable_hash
    from phase2.protocol import evaluation_identity, evaluation_jobs
    if update not in (0, 5):
        raise ValueError('Only the existing U0/U5 endpoints')
    rows = []
    for position, job in enumerate(evaluation_jobs(config, REPO)):
        identity = evaluation_identity(config, update, job)
        rows.append((position % config['evaluation']['shards'], stable_hash(identity)[:24], identity, job))
    if len({r[1] for r in rows}) != len(rows):
        raise ValueError('Duplicate evaluation identity')
    return rows


def import_endpoint(source, output, old, config, update):
    """Copy validated retained episodes into the new partition; no resampling."""
    from .first_calls_recovery import audit_endpoint
    audit = audit_endpoint(source, update)
    if audit['missing'] or audit['complete_shards'] != list(range(8)):
        raise ValueError('Original endpoint must be complete')
    expected_old = {r[1] for r in jobs_with_ids(old, update)}
    rows = {}
    for path in sorted((source/'evaluations'/f'u{update:04d}').glob('shard-*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['trajectory_id'] in rows:
                raise ValueError('Duplicate retained row')
            rows[row['trajectory_id']] = row
    if set(rows) != expected_old:
        raise ValueError('Retained identities differ from the frozen protocol')
    by_rank = [[] for _ in range(8)]; reused = []
    for rank, tid, identity, job in jobs_with_ids(config, update):
        if tid not in rows:
            if identity['purpose'] != 'gold' or identity['continuation_seed'] not in NEW_GOLD:
                raise ValueError('Only newly registered gold repeats may be missing')
            continue
        row = rows[tid]; original = Path(row['trajectory_path'])
        target = output/'evaluations'/f'u{update:04d}'/'trajectories'/identity['skill_id']/(tid+'.json')
        provenance = {'source_path': str(original), 'source_sha256': file_hash(original),
                      'source_protocol_sha256': file_hash(source/'protocol.json'), 'episode_reexecuted': False}
        result = read_json(original)
        result.update(trajectory_path=str(target), precision_reuse=provenance)
        write_new_json(target, result)
        by_rank[rank].append({**row, 'trajectory_path': str(target), 'precision_reuse': provenance})
        reused.append({'trajectory_id': tid, **provenance})
    for rank, values in enumerate(by_rank):
        write_new_bytes(output/'evaluations'/f'u{update:04d}'/f'shard-{rank}.jsonl',
            ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in values).encode())
    write_new_json(output/'reuse'/f'u{update:04d}.json', {'reused': len(reused),
        'new_required': len(jobs_with_ids(config, update))-len(reused), 'records': reused})
    return audit


def check_tests(path):
    suites = list(ET.parse(path).getroot().iter('testsuite'))
    if (not suites or sum(int(s.get('tests', 0)) for s in suites) < 20
            or any(int(s.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped'))):
        raise ValueError('At least20 passing CPU tests without skips required')


def capacity(output, plan, *, reserve_new=False):
    from .first_calls_storage import disk_gate
    limits = plan['storage']
    extra = plan['projected_new_bytes'] if reserve_new else 0
    return disk_gate(output, limits['checkpoint_reserve_bytes']+extra,
        minimum_free_bytes=limits['minimum_free_bytes'], maximum_run_bytes=limits['maximum_run_bytes'])


def prepare(output, tests):
    from .assets import model_inventory
    from .reward_variant_analysis import check_origin
    from phase2.protocol import anchor_sets, validate_extended
    output = scope(output)
    if output.exists():
        raise FileExistsError('Never prepare over an existing attempt')
    check_tests(tests); check_origin()
    old = read_json(SOURCE/'protocol.json')
    prior = read_json(SCORES/'plan.json')
    sources = prior['sources']+[record(p) for p in NEW_FILES]
    verify(sources)
    for source in (SOURCE, SCORES):
        if read_json(source/'complete.json')['status'] != 'complete':
            raise ValueError('Retained source must be complete')
    score_complete = read_json(SCORES/'complete.json')
    if score_complete['provenance_sha256'] != file_hash(SCORES/'provenance.json'):
        raise ValueError('Changed score provenance')
    score_files = {r['path']: r['sha256'] for r in read_json(SCORES/'provenance.json')['files']}
    for name in ('skill_scores.csv', 'registry.csv', 'ranking_diagnostics.csv'):
        if score_files[name] != file_hash(SCORES/name):
            raise ValueError('Unsealed predictor input')
    legacy_plan = read_json(SOURCE.parent/'plan.json')
    models = legacy_plan['model_inventory']
    for name, inventory in models.items():
        if model_inventory(SOURCE/'models'/name) != inventory:
            raise ValueError('Retained policy changed: '+name)
    config = expanded_config(old, output); validate_expansion(old, config, output)
    sets = anchor_sets(old, REPO)
    if len(sets) != 25 or sum(len(s['anchors']) for s in sets) != 404:
        raise ValueError('Do not select a different skill/anchor population')
    total = len(jobs_with_ids(config, 0))*2
    new_count = len(NEW_GOLD)*404*3*2
    storage = {**legacy_plan['storage'], 'cohort_storage_root': str(COHORT)}
    plan = {'schema_version': VERSION, 'approved': True, 'root': str(output), 'seed': 404,
        'authority': '用户：根据讨论增加续跑组数、提高效用标签可信度，修改并重新评估seed404',
        'created_unix': time.time(), 'source': str(SOURCE), 'numerical': str(NUMERICAL), 'scores': str(SCORES),
        'gold_seeds': list(GOLD), 'new_gold_seeds': list(NEW_GOLD), 'evidence_seeds': old['evaluation']['old_evidence_seeds'],
        'prefix_sizes': [2, 4, 8], 'fixed_added_only_sensitivity': list(NEW_GOLD),
        'original_training_seed': 404, 'new_seed_derivation': '404; 404*1000 + 100*r for r=1..5',
        'arms_and_endpoints_share_seed': True, 'environment_seed_and_prefix_unchanged': True,
        'stop_after_fixed_repetitions_not_results': True, 'new_training': False,
        'new_readout_forward': False, 'other_seeds': [], 'external_api_calls': 0,
        'automatic_retry': False, 'hard_limit_seconds': None, 'storage': storage,
        'total_continuations': total, 'reused_continuations': total-new_count,
        'new_continuations': new_count, 'projected_new_bytes': new_count*4*2**20+8*2**30,
        'bootstrap_repetitions': 2000, 'bootstrap_rng_seed': 20260922, 'event_thresholds': [0., .05],
        'all_prior_285_score_columns_retained': True, 'original_reports_overwritten': False,
        'scientific_status': 'posthoc_precision_amendment_fixed_predictors_not_new_RL_replication',
        'model_inventory': models, 'sources': sources, 'tests': record(tests)}
    # No old files are touched. Runtime paths all belong to the new directory.
    write_new_json(output/'resource_limits.json', storage)
    plan['capacity_at_preparation'] = capacity(output, plan, reserve_new=True)
    write_new_json(output/'protocol.json', config)
    write_new_json(output/'permit.json', {'schema_version': VERSION, 'approved': True,
        'operations': ['evaluation', 'analysis'], 'run_root': str(output), 'seed': 404,
        'preparation_sha256': file_hash(config['runtime']['preparation']),
        'router_max_api_calls': 0, 'router_max_local_calls': config['runtime']['max_local_calls'],
        'gpu_ids': list(range(8)), 'hard_limit_seconds': None, 'automatic_retry': False})
    (output/'models').symlink_to(SOURCE/'models', target_is_directory=True)
    validate_extended(config, REPO)
    retained = []
    for update in (0, 5):
        audit = import_endpoint(SOURCE, output, old, config, update)
        retained += audit['files']
        retained += [record(SOURCE/'evaluations'/f'u{update:04d}'/f'shard-{r}{suffix}')
                     for r in range(8) for suffix in ('.jsonl', '-complete.json')]
    write_new_json(output/'retained-inputs.json', retained)
    from .utility_precision_analysis import preflight
    print('PREPARE: validating actual historical reporting path (CPU only)', flush=True)
    write_new_json(output/'analysis-preflight.json', preflight())
    paths = [SOURCE/'protocol.json', SOURCE/'complete.json', SOURCE/'support/coverage.csv',
        SOURCE/'window_metrics/utility_units.parquet', SOURCE/'reports/performance.csv',
        SOURCE/'window_signals/u0000-u0005/skill_context_features.parquet',
        NUMERICAL/'skill_context_features.parquet', NUMERICAL/'ranking-snapshots.json',
        NUMERICAL/'committed.json', SCORES/'skill_scores.csv', SCORES/'registry.csv',
        SCORES/'ranking_diagnostics.csv', SCORES/'complete.json', SCORES/'provenance.json']
    paths += [Path(s[k]) for s in config['evaluation']['anchor_sets'] for k in ('anchors_path', 'placebo_path')]
    paths += [output/n for n in ('protocol.json', 'permit.json', 'resource_limits.json', 'retained-inputs.json', 'analysis-preflight.json')]
    paths += [output/'reuse'/f'u{u:04d}.json' for u in (0, 5)]
    plan['inputs'] = [record(REPO/p if not p.is_absolute() else p) for p in paths]
    for name in ('skill_scores.csv', 'registry.csv'):
        write_new_bytes(output/name, (SCORES/name).read_bytes())
        plan['inputs'].append(record(output/name))
    write_new_json(output/'predictor-lock.json', {'scores_sha256': file_hash(output/'skill_scores.csv'),
        'registry_sha256': file_hash(output/'registry.csv'), 'source': str(SCORES),
        'historical_labels_seen': True, 'new_gold_not_yet_generated': True, 'refit_or_formula_search': False})
    plan['inputs'].append(record(output/'predictor-lock.json'))
    write_new_json(output/'plan.json', plan)
    for path in NEW_FILES:
        write_new_bytes(output/'source-snapshot'/path.relative_to(REPO), path.read_bytes())
    print(json.dumps({'status': 'PREPARED_NOT_STARTED', 'root': str(output),
        'gold_seeds': list(GOLD), 'new_continuations': new_count, 'total_continuations': total}), flush=True)
    return plan


def binding(output, *, retained=False, models=False):
    output = scope(output); plan = read_json(output/'plan.json')
    if (plan['schema_version'] != VERSION or plan['approved'] is not True
            or plan['root'] != str(output) or plan['seed'] != 404
            or plan['gold_seeds'] != list(GOLD) or plan['new_gold_seeds'] != list(NEW_GOLD)
            or plan['new_training'] or plan['new_readout_forward'] or plan['other_seeds']
            or plan['external_api_calls'] != 0 or plan['automatic_retry']
            or plan['total_continuations'] != 21816 or plan['new_continuations'] != 14544):
        raise PermissionError('Changed seed404 precision experiment scope')
    verify(plan['inputs']+plan['sources']+[plan['tests']])
    validate_expansion(read_json(SOURCE/'protocol.json'), read_json(output/'protocol.json'), output)
    permit = read_json(output/'permit.json')
    if (permit['operations'] != ['evaluation', 'analysis'] or not permit['approved']
            or permit['run_root'] != str(output) or permit['router_max_api_calls'] != 0
            or permit['router_max_local_calls'] != read_json(output/'protocol.json')['runtime']['max_local_calls']):
        raise PermissionError('Incorrect evaluation-only permit')
    if retained:
        verify(read_json(output/'retained-inputs.json'))
    if models:
        from .assets import model_inventory
        for name, inventory in plan['model_inventory'].items():
            if model_inventory(output/'models'/name) != inventory:
                raise ValueError('Policy weights changed')
    return plan
