"""Eight-repeat seed505 paired utility and all-285 readout prediction pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .common import REPO, exclusive_writer, file_hash, read_json, write_new_bytes, write_new_json
from .first_calls_storage import disk_gate
from .realized_reward_analysis import read_csv, direction_table
from .reward_variant_analysis import csv, point_tables
from .utility_precision_analysis import (analyze_frame, repeat_diagnostics, subset_frame,
    uncertainty)
from .utility_magnitude_analysis import (magnitude_bootstrap, magnitude_points,
    paired_contrasts, precision_curve)
from .seed505_readout import (COHORT, SOURCE as NUMERICAL, TRAJECTORY as SOURCE,
    OUTPUT as SCORES)

OUTPUT = COHORT/'utility-precision-s505-v1'
MODULE = 'skillnet_cohort.seed505_precision'
OLD_GOLD = (63011, 63021)
NEW_GOLD = (505, 505100, 505200, 505300, 505400, 505500)
GOLD = OLD_GOLD+NEW_GOLD
VERSION = 'skillnet.utility_precision_seed505.v1'


def scope(output):
    output = Path(output).resolve()
    if output.parent != COHORT or not output.name.startswith('utility-precision-s505-v'):
        raise PermissionError('Only a new scoped seed505 precision directory is allowed')
    return output


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_hash(path)}


def verify(records):
    for row in records:
        if file_hash(row['path']) != row['sha256']:
            raise ValueError('Immutable input changed: '+row['path'])


def expanded_config(old, output):
    import copy
    if tuple(old['evaluation']['gold_seeds']) != OLD_GOLD:
        raise ValueError('Expected original two gold repeats')
    config = copy.deepcopy(old)
    config['root'] = str(output)
    config['evaluation']['gold_seeds'] = list(GOLD)
    config['runtime']['cache_path'] = str(output/'router.sqlite3')
    config['runtime']['authorization_path'] = str(output/'permit.json')
    config['runtime']['max_local_calls'] = (
        old['evaluation']['anchor_count']*2*3*len(NEW_GOLD)*old['evaluation']['max_steps'])
    return config


def validate_config(old, config, output):
    from phase2.protocol import seed_streams
    if config != expanded_config(old, output):
        raise ValueError('Only gold repeat count and owned runtime paths may change')
    seed_streams(config)
    if (config['evaluation']['max_steps'] != 50 or config['evaluation']['shards'] != 8
            or config['runtime']['router_backend'] != 'skillrl_embedding_state_batch'
            or config['runtime']['max_api_calls'] != 0):
        raise ValueError('Keep frozen model, router, horizon and eight shards')
    for seed in NEW_GOLD:
        if any(seed != other and abs(seed-other) < 50 for other in GOLD):
            raise ValueError('New decoding seed ranges overlap')


def jobs_with_ids(config, update):
    from .utility_precision import jobs_with_ids as original_jobs
    return original_jobs(config, update)


def import_endpoint(output, old, config, update):
    from .first_calls_recovery import audit_endpoint
    from phase1.archive import stable_hash
    source_audit = audit_endpoint(SOURCE, update)
    if source_audit['missing'] or source_audit['complete_shards'] != list(range(8)):
        raise ValueError('Original seed505 endpoint not complete')
    expected = {r[1] for r in jobs_with_ids(old, update)}
    rows = {}
    for path in sorted((SOURCE/'evaluations'/f'u{update:04d}').glob('shard-*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['trajectory_id'] in rows:
                raise ValueError('Duplicate retained evaluation')
            rows[row['trajectory_id']] = row
    if set(rows) != expected:
        raise ValueError('Retained evaluation identities changed')
    by_rank, reused = [[] for _ in range(8)], []
    for rank, tid, identity, _ in jobs_with_ids(config, update):
        if tid not in rows:
            if identity['purpose'] != 'gold' or identity['continuation_seed'] not in NEW_GOLD:
                raise ValueError('Only newly registered gold may be missing')
            continue
        row = rows[tid]
        original = Path(row['trajectory_path'])
        target = output/'evaluations'/f'u{update:04d}'/'trajectories'/identity['skill_id']/(tid+'.json')
        provenance = {'source_path': str(original), 'source_sha256': file_hash(original),
                      'source_protocol_sha256': file_hash(SOURCE/'protocol.json'),
                      'episode_reexecuted': False}
        value = read_json(original)
        value.update(trajectory_path=str(target), precision_reuse=provenance)
        write_new_json(target, value)
        by_rank[rank].append({**row, 'trajectory_path': str(target), 'precision_reuse': provenance})
        reused.append({'trajectory_id': tid, **provenance})
    for rank, values in enumerate(by_rank):
        write_new_bytes(output/'evaluations'/f'u{update:04d}'/f'shard-{rank}.jsonl',
            ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in values).encode())
    write_new_json(output/'reuse'/f'u{update:04d}.json', {'reused': len(reused),
        'new_required': len(jobs_with_ids(config, update))-len(reused), 'records': reused})
    return source_audit


def capacity(output, plan, *, reserve_new=False):
    limits = plan['storage']
    extra = plan['projected_new_bytes'] if reserve_new else 0
    return disk_gate(output, limits['checkpoint_reserve_bytes']+extra,
        minimum_free_bytes=limits['minimum_free_bytes'], maximum_run_bytes=limits['maximum_run_bytes'])


def preflight(old):
    from phase2.protocol import anchor_sets
    from phase2.utilities import read_evaluations
    from .first_calls_report import clustered_units
    from .realized_reward_analysis import independent_metrics
    frame = read_evaluations(SOURCE)
    anchors = [a for group in anchor_sets(old, REPO) for a in group['anchors']]
    units, _, _ = clustered_units(frame, anchors,
        repetitions=old['evaluation']['bootstrap_repetitions'])
    historical = pd.read_parquet(SOURCE/'window_metrics/utility_units.parquet')
    keys = ['control', 'context_id', 'skill_id', 'phase']
    joined = units.merge(historical, on=keys, validate='one_to_one',
                         suffixes=('_now', '_old'), indicator=True)
    if (len(joined) != len(units) or len(joined) != len(historical)
            or not joined._merge.eq('both').all()):
        raise ValueError('Original seed505 utility population changed')
    for field in ('utility_old', 'utility_new', 'delta_utility'):
        if not np.allclose(joined[field+'_now'], joined[field+'_old'],
                           atol=1e-12, rtol=1e-12, equal_nan=True):
            raise ValueError('Old seed505 utility did not reproduce: '+field)
    scores = read_csv(SCORES/'skill_scores.csv')
    metadata = read_csv(SCORES/'registry.csv').to_dict('records')
    pools = read_json(NUMERICAL/'ranking-snapshots.json')['candidate_pools']
    diagnostics, _ = point_tables(scores, units, pools, metadata, [0., .05])
    if len(metadata) != 285 or len(diagnostics) != 5700:
        raise ValueError('Expected complete frozen 285-score grid')
    return {'historical_utility_units_reproduced': len(units),
            'new_readout_rows_on_old_labels': len(diagnostics),
            'independent_ranking': independent_metrics(scores, units, pools, metadata, diagnostics),
            'new_gold_labels_read': False, 'candidate_pools_frozen': True}


def prepare(output):
    from .assets import model_inventory
    from phase2.protocol import anchor_sets, validate_extended
    output = scope(output)
    if output.exists():
        raise FileExistsError('Never prepare over an existing attempt')
    if read_json(SOURCE/'complete.json')['status'] != 'complete':
        raise ValueError('Seed505 old paired continuations must be complete')
    if read_json(SCORES/'complete.json')['status'] != 'complete':
        raise ValueError('Seed505 all-285 readout not complete')
    if read_json(SCORES/'complete.json')['provenance_sha256'] != file_hash(SCORES/'provenance.json'):
        raise ValueError('Seed505 readout provenance changed')
    old = read_json(SOURCE/'protocol.json')
    config = expanded_config(old, output)
    validate_config(old, config, output)
    sets = anchor_sets(old, REPO)
    if len(sets) != 25 or sum(len(s['anchors']) for s in sets) != 404:
        raise ValueError('Do not change the natural first-call anchor population')
    inventory = read_json(SCORES/'plan.json')['model_inventory']
    for endpoint in ('u0000', 'u0005'):
        if model_inventory(SOURCE/'models'/endpoint) != inventory[endpoint]:
            raise ValueError('Policy endpoint differs from readout score source')
    total = len(jobs_with_ids(config, 0))*2
    new = len(NEW_GOLD)*404*3*2
    if total != 21816 or new != 14544:
        raise ValueError('Expected identical seed404 evaluation volume')
    storage = read_json(SOURCE/'resource_limits.json')
    plan = {'version': VERSION, 'seed': 505, 'approved': True, 'root': str(output),
        'authority': 'User requested a second independent RL seed with all 285 readouts and -m/|m|',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source': str(SOURCE),
        'numerical': str(NUMERICAL), 'scores': str(SCORES),
        'gold_seeds': list(GOLD), 'new_gold_seeds': list(NEW_GOLD),
        'evidence_seeds': old['evaluation']['old_evidence_seeds'],
        'new_training': False, 'new_readout_forward': False, 'new_api_calls': 0,
        'automatic_retry': False, 'hard_limit_seconds': None,
        'same_anchor_and_seed_across_endpoint_and_arm': True,
        'total_continuations': total, 'reused_continuations': total-new,
        'new_continuations': new, 'projected_new_bytes': new*4*2**20+8*2**30,
        'bootstrap_repetitions': 2000, 'bootstrap_rng_seed': 20260922,
        'event_thresholds': [0., .05], 'model_inventory': inventory,
        'storage': storage, 'score_columns': 285,
        'scientific_status': 'independent_RL_seed_replication_but_505_old_labels_previously_seen',
        'sources': [record(REPO/p) for p in ('skillnet_cohort/seed505_precision.py',
            'skillnet_cohort/utility_precision_run.py',
            'skillnet_cohort/utility_precision_analysis.py',
            'skillnet_cohort/utility_magnitude_analysis.py',
            '2026-09-24-seed505-eight-repeat-replication-protocol.md')],
        'inputs': [record(p) for p in (SOURCE/'protocol.json', SOURCE/'complete.json',
            SOURCE/'support/coverage.csv', SOURCE/'window_metrics/utility_units.parquet',
            SOURCE/'reports/performance.csv', NUMERICAL/'ranking-snapshots.json',
            NUMERICAL/'skill_context_features.parquet', SCORES/'complete.json',
            SCORES/'provenance.json', SCORES/'skill_scores.csv', SCORES/'registry.csv')]}
    plan['capacity_at_preparation'] = capacity(output, plan, reserve_new=True)
    write_new_json(output/'resource_limits.json', storage)
    write_new_json(output/'protocol.json', config)
    write_new_json(output/'permit.json', {'schema_version': VERSION, 'approved': True,
        'operations': ['evaluation', 'analysis'], 'run_root': str(output), 'seed': 505,
        'preparation_sha256': file_hash(config['runtime']['preparation']),
        'router_max_api_calls': 0,
        'router_max_local_calls': config['runtime']['max_local_calls'],
        'gpu_ids': list(range(8)), 'automatic_retry': False})
    (output/'models').symlink_to(SOURCE/'models', target_is_directory=True)
    validate_extended(config, REPO)
    retained = []
    for update in (0, 5):
        audit = import_endpoint(output, old, config, update)
        retained += audit['files']
        retained += [record(SOURCE/'evaluations'/f'u{update:04d}'/f'shard-{rank}{suffix}')
                     for rank in range(8) for suffix in ('.jsonl', '-complete.json')]
    write_new_json(output/'retained-inputs.json', retained)
    write_new_json(output/'preflight.json', preflight(old))
    for name in ('skill_scores.csv', 'registry.csv'):
        write_new_bytes(output/name, (SCORES/name).read_bytes())
    write_new_json(output/'predictor-lock.json', {'scores_sha256': file_hash(output/'skill_scores.csv'),
        'registry_sha256': file_hash(output/'registry.csv'), 'new_gold_not_yet_generated': True,
        'seed505_old_labels_previously_seen': True, 'readout_selection_frozen_from_seed404': True})
    plan['inputs'] += [record(output/n) for n in ('protocol.json', 'permit.json',
        'resource_limits.json', 'retained-inputs.json', 'preflight.json',
        'skill_scores.csv', 'registry.csv', 'predictor-lock.json')]
    write_new_json(output/'plan.json', plan)
    print(json.dumps({'status': 'PREPARED_NOT_STARTED', 'seed': 505,
        'new_continuations': new, 'total_continuations': total,
        'score_columns': 285, 'output': str(output)}), flush=True)


def binding(output, *, retained=False, models=False):
    from .assets import model_inventory
    output = scope(output); plan = read_json(output/'plan.json')
    if (plan['version'] != VERSION or plan['seed'] != 505 or plan['approved'] is not True
            or plan['root'] != str(output) or plan['gold_seeds'] != list(GOLD)
            or plan['new_gold_seeds'] != list(NEW_GOLD) or plan['score_columns'] != 285
            or plan['total_continuations'] != 21816 or plan['new_continuations'] != 14544
            or plan['new_training'] or plan['new_readout_forward'] or plan['new_api_calls']
            or plan['automatic_retry']):
        raise PermissionError('Changed seed505 utility precision scope')
    verify(plan['inputs']+plan['sources'])
    validate_config(read_json(SOURCE/'protocol.json'), read_json(output/'protocol.json'), output)
    if retained:
        verify(read_json(output/'retained-inputs.json'))
    if models:
        for endpoint, expected in plan['model_inventory'].items():
            if model_inventory(output/'models'/endpoint) != expected:
                raise ValueError('Policy weights changed')
    return plan


def worker(output, update, shard):
    from . import utility_precision_run as upstream
    upstream.binding = binding
    upstream.NEW_GOLD = NEW_GOLD
    upstream.jobs_with_ids = jobs_with_ids
    upstream.worker(scope(output), update, shard)


def analyze(output):
    from phase2.protocol import anchor_sets
    from phase2.utilities import read_evaluations
    from .first_calls_recovery import audit_endpoint
    from .first_calls_report import clustered_units
    from .numerical_readout_report import comparison_tables
    from .realized_reward_analysis import independent_metrics
    from .reports import passport
    output = scope(output); plan = binding(output, retained=True, models=True)
    if (output/'analysis-intent.json').exists():
        raise FileExistsError('No implicit analysis retry')
    write_new_json(output/'analysis-intent.json', {'started_unix': time.time(),
        'plan_sha256': file_hash(output/'plan.json')})
    for update in (0, 5):
        audit = audit_endpoint(output, update)
        if audit['missing'] or audit['complete_shards'] != list(range(8)):
            raise ValueError('Incomplete paired endpoint')
        write_new_json(output/'audits'/f'u{update:04d}.json', audit)
    config = read_json(output/'protocol.json')
    frame = read_evaluations(output)
    if len(frame) != plan['total_continuations']:
        raise ValueError('Incomplete paired continuation panel')
    anchors = [a for group in anchor_sets(config, REPO) for a in group['anchors']]
    scores = read_csv(output/'skill_scores.csv')
    metadata = read_csv(output/'registry.csv').to_dict('records')
    pools = read_json(NUMERICAL/'ranking-snapshots.json')['candidate_pools']
    precision, rankings = [], []
    for panel, seeds in (('gold2', GOLD[:2]), ('gold4', GOLD[:4]),
                         ('gold8', GOLD), ('added6_only', NEW_GOLD)):
        u, g, m, d, b = analyze_frame(subset_frame(frame, seeds), anchors,
            scores, pools, metadata, config['evaluation']['bootstrap_repetitions'])
        csv(output/'precision'/f'{panel}-utility.csv', u)
        csv(output/'precision'/f'{panel}-ranking.csv', d)
        precision.append(u.assign(precision_panel=panel))
        rankings.append(d.assign(precision_panel=panel))
        if panel == 'gold8':
            units, games, margins, direction, budgets = u, g, m, d, b
    old = pd.read_parquet(SOURCE/'window_metrics/utility_units.parquet')
    gold2 = precision[0]
    keys = ['control', 'context_id', 'phase', 'skill_id']
    joined = gold2.merge(old, on=keys, validate='one_to_one', suffixes=('_now','_old'))
    if len(joined) != len(old) or not np.allclose(joined.delta_utility_now,
        joined.delta_utility_old, atol=1e-12, rtol=1e-12, equal_nan=True):
        raise ValueError('Original two-repeat labels did not reproduce')
    for name, table in (('utility_units', units), ('utility_games', games),
                        ('anchor_margins', margins)):
        write_new_bytes(output/'window_metrics'/f'{name}.parquet', table.to_parquet(index=False))
    csv(output/'reports/precision-curve-utility.csv', pd.concat(precision, ignore_index=True))
    csv(output/'reports/precision-curve-ranking.csv', pd.concat(rankings, ignore_index=True))
    csv(output/'reports/direction-recomputed.csv', direction)
    independent = independent_metrics(scores, units, pools, metadata, direction)
    write_new_json(output/'reports/independent-ranking-verification.json', independent)
    csv(output/'reports/ranking_budgets.csv', budgets)
    csv(output/'reports/direction-confusions.csv',
        direction_table(scores, units, pools, metadata, plan['event_thresholds']))
    repeats, stability = repeat_diagnostics(margins)
    csv(output/'reports/per-continuation-seed-utility.csv', repeats)
    csv(output/'reports/repeat-sign-stability.csv', stability)
    summary, paired, reliability, draws, receipts = uncertainty(
        scores, units, margins, pools, metadata, plan)
    csv(output/'reports/bootstrap-summary-direction.csv', summary)
    csv(output/'reports/paired-readout-gains-direction.csv', paired)
    csv(output/'reports/label-bootstrap-stability.csv', reliability)
    write_new_json(output/'bootstrap-receipts.json', receipts)
    for control, table in draws.items():
        write_new_bytes(output/f'bootstrap-label-draws-{control}.parquet',
                        table.to_parquet(index=False))
    magnitude = magnitude_points(scores, units, pools, metadata)
    magnitude_ci = magnitude_bootstrap(scores, units, pools, metadata, output)
    magnitude_pairs = paired_contrasts(scores, units, pools, metadata, output)
    magnitude_curve = precision_curve(scores, pools, metadata, output)
    csv(output/'reports/magnitude-points.csv', magnitude)
    csv(output/'reports/magnitude-bootstrap.csv', magnitude_ci)
    csv(output/'reports/paired-magnitude-contrasts.csv', magnitude_pairs)
    csv(output/'reports/precision-curve-magnitude.csv', magnitude_curve)
    main_direction = direction[(direction.control == 'placebo') & (direction.phase == 'all')
        & (direction.threshold == 0)].copy()
    main_magnitude = magnitude[(magnitude.control == 'placebo') & (magnitude.phase == 'all')]
    raw_magnitude = main_magnitude[main_magnitude['transform'] == 'raw'].copy()
    abs_magnitude = main_magnitude[main_magnitude['transform'] == 'absolute_readout'].copy()
    raw_magnitude = raw_magnitude.rename(columns={c: 'magnitude_raw_'+c for c in (
        'spearman', 'kendall', 'average_precision', 'auroc_large_change_vs_rest',
        'top_quartile_mean_absolute_delta', 'top_quartile_lift')})
    abs_magnitude = abs_magnitude.rename(columns={c: 'magnitude_abs_readout_'+c for c in (
        'spearman', 'kendall', 'average_precision', 'auroc_large_change_vs_rest',
        'top_quartile_mean_absolute_delta', 'top_quartile_lift')})
    keep = ['score'] + [c for c in raw_magnitude if c.startswith('magnitude_raw_')]
    primary_all = main_direction.merge(raw_magnitude[keep], on='score', validate='one_to_one')
    keep = ['score'] + [c for c in abs_magnitude if c.startswith('magnitude_abs_readout_')]
    primary_all = primary_all.merge(abs_magnitude[keep], on='score', validate='one_to_one')
    if len(primary_all) != 285 or primary_all.score.nunique() != 285:
        raise ValueError('Primary report must include every frozen readout')
    csv(output/'reports/all-285-primary-direction-and-magnitude.csv', primary_all)
    legacy = pd.read_parquet(SOURCE/'window_signals/u0000-u0005/skill_context_features.parquet')
    stable = pd.read_parquet(NUMERICAL/'skill_context_features.parquet')
    nd, nb, ns, _, new_pools = comparison_tables(legacy, stable, units, margins, config)
    if new_pools != pools:
        raise ValueError('Registered common candidate pools changed')
    for name, table in (('numerical-variant-rankings', nd),
                        ('numerical-variant-budgets', nb), ('numerical-scores-and-gold', ns)):
        csv(output/'reports'/f'{name}.csv', table)
    coverage = read_csv(SOURCE/'support/coverage.csv')
    utility = units[units.control == 'placebo'][['skill_id', 'context_id', 'phase',
        'anchor_count', 'utility_old', 'utility_new', 'delta_utility', 'ci_low', 'ci_high']]
    coverage = coverage.merge(utility, on=['skill_id','context_id','phase'],
        how='left', validate='one_to_one')
    csv(output/'reports/coverage_and_effects.csv', coverage)
    write_new_bytes(output/'reports/performance.csv',
        (SOURCE/'reports/performance.csv').read_bytes())
    primary = direction[(direction.control == 'placebo')&(direction.phase == 'all')
        &(direction.threshold == 0)&direction.score.isin((
            'D_original::token::reward', 'D_signed::token::reward',
            'D_factor::token::reward', 'D_real::token::reward',
            'M_delta_centered::token::magnitude'))]
    primary_mag = magnitude[(magnitude.control == 'placebo')&(magnitude.phase == 'all')
        &(magnitude['transform'] == 'raw')&magnitude.score.isin(primary.score)]
    p1 = (passport('utility_precision_seed505_v1')+
        '# Phase1：seed505 U0→U5 独立RL窗口，8组gold配对效用\n\n'
        '固定37技能库、逐状态top1路由、404首调用锚点；复用原2组，新增6组。'
        '同一anchor、arm和endpoint共用续跑seed；不重训、不重采样旧轨迹。\n\n'
        +coverage[(coverage.phase == 'all') & coverage.utility_evaluable][[
            'skill_id','source_games','anchor_count','delta_utility','ci_low','ci_high']]
          .to_markdown(index=False)+'\n\n'
        '全部自然技能、未支持原因、原始成功率与精度曲线分别见CSV。\n')
    p2 = (passport('utility_precision_seed505_v1')+
        '# Phase2：seed505 全部285冻结读出与效用变化预测\n\n'
        '主口径PLACEBO/all/0pp；方向目标`−m=−ΔM`，幅度目标`|m|=|ΔM|`。'
        '没有依据505新增gold调整公式、阈值、候选池或聚合方向。\n\n'
        '## 方向主表\n\n'+primary[['score','candidates','declines','increases',
            'average_precision','auroc_decline_vs_increase','spearman']]
          .to_markdown(index=False,floatfmt='.4f')+'\n\n'
        '## 幅度主表\n\n'+primary_mag[['score','candidates','large_change_gt_5pp',
            'spearman','kendall','average_precision','auroc_large_change_vs_rest',
            'top_quartile_lift']].to_markdown(index=False,floatfmt='.4f')+'\n\n'
        '全部285指标×PLACEBO/NULL×5阶段×0/5pp方向和raw/abs(score)幅度见CSV；'
        '其中`all-285-primary-direction-and-magnitude.csv`逐行合并主口径的两类目标；'
        '配对game×continuation区间及所有D的匹配无reward对照一并归档。'
        '8组重复是标签精度扩展，不是8个独立RL seed。505原两组标签在部分公式开发时已可见，'
        '这是独立RL种子的复核，不是完全盲法的前瞻确认。'
        '统计误用核查11/11：Simpson、生态谬误、Berkson、collider、基率、均值回归、'
        '幸存者、多重比较、分析路径、相关非因果、反向因果/泄漏。\n')
    write_new_bytes(output/'reports/phase1-results.md', p1.encode())
    write_new_bytes(output/'reports/phase2-results.md', p2.encode())
    write_new_json(output/'analysis-complete.json', {'status': 'complete',
        'continuations': len(frame), 'score_columns': len(metadata),
        'direction_rows': len(direction), 'magnitude_rows': len(magnitude),
        'bootstrap_direction_rows': len(summary), 'bootstrap_magnitude_rows': len(magnitude_ci)})


def child_environment(gpu=''):
    from .utility_precision_run import environment
    return environment(gpu)


def commands(output, jobs, stage):
    plan = binding(output); capacity(output, plan)
    children = []
    try:
        for argv, label, gpu in jobs:
            path = output/'logs'/(label+'.log'); path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('x')
            proc = subprocess.Popen([sys.executable, '-u', '-B', '-m', MODULE, *argv,
                '--output', str(output)], cwd=REPO, env=child_environment(gpu),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
            children.append((proc, stream, label))
        write_new_json(output/'stages'/f'{stage}.json', {'started_unix': time.time(),
            'children': [{'pid': p.pid, 'label': label} for p,_,label in children]})
        tick = 0
        while any(p.poll() is None for p,_,_ in children):
            if any(p.poll() not in (None,0) for p,_,_ in children):
                raise RuntimeError('Child failed, preserve evidence and do not auto-retry')
            if tick % 2 == 0:
                capacity(output, plan)
            write_new_json(output/'heartbeats'/f'{stage}-{tick:06d}.json',
                {'utc': datetime.now(timezone.utc).isoformat(),
                 'children': [{'pid': p.pid,'label': label,'exit_code': p.poll()}
                              for p,_,label in children]})
            tick += 1; time.sleep(30)
        if any(p.returncode != 0 for p,_,_ in children):
            raise RuntimeError('Stage failed; see preserved child logs')
    finally:
        for p,_,_ in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p, stream, _ in children:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            stream.close()


def run(output):
    from .first_calls_defer import gpu_users
    from .first_calls_recovery import audit_endpoint
    output = scope(output); plan = binding(output, retained=True, models=True)
    if gpu_users() or (output/'run-intent.json').exists():
        raise PermissionError('Idle GPUs and a fresh explicit attempt required')
    with exclusive_writer(output):
        capacity(output, plan, reserve_new=True)
        write_new_json(output/'run-intent.json', {'pid': os.getpid(),
            'started_unix': time.time(), 'plan_sha256': file_hash(output/'plan.json')})
        stage = 'admission'
        try:
            for update in (0,5):
                stage = f'utility-u{update:04d}'
                audit = audit_endpoint(output, update)
                if audit['completed'] != plan['reused_continuations']//2 or audit['complete_shards']:
                    raise ValueError('Unexpected retained endpoint boundary')
                commands(output, [(['worker','--update',str(update),'--shard',str(rank)],
                    f'u{update:04d}-shard{rank}',str(rank)) for rank in range(8)], stage)
                audit = audit_endpoint(output, update)
                if audit['missing'] or audit['complete_shards'] != list(range(8)):
                    raise ValueError('Incomplete endpoint after workers returned')
            stage = 'analysis'
            commands(output, [(['analyze'],'analysis','')], stage)
            binding(output, retained=True, models=True)
            paths = [p for folder in ('reports','precision','window_metrics','audits','reuse')
                     for p in sorted((output/folder).rglob('*')) if p.is_file()]
            paths += [output/n for n in ('plan.json','protocol.json','predictor-lock.json',
                'skill_scores.csv','registry.csv','preflight.json','analysis-complete.json',
                'bootstrap-receipts.json','bootstrap-label-draws-placebo.parquet',
                'bootstrap-label-draws-null.parquet')]
            write_new_json(output/'provenance.json', {'plan_sha256': file_hash(output/'plan.json'),
                'files': [{'path': str(p.relative_to(output)), 'sha256': file_hash(p)} for p in paths]})
            write_new_json(output/'complete.json', {'status': 'complete', 'seed': 505,
                'gold_repeats': 8, 'total_continuations': plan['total_continuations'],
                'new_continuations': plan['new_continuations'],
                'readout_columns': 285, 'old_files_preserved': True,
                'provenance_sha256': file_hash(output/'provenance.json')})
            print('COMPLETE seed505 eight-repeat all-readout Phase1/2', flush=True)
        except BaseException as error:
            write_new_json(output/'stopped.json', {'stage': stage, 'error': repr(error),
                'stopped_unix': time.time(), 'automatic_retry': False})
            raise


def launch(output):
    from .first_calls_defer import gpu_users, process_identity
    output = scope(output); binding(output)
    if gpu_users() or any((output/n).exists() for n in ('launch.json','run-intent.json','workflow.log')):
        raise PermissionError('Fresh launch only, GPUs must be idle')
    capacity(output, read_json(output/'plan.json'), reserve_new=True)
    with (output/'workflow.log').open('x') as stream:
        proc = subprocess.Popen([sys.executable,'-u','-B','-m',MODULE,'run',
            '--output',str(output)],cwd=REPO,env=child_environment(),
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            start_new_session=True)
    write_new_json(output/'launch.json', {'pid': proc.pid,'identity':process_identity(proc.pid),
        'started_unix':time.time(),'plan_sha256':file_hash(output/'plan.json')})
    print(json.dumps({'status':'LAUNCHED_NOT_COMPLETE','pid':proc.pid,
                      'output':str(output)}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','launch','run','worker','analyze'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--update',type=int)
    parser.add_argument('--shard',type=int)
    args = parser.parse_args()
    if args.mode == 'worker':
        if args.update is None or args.shard is None:
            parser.error('worker needs --update and --shard')
        worker(args.output,args.update,args.shard)
    else:
        {'prepare':prepare,'launch':launch,'run':run,'analyze':analyze}[args.mode](args.output)


if __name__ == '__main__':
    main()
