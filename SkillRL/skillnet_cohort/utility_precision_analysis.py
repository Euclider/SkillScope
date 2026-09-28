"""Fixed-score re-evaluation on augmented paired labels, with precision curves."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .common import file_hash, read_json, write_new_bytes, write_new_json
from .utility_precision import GOLD, NEW_GOLD, NUMERICAL, SOURCE, SCORES, binding
from .first_calls_report import clustered_units
from .reward_variant_analysis import csv, interval, point_tables, pool_data, SIGN_METRICS
from .reward_variant_statistics import bootstrap_targets
from .factorized_reward_analysis import metric_arrays_extended, BALANCED, verify_direction, magnitude_diagnostic
from .realized_reward_analysis import read_csv, direction_table, independent_metrics


def subset_frame(frame, seeds):
    """Evidence stays separate and unchanged; only gold repeats define labels."""
    return frame[(frame.purpose == 'evidence') | frame.continuation_seed.isin(seeds)].copy()


def repeat_diagnostics(margins):
    keys = ['skill_id', 'anchor_id', 'game_id', 'context_id', 'phase', 'trigger_step', 'continuation_seed']
    gold = margins[margins.purpose == 'gold']
    paired = gold[gold['update'] == 0].merge(gold[gold['update'] == 5], on=keys,
        how='outer', validate='one_to_one', suffixes=('_old', '_new'), indicator=True)
    if not paired._merge.eq('both').all():
        raise ValueError('Missing endpoint pair')
    if paired.duplicated(['skill_id', 'game_id', 'continuation_seed']).any():
        raise ValueError('Expected one first-call source trajectory per skill/game')
    rows = []
    for control in ('placebo', 'null'):
        for phase in ('all', 'initial', 'early', 'middle', 'late'):
            q = paired if phase == 'all' else paired[paired.phase == phase]
            q = q.copy(); q['delta'] = q[f'M_{control}_new']-q[f'M_{control}_old']
            for (skill, context, seed), group in q.groupby(['skill_id', 'context_id', 'continuation_seed']):
                values = group.groupby('game_id').delta.mean()
                rows.append({'control': control, 'phase': phase, 'skill_id': skill,
                    'context_id': context, 'continuation_seed': seed, 'game_count': len(values),
                    'delta_utility': float(values.mean())})
    repeats = pd.DataFrame(rows); summary = []
    for key, group in repeats.groupby(['control', 'phase', 'skill_id', 'context_id']):
        x = group.delta_utility.to_numpy(); signs = np.where(np.abs(x) <= 1e-12, 0., np.sign(x))
        n = len(x); pairs = n*(n-1)//2
        opposite = sum(signs[i]*signs[j] < 0 for i in range(n) for j in range(i))
        summary.append({**dict(zip(['control', 'phase', 'skill_id', 'context_id'], key)),
            'repeat_count': n, 'repeat_negative': int((signs < 0).sum()), 'repeat_zero': int((signs == 0).sum()),
            'repeat_positive': int((signs > 0).sum()), 'repeat_min': float(x.min()), 'repeat_max': float(x.max()),
            'repeat_sd': float(x.std(ddof=1)) if n > 1 else np.nan,
            'nonzero_opposite_sign_pairs': int(opposite), 'all_repeat_pairs': pairs,
            'opposite_sign_fraction': opposite/pairs if pairs else np.nan,
            'repeat_seed_is_not_an_independent_RL_seed': True})
    return repeats, pd.DataFrame(summary)


def uncertainty(scores, units, margins, pools, metadata, plan):
    summaries, comparisons, reliability, draws_out, receipts = [], [], [], {}, {}
    names = [m['score'] for m in metadata]; by_name = {m['score']: m for m in metadata}
    for control in ('placebo', 'null'):
        pool, = [p for p in pools[control+'/stable_raw'] if p['phase'] == 'all']
        matrix, target, _ = pool_data(scores, units, pool, control, metadata)
        draws, point, receipt = bootstrap_targets(margins[margins.context_id == pool['context_id']],
            pool['shared_skill_ids'], control, plan['bootstrap_repetitions'], plan['bootstrap_rng_seed'])
        if not np.allclose(target, point, rtol=1e-13, atol=1e-13):
            raise ValueError('Bootstrap point and reported game-weighted labels disagree')
        receipts[control] = receipt
        draws_out[control] = pd.DataFrame(draws, columns=pool['shared_skill_ids'])
        good = np.isfinite(draws).all(axis=1)
        for i, skill in enumerate(pool['shared_skill_ids']):
            v = draws[:, i]; v = v[np.isfinite(v)]
            reliability.append({'control': control, 'phase': 'all', 'skill_id': skill,
                'context_id': pool['context_id'], 'valid_label_draws': len(v),
                'bootstrap_fraction_decline': float(np.mean(v > 1e-12)) if len(v) else np.nan,
                'bootstrap_fraction_increase': float(np.mean(v < -1e-12)) if len(v) else np.nan,
                'bootstrap_fraction_zero': float(np.mean(np.abs(v) <= 1e-12)) if len(v) else np.nan,
                'not_posterior_probability': True})
        for threshold in plan['event_thresholds']:
            arrays = metric_arrays_extended(matrix, draws[good], threshold)
            points = metric_arrays_extended(matrix, target[None], threshold)
            for i, method in enumerate(metadata):
                base = {'control': control, 'phase': 'all', 'threshold': threshold, 'score': method['score'],
                    'complete_pool_draws': int(good.sum()), 'missing_any_skill_draws': int((~good).sum())}
                for metric, values in arrays.items():
                    if metric in SIGN_METRICS+[BALANCED] and not method['signed']:
                        continue
                    summaries.append({**base, 'metric': metric, 'point': float(points[metric][0, i]),
                                      **interval(values[:, i])})
                if method['mode'] != 'reward':
                    continue
                agg = method['aggregation']
                candidates = ['D_original::token::reward', f'M_delta_centered::{agg}::magnitude',
                    f'{method["name"]}::{agg}::unsigned', f'{method["name"]}_A1::{agg}::reward_free']
                if method['name'] in ('D_factor', 'D_real'):
                    candidates += [f'D_orientation::{agg}::reward', f'D_policy_action_adv::{agg}::reward']
                for ref in dict.fromkeys(candidates):
                    if ref not in by_name or ref == method['score']:
                        continue
                    j = names.index(ref)
                    for metric in ('average_precision', 'auroc_decline_vs_rest', 'auroc_decline_vs_increase', 'spearman'):
                        diff = arrays[metric][:, i]-arrays[metric][:, j]
                        comparisons.append({**base, 'reference': ref, 'metric': metric,
                            'point_difference': float(points[metric][0, i]-points[metric][0, j]),
                            **interval(diff), 'not_selection_adjusted': True})
    return (pd.DataFrame(summaries), pd.DataFrame(comparisons), pd.DataFrame(reliability), draws_out, receipts)


def analyze_frame(frame, anchors, scores, pools, metadata, repetitions):
    units, games, margins = clustered_units(frame, anchors, repetitions=repetitions)
    units['ci_width'] = units.ci_high-units.ci_low
    units['ci_contains_zero'] = (units.ci_low <= 0) & (units.ci_high >= 0)
    units['degenerate_interval'] = units.ci_low.notna() & units.ci_low.eq(units.ci_high)
    # Existing direction uses the historical 5pp practical threshold. Expose
    # zero-threshold identifiability separately; never silently redefine it.
    units['direction_at_zero'] = np.select(
        [units.ci_low.isna() | units.ci_high.isna(), units.ci_high < 0, units.ci_low > 0],
        ['interval_undefined', 'decline_resolved', 'increase_resolved'], default='unresolved')
    diagnostics, budgets = point_tables(scores, units, pools, metadata, [0., .05])
    return units, games, margins, diagnostics, budgets


def verify_historical(units, diagnostics):
    old = pd.read_parquet(SOURCE/'window_metrics/utility_units.parquet')
    keys = ['control', 'skill_id', 'context_id', 'phase']
    joined = units.merge(old, on=keys, validate='one_to_one', suffixes=('_now', '_old'))
    if len(joined) != len(units) or len(units) != len(old):
        raise ValueError('Original utility pool changed')
    for column in ('utility_old', 'utility_new', 'delta_utility', 'ci_low', 'ci_high'):
        if not np.allclose(joined[column+'_now'], joined[column+'_old'], rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError('Original two-repeat labels did not reproduce: '+column)
    old_diag = read_csv(SCORES/'ranking_diagnostics.csv')
    dk = ['control', 'context_id', 'phase', 'threshold', 'score']
    compared = diagnostics.merge(old_diag, on=dk, validate='one_to_one', suffixes=('_now', '_old'))
    if len(compared) != len(diagnostics) or len(diagnostics) != len(old_diag):
        raise ValueError('Original score comparison scope changed')
    for column in ('average_precision', 'auroc_decline_vs_rest', 'auroc_decline_vs_increase', 'spearman'):
        if not np.allclose(compared[column+'_now'], compared[column+'_old'], rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError('Original ranking did not reproduce: '+column)
    return {'original_two_repeat_units': len(units), 'original_ranking_rows_reproduced': len(diagnostics)}


def preflight():
    """Run the actual old-label reporting path before expensive new sampling."""
    from phase2.protocol import anchor_sets
    from phase2.utilities import read_evaluations
    from .common import REPO
    from .numerical_readout_report import comparison_tables
    config = read_json(SOURCE/'protocol.json')
    frame = read_evaluations(SOURCE)
    anchors = [a for s in anchor_sets(config, REPO) for a in s['anchors']]
    scores = read_csv(SCORES/'skill_scores.csv'); metadata = read_csv(SCORES/'registry.csv').to_dict('records')
    pools = read_json(NUMERICAL/'ranking-snapshots.json')['candidate_pools']
    u, _, m, d, _ = analyze_frame(frame, anchors, scores, pools, metadata, config['evaluation']['bootstrap_repetitions'])
    check = verify_historical(u, d)
    legacy = pd.read_parquet(SOURCE/'window_signals/u0000-u0005/skill_context_features.parquet')
    stable = pd.read_parquet(NUMERICAL/'skill_context_features.parquet')
    if comparison_tables(legacy, stable, u, m, config)[-1] != pools:
        raise ValueError('Prelaunch candidate pools changed')
    check['independent_metrics'] = independent_metrics(scores, u, pools, metadata, d)
    return {'status': 'PASS', 'new_environment_rollouts': 0, 'existing_continuations': len(frame),
            'score_columns': len(metadata), 'candidate_pools_identical': True, **check}


def report(output):
    from phase2.protocol import anchor_sets
    from phase2.utilities import read_evaluations
    from .common import REPO
    from .first_calls_recovery import audit_endpoint
    from .numerical_readout_report import comparison_tables, ID_KEYS
    from .reports import passport
    plan = binding(output, retained=True)
    if (output/'analysis-intent.json').exists():
        raise FileExistsError('No implicit analysis retry')
    write_new_json(output/'analysis-intent.json', {'started_unix': time.time(), 'plan_sha256': file_hash(output/'plan.json')})
    for update in (0, 5):
        audit = audit_endpoint(output, update)
        if audit['missing'] or audit['complete_shards'] != list(range(8)):
            raise ValueError('All paired endpoints must finish before analysis')
    config = read_json(output/'protocol.json'); frame = read_evaluations(output)
    if len(frame) != plan['total_continuations']:
        raise ValueError('Incomplete paired continuation panel')
    anchors = [a for s in anchor_sets(config, REPO) for a in s['anchors']]
    scores = read_csv(output/'skill_scores.csv'); metadata = read_csv(output/'registry.csv').to_dict('records')
    pools = read_json(NUMERICAL/'ranking-snapshots.json')['candidate_pools']
    if len(metadata) != 285:
        raise ValueError('Preserve every existing readout column')
    precision, curve_diagnostics, checks = [], [], {}
    schedules = [('gold2', GOLD[:2]), ('gold4', GOLD[:4]), ('gold8', GOLD), ('added6_only', NEW_GOLD)]
    for label, seeds in schedules:
        result = analyze_frame(subset_frame(frame, seeds), anchors, scores, pools, metadata,
                               config['evaluation']['bootstrap_repetitions'])
        u, g, m, d, b = result
        csv(output/'precision'/(label+'-utility.csv'), u)
        csv(output/'precision'/(label+'-ranking.csv'), d)
        precision.append(u.assign(precision_panel=label)); curve_diagnostics.append(d.assign(precision_panel=label))
        if label == 'gold2':
            checks.update(verify_historical(u, d))
        if label == 'gold8':
            units, games, margins, diagnostics, budgets = result
    for name, table in (('utility_units', units), ('utility_games', games), ('anchor_margins', margins)):
        write_new_bytes(output/'window_metrics'/(name+'.parquet'), table.to_parquet(index=False))
    csv(output/'reports/precision-curve-utility.csv', pd.concat(precision, ignore_index=True))
    csv(output/'reports/precision-curve-ranking.csv', pd.concat(curve_diagnostics, ignore_index=True))
    csv(output/'reports/ranking_diagnostics.csv', diagnostics)
    csv(output/'reports/ranking_budgets.csv', budgets)
    directions = direction_table(scores, units, pools, metadata, plan['event_thresholds'])
    checks['independent_ranking'] = independent_metrics(scores, units, pools, metadata, diagnostics)
    checks['independent_direction'] = verify_direction(scores, units, pools, metadata, directions)
    csv(output/'reports/direction-confusions.csv', directions)
    csv(output/'reports/skill_scores_and_gold.csv', scores.merge(units,
        on=['control', 'context_id', 'skill_id', 'phase'], how='left', validate='many_to_one'))
    repeats, stability = repeat_diagnostics(margins)
    csv(output/'reports/per-continuation-seed-utility.csv', repeats)
    csv(output/'reports/repeat-sign-stability.csv', stability)
    summary, paired, reliability, draws, receipts = uncertainty(scores, units, margins, pools, metadata, plan)
    csv(output/'reports/bootstrap_summary.csv', summary)
    csv(output/'reports/paired_readout_gains.csv', paired)
    csv(output/'reports/label-bootstrap-stability.csv', reliability)
    write_new_json(output/'bootstrap-receipts.json', receipts)
    for control, table in draws.items():
        write_new_bytes(output/f'bootstrap-label-draws-{control}.parquet', table.to_parquet(index=False))
    legacy = pd.read_parquet(SOURCE/'window_signals/u0000-u0005/skill_context_features.parquet')
    stable = pd.read_parquet(NUMERICAL/'skill_context_features.parquet')
    nd, nb, ns, snapshots, new_pools = comparison_tables(legacy, stable, units, margins, config)
    if new_pools != pools:
        raise ValueError('Label precision must not alter the registered candidate pools')
    for name, table in (('numerical-variant-rankings', nd), ('numerical-variant-budgets', nb), ('numerical-scores-and-gold', ns)):
        csv(output/'reports'/(name+'.csv'), table)
    coverage = read_csv(SOURCE/'support/coverage.csv').merge(
        stable[stable.control == 'placebo'][ID_KEYS+['supported', 'P_int', 'P_int_centered', 'C_upd',
            'C_upd_centered', 'D_contribution', 'D_centered_contribution', 'direction_coverage',
            'gate_coverage', 'gate_centered_coverage']], on=ID_KEYS, how='left', validate='one_to_one').merge(
        units[units.control == 'placebo'].drop(columns=['quantity_filter_applied']),
        on=ID_KEYS, how='left', validate='one_to_one')
    csv(output/'reports/coverage_and_effects.csv', coverage)
    csv(output/'reports/magnitude-absolute-utility.csv', magnitude_diagnostic(scores, units, pools, metadata))
    write_new_bytes(output/'reports/performance.csv', (SOURCE/'reports/performance.csv').read_bytes())
    boundary = ('这是已见404/505旧标签后的精度扩展，不是新的独立RL seed或确认性检验。'
        '固定U0/U5、37技能库、0.6B逐状态top1路由、全部404个首次调用锚点及全部285列读出；'
        '原7,272条续跑复用，新增14,544条。仅gold重复数2→8，evidence仍独立且不用于gold标签。'
        '训练、seen/unseen完整成功率、读出模型前向不重跑。\n\n'
        '所有arm/endpoint按相同anchor和续跑seed配对；环境seed和原前缀不变。'
        'seed编号等于404不代表更准确；新base间隔100避免新重复之间的50步seed区间重叠。'
        '旧63011/63021的base+step区间重叠是保留实现的边界，新增6组单独分析同时报告。\n\n'
        '主结果固定8组；2/4/8曲线及added6-only均固定输出，不按结果选用哪组。'
        '点估计等权game；区间配对game/continuation重采样，保留单game区间NA。'
        '原点标签指标不删除；区间跨0标为方向未分辨，零宽区间不能视为充分精度。'
        'bootstrap比例不是后验真值概率；不确定性未覆盖训练随机性、状态分布差异或多公式选择。\n\n')
    q = coverage[(coverage.phase == 'all') & coverage.utility_evaluable]
    cols = ['skill_id', 'source_calls', 'source_games', 'train_decisions', 'anchor_count', 'continuation_repeats',
            'utility_old', 'utility_new', 'delta_utility', 'ci_low', 'ci_high', 'direction_at_zero', 'interval_status']
    p1 = passport('utility_precision_seed404_v1')+'# Phase1：seed404，8组gold配对效用\n\n'+boundary
    p1 += q[cols].to_markdown(index=False)+'\n\n## 重复间方向稳定性\n\n'
    p1 += stability[(stability.control == 'placebo') & (stability.phase == 'all')].to_markdown(index=False)+'\n'
    primary_names = ['M_delta_centered::token::magnitude', 'D_original::token::reward',
        'D_signed_gate::token::reward', 'D_signed::token::reward', 'C_centered::token::reward',
        'D_real::token::reward', 'D_factor::token::reward', 'D_orientation::token::reward']
    primary = diagnostics[(diagnostics.control == 'placebo') & (diagnostics.phase == 'all')
        & (diagnostics.threshold == 0) & diagnostics.score.isin(primary_names)]
    p2 = passport('utility_precision_seed404_v1')+'# Phase2：冻结读出与提高精度后的效用标签\n\n'+boundary
    p2 += primary[['score', 'candidates', 'declines', 'increases', 'average_precision',
        'auroc_decline_vs_increase', 'auroc_decline_vs_rest', 'spearman']].to_markdown(index=False)+'\n\n'
    p2 += ('完整285指标及去reward对照、全部phase/control/0及5pp口径见CSV。'
        '历史公式搜索全部保留，不能从更新标签中选择赢家后称为预登记确认。'
        '配对指标差区间见paired_readout_gains.csv，不用两个边际区间重叠与否替代差值检验。\n\n'
        '## 统计解释检查\n\n覆盖11/11类：Simpson（保留分层）、生态谬误（不外推token/RL seed）、'
        'Berkson（共同池及完整NA覆盖）、collider（不按结果筛选）、基率（记录下降/上升数）、'
        '均值回归（不选极端技能追加）、幸存者偏差（必须全部完成）、多重检验（不宣布显著赢家）、'
        '分析路径（事后扩展披露）、相关/因果（不是编辑收益）、反向因果（冻结读出先于新标签）。\n')
    write_new_bytes(output/'reports/phase1-results.md', p1.encode())
    write_new_bytes(output/'reports/phase2-results.md', p2.encode())
    write_new_json(output/'independent-verification.json', checks)
    return {'metric_rows': len(diagnostics), 'score_columns': len(metadata), 'continuations': len(frame)}
