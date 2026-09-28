"""Full matched-pool analysis of one preregistered realized-update formula.

New scores are committed before opening utility labels. Historical seed404
labels were already seen: this ordering does not make the study prospective.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from .common import file_hash, read_json, write_new_bytes, write_new_json
from .realized_reward import FIELDS, KEYS, aggregate_scores, registry
from .reward_variants import AGGREGATIONS, EPS, aggregation_weights
from .reward_variant_analysis import (REFERENCE_COLUMNS, SIGN_METRICS, csv,
    interval, point_tables, pool_data, progress)
from .reward_variant_statistics import metric_arrays


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[''], float_precision='round_trip')


def sealed_artifacts(output):
    """Never hash live monitor logs or atomic-publication temporary files."""
    result = []
    for path in sorted(output.rglob('*')):
        parts = path.relative_to(output).parts
        # Filter before stat/hash: another process can unlink .publish-* at any time.
        if (any(p.startswith('.') for p in parts) or 'logs' in parts or 'heartbeats' in parts
                or path.name in ('workflow.log', 'runtime-status.json')):
            continue
        if path.is_file():
            result.append(path)
    return result


def metadata_for(plan):
    metadata = list(read_json(Path(plan['prior'])/'plan.json')['registry'])
    metadata += [{'score': 'B_'+name, 'name': name, 'family': 'retained_reference',
        'aggregation': 'original', 'mode': 'reference', 'signed': False, 'geometry': False}
        for name in REFERENCE_COLUMNS]
    metadata += registry()
    return metadata


def joined_tokens(output, plan):
    parts = []
    for shard in range(8):
        receipt = read_json(output/f'shard-{shard}.json')
        for name, field in ((f'tokens-shard-{shard}.parquet', 'tokens_sha256'),
                (f'input-audit-shard-{shard}.json', 'input_audit_sha256'),
                (f'witness-shard-{shard}.pt', 'witness_sha256')):
            if file_hash(output/name) != receipt[field]:
                raise ValueError('Shard artifact changed: '+name)
        if not receipt['all_stable_scalar_signals_exact']:
            raise ValueError('Old comparators were not reproduced')
        parts.append(pd.read_parquet(output/f'tokens-shard-{shard}.parquet'))
    extra = pd.concat(parts, ignore_index=True)
    old = pd.read_parquet(Path(plan['source'])/'token_signals.parquet')
    result = old.merge(extra, on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if not result._merge.eq('both').all() or len(result) != plan['expected_token_control_rows']:
        raise ValueError('Must cover ALL original token/control rows exactly once')
    result = result.drop(columns='_merge')
    if not np.isfinite(result[list(FIELDS)].to_numpy(float)).all():
        raise ValueError('Incomplete new scalar geometry')
    q = result.advantage.to_numpy(float)*result.chosen_u_original.to_numpy(float)
    if not np.array_equal(q, result.real_q.to_numpy(float)):
        raise ValueError('q must use the original uncentered action log likelihood')
    if not np.array_equal(-q*result.real_projection_coefficient, result.real_D):
        raise ValueError('Realized reward formula mismatch')
    if result.loc[result.advantage == 0, 'real_D'].ne(0).any():
        raise ValueError('Zero reward must contribute zero without dropping denominator rows')
    return result


def direction_table(scores, units, pools, metadata, thresholds):
    """Abstentions count as wrong in balanced accuracy; disclose class baselines."""
    rows = []
    for control in ('placebo', 'null'):
        for pool in pools[control+'/stable_raw']:
            if not pool['shared_skill_ids']:
                continue
            matrix, target, _ = pool_data(scores, units, pool, control, metadata)
            for threshold in thresholds:
                pos, neg = target > threshold+EPS, target < -threshold-EPS
                labeled = pos | neg
                for m, x in zip(metadata, matrix):
                    called = np.abs(x) > EPS
                    correct_pos = int(((x > EPS)&pos).sum())
                    correct_neg = int(((x < -EPS)&neg).sum())
                    n = int(labeled.sum())
                    base = {'control': control, 'context_id': pool['context_id'], 'phase': pool['phase'],
                        'threshold': threshold, 'score': m['score'], 'signed': m['signed'],
                        'candidates': len(target), 'declines': int(pos.sum()), 'increases': int(neg.sum()),
                        'always_decline_accuracy': float(pos.sum()/n) if n else np.nan,
                        'always_increase_accuracy': float(neg.sum()/n) if n else np.nan,
                        'majority_accuracy': float(max(pos.sum(), neg.sum())/n) if n else np.nan,
                        'constant_classifier_balanced_accuracy': .5 if pos.any() and neg.any() else np.nan}
                    # A reward-free scalar is a ranking control, not a calibrated direction call.
                    extra = {'correct_declines': correct_pos, 'correct_increases': correct_neg,
                        'abstentions': int((labeled&~called).sum()),
                        'balanced_accuracy_abstention_wrong': .5*(correct_pos/pos.sum()+correct_neg/neg.sum())
                            if pos.any() and neg.any() else np.nan}
                    rows.append({**base, **(extra if m['signed'] else {k: np.nan for k in extra})})
    return pd.DataFrame(rows)


def compare_bootstrap(scores, units, pools, metadata, plan):
    summaries, differences = [], []
    metrics_of_interest = ('auroc_decline_vs_increase', 'spearman', 'average_precision',
        'auroc_decline_vs_rest', 'sign_accuracy_called', 'sign_accuracy_abstention_wrong')
    names = [m['score'] for m in metadata]
    methods = {m['score']: m for m in metadata}
    new = [m for m in registry() if m['name'] == 'D_real']
    for control in ('placebo', 'null'):
        pool, = [p for p in pools[control+'/stable_raw'] if p['phase'] == 'all']
        matrix, target, _ = pool_data(scores, units, pool, control, metadata)
        draws = pd.read_parquet(Path(plan['prior'])/f'bootstrap-label-draws-{control}.parquet')
        if list(draws.columns) != pool['shared_skill_ids'] or len(draws) != 2000:
            raise ValueError('Changed paired bootstrap label draws')
        valid = np.isfinite(draws.to_numpy()).all(axis=1)
        for threshold in plan['event_thresholds']:
            arrays = metric_arrays(matrix, draws.to_numpy()[valid], threshold)
            point = metric_arrays(matrix, target[None], threshold)
            for method in registry():
                i = names.index(method['score'])
                for key, values in arrays.items():
                    if key in SIGN_METRICS and not method['signed']:
                        continue
                    summaries.append({'control': control, 'phase': 'all', 'threshold': threshold,
                        'score': method['score'], 'metric': key, 'point': float(point[key][0, i]),
                        **interval(values[:, i]), 'complete_pool_draws': int(valid.sum()),
                        'missing_any_skill_draws': int((~valid).sum()),
                        'scope': 'fixed-readout conditional gold-label uncertainty, not training-seed uncertainty'})
            for method in new:
                agg = method['aggregation']; sid = method['score']; i = names.index(sid)
                refs = [f'D_real_A1::{agg}::reward_free', f'D_real_absA::{agg}::reward_sign_removed',
                    f'D_real_projection_only::{agg}::geometry_only', f'D_original::{agg}::reward',
                    f'D_signed::{agg}::reward', f'C_centered::{agg}::reward',
                    f'M_delta_raw::{agg}::magnitude', f'M_delta_centered::{agg}::magnitude',
                    f'D_policy_action_adv::{agg}::reward', f'D_action_adv::{agg}::reward']
                for reference in refs:
                    j = names.index(reference)
                    for key in metrics_of_interest:
                        if key in SIGN_METRICS and not methods[reference]['signed']:
                            continue
                        diff = arrays[key][:, i]-arrays[key][:, j]
                        good = np.isfinite(diff)
                        differences.append({'control': control, 'phase': 'all', 'threshold': threshold,
                            'score': sid, 'reference': reference, 'metric': key,
                            'point_difference': float(point[key][0, i]-point[key][0, j]),
                            **interval(diff), 'draw_fraction_positive': float(np.mean(diff[good] > 0)) if good.any() else np.nan,
                            'not_a_confirmatory_p_value': True})
    return pd.DataFrame(summaries), pd.DataFrame(differences)


def sign_null(tokens, scores, units, pools, metadata, plan):
    """Reuse 512 trajectory-block sign masks; preserve all within-trajectory A."""
    masks = pd.read_parquet(Path(plan['prior'])/'sign-null-trajectory-masks.parquet')
    trajectories = list(masks.columns)
    if set(tokens.trajectory_id) != set(trajectories) or masks.shape != (512, 128):
        raise ValueError('Trajectory-block sign null changed')
    signs = masks.to_numpy(float)
    if not np.isin(signs, [-1., 1.]).all():
        raise ValueError('Not a +/-1 sign mask')
    ti = {t: i for i, t in enumerate(trajectories)}
    methods = [m for m in registry() if m['name'] == 'D_real']
    summaries, draws, maxima = [], [], []
    for control in ('placebo', 'null'):
        pool, = [p for p in pools[control+'/stable_raw'] if p['phase'] == 'all']
        matrix, target, _ = pool_data(scores, units, pool, control, methods)
        cube = np.empty((len(signs), len(methods), len(target)))
        part = tokens[tokens.control == control]
        for si, skill in enumerate(pool['shared_skill_ids']):
            g = part[(part.skill_id == skill)&(part.context_id == pool['context_id'])]
            ids = g.trajectory_id.map(ti).to_numpy(int)
            vals = g.real_D.to_numpy(float)
            lo, hi = np.full(len(ti), np.inf), np.full(len(ti), -np.inf)
            np.minimum.at(lo, ids, vals); np.maximum.at(hi, ids, vals)
            occupied = np.isfinite(lo)
            low = np.where(signs[:, occupied] > 0, lo[occupied], -hi[occupied]).min(axis=1)
            high = np.where(signs[:, occupied] > 0, hi[occupied], -lo[occupied]).max(axis=1)
            for ai, method in enumerate(methods):
                blocks = np.zeros(len(ti))
                np.add.at(blocks, ids, aggregation_weights(g, method['aggregation'])*vals)
                # Guard an analytic identity before using linear block sums.
                if not np.isclose(blocks.sum(), matrix[ai, si], rtol=1e-12, atol=1e-14):
                    raise ValueError('Null aggregation differs from observed score')
                cube[:, ai, si] = np.where(low == high, low, signs @ blocks)
        for threshold in plan['event_thresholds']:
            observed = metric_arrays(matrix, target[None], threshold)
            simulated = metric_arrays(cube.reshape(-1, len(target)), target[None], threshold)
            for key in ('auroc_decline_vs_increase', 'spearman', 'average_precision', 'auroc_decline_vs_rest'):
                values = simulated[key][0].reshape(len(signs), len(methods))
                finite_any = np.isfinite(values).any(axis=1)
                mx = np.max(np.where(np.isfinite(values), values, -np.inf), axis=1)
                mx[~finite_any] = np.nan
                for di, value in enumerate(mx):
                    maxima.append({'control': control, 'threshold': threshold, 'metric': key,
                        'draw': di, 'maximum_over_three_D_real_aggregations': value})
                for mi, method in enumerate(methods):
                    obs = float(observed[key][0, mi]); v = values[:, mi]; good = np.isfinite(v)
                    summaries.append({'control': control, 'phase': 'all', 'threshold': threshold,
                        'score': method['score'], 'metric': key, 'observed': obs, **interval(v),
                        'reference_fraction_ge_observed': float(np.mean(v[good] >= obs-EPS)) if good.any() and np.isfinite(obs) else np.nan,
                        'three_aggregation_max_fraction_ge_observed': float(np.mean(mx[finite_any] >= obs-EPS)) if finite_any.any() and np.isfinite(obs) else np.nan,
                        'not_a_confirmatory_p_value': True,
                        'does_not_adjust_historical_candidate_search': True})
                    draws.extend({'control': control, 'threshold': threshold, 'score': method['score'],
                        'metric': key, 'draw': di, 'value': float(value)} for di, value in enumerate(v))
    return pd.DataFrame(summaries), pd.DataFrame(draws), pd.DataFrame(maxima)


def verify_old_metrics(diag, prior):
    old = read_csv(prior/'ranking_diagnostics.csv')
    keys = ['control', 'context_id', 'phase', 'threshold', 'score']
    new = diag[diag.score.isin(old.score)]
    merged = new.merge(old, on=keys, how='outer', validate='one_to_one', indicator=True, suffixes=('_new', '_old'))
    if not merged._merge.eq('both').all():
        raise ValueError('Old comparison rows changed')
    for name in ('average_precision', 'auroc_decline_vs_rest', 'auroc_decline_vs_increase', 'spearman'):
        if not np.allclose(merged[name+'_new'], merged[name+'_old'], rtol=1e-11, atol=1e-12, equal_nan=True):
            raise ValueError('Preserved comparison metric changed: '+name)
    return {'old_metric_rows_reproduced': len(old), 'tolerance': 'rtol=1e-11 atol=1e-12'}


def independent_metrics(scores, units, pools, metadata, diag):
    """Independent sklearn/scipy check, including direction-class filtering."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from scipy.stats import spearmanr
    checks = 0
    for control in ('placebo', 'null'):
        for pool in pools[control+'/stable_raw']:
            if not pool['shared_skill_ids']:
                continue
            matrix, target, _ = pool_data(scores, units, pool, control, metadata)
            for threshold in (0., .05):
                selected = diag[(diag.control == control)&(diag.phase == pool['phase'])
                    &(diag.context_id == pool['context_id'])&(diag.threshold == threshold)].set_index('score')
                positive, negative = target > threshold+EPS, target < -threshold-EPS
                labeled = positive | negative
                for method, x in zip(metadata, matrix):
                    expected = {
                        'average_precision': average_precision_score(positive, x) if positive.any() else np.nan,
                        'auroc_decline_vs_rest': roc_auc_score(positive, x) if 0 < positive.sum() < len(x) else np.nan,
                        'auroc_decline_vs_increase': roc_auc_score(positive[labeled], x[labeled]) if positive.any() and negative.any() else np.nan,
                        'spearman': spearmanr(x, target).statistic if len(np.unique(x)) > 1 and len(np.unique(target)) > 1 else np.nan}
                    row = selected.loc[method['score']]
                    for key, value in expected.items():
                        if not np.isclose(value, row[key], rtol=1e-11, atol=1e-12, equal_nan=True):
                            raise ValueError('Independent metric disagreement: '+method['score']+' '+key)
                    checks += 1
    return {'independently_checked_rows': checks, 'libraries': ['sklearn', 'scipy'], 'passed': True}


def report(diag, comparisons, null, plan):
    chosen_names = ['D_real', 'D_real_A1', 'D_real_absA', 'D_real_projection_only',
        'D_original', 'D_signed', 'C_centered', 'M_delta_raw', 'M_delta_centered',
        'D_policy_action_adv', 'D_action_adv']
    main = diag[(diag.control == 'placebo')&(diag.phase == 'all')&(diag.threshold == 0)
                &diag.name.isin(chosen_names)]
    primary = main[main.aggregation == 'token']
    gain = comparisons[(comparisons.control == 'placebo')&(comparisons.threshold == 0)
        &comparisons.score.eq('D_real::token::reward')
        &comparisons.metric.isin(['auroc_decline_vs_increase', 'spearman'])]
    nullmain = null[(null.control == 'placebo')&(null.threshold == 0)&null.score.eq('D_real::token::reward')]
    return ('# Seed404：reward 校准的实际更新读出（新增探索）\n\n'
        'Material Passport — Origin Skill: academic-research-suite / experiment-agent; '
        'Mode: run/validate; Version: realized_reward_secant_v1; Status: ANALYZED，单 seed 事后探索。\n\n'
        '## 1. 公式与研究边界\n\n'
        '`u=log p5(skill)-log p0(skill); v=Hu; xi=Hdelta; q=A*u(a)`；'
        '`D_real=-Agg[q*dot(v,xi)/(||v||²+1e-12)]`。H为逐位置词表中心化。'
        '保留全部正负贡献与零优势分母，不额外门控。q使用未中心化的动作log概率差。\n\n'
        'q是固定优势局部log-likelihood surrogate的变化，不是实际PPO/GRPO总目标变化；'
        '该方向是rank-one/secant近似，不是纯reward梯度，更不是逐步因果credit。'
        'u同时出现在delta内，存在机械几何关联，必须对比A=1、abs(A)、仅投影系数和轨迹块符号随机化。\n\n'
        '主口径预先固定PLACEBO/all/token/0pp。decision与game聚合、NULL、阶段分层和5pp仅敏感性分析。'
        '旧404/505标签已见，先锁代码再计算也不能称为独立确认性检验。\n\n'
        '## 2. 方向预测主表\n\n'
        '下降风险固定为分数越高越大；原始delta是向量，幅度对照是||delta||与||Hdelta||。'
        '幅度或去reward对照没有校准的正负方向，不把其数值符号当预测。\n\n'+
        primary[['score', 'candidates', 'declines', 'increases', 'auroc_decline_vs_increase',
            'spearman', 'average_precision', 'auroc_decline_vs_rest', 'sign_accuracy_called',
            'sign_coverage']].to_markdown(index=False, floatfmt='.6f')+'\n\n'
        '完整三聚合表见 `reports/primary-all-aggregations.csv`；方向混淆、平衡准确率、'
        '恒预测上升/下降的类别基率对照见 `direction-confusions.csv`。\n\n'
        '## 3. 配对增益及不确定性\n\n'+gain.to_markdown(index=False, floatfmt='.6f')+'\n\n'
        '与v2共用2000组game×continuation配对标签bootstrap；任何技能缺失则整池draw失效，'
        '不逐方法删样本。区间仅反映固定readout条件下的标签不确定性，不是训练seed泛化区间。\n\n'
        '## 4. Reward符号负对照\n\n'+nullmain.to_markdown(index=False, floatfmt='.6f')+'\n\n'
        '共用512组trajectory-block ±1符号；同轨迹所有token一起翻转，保留优势绝对值与内部差异。'
        '这不是交换性已证明的随机化试验，不报告成确认性p值。三聚合max不校正此前39公式的历史搜索。\n\n'
        '## 5. 覆盖、数值和成本\n\n'
        '复用seed404 U0/U5、原U0实际训练批次、所有155638 token×control行；逐决策复现旧稳定分数。'
        '效用标签仍来自每轨迹每技能首次自然调用；25种效用技能、18种全阶段共同可比较技能。'
        '零优势和稀疏支持不额外过滤，覆盖/频次/game数以及极小更新范数单列记录。\n\n'
        '仅新增模型前向与标量读出；无新RL、环境rollout、API、505/606执行、阈值拟合或Phase3换指标。'
        '保留历史报告、125份原运行源码及既有v1/v2；所有旧比较指标复核，完整指标使用独立sklearn/scipy检查。\n\n'
        '## 6. 统计误用检查（11/11）\n\n'
        '1.非显著不等于等效；2.效应量与区间并报；3.技能不是独立token样本；'
        '4.类别基率及弃权透明；5.平均值不掩盖阶段/聚合差异；6.稀疏技能支持量不隐去；'
        '7.配对及共享game依赖保持；8.不从多重尝试择优宣称确认；9.已见标签探索如实标注；'
        '10.预测相关不等于编辑因果收益；11.实际更新共项可能导致机械相关，负对照不能自动证明无泄漏。\n\n'
        '这次目标是检验方向性增益，而不是保证新公式获胜。即使单seed更优，'
        '仍需先冻结少数指标再用于独立seed/窗口，不能自动替换Phase3主指标。\n')


def analyze(output):
    from .realized_reward_run import binding
    output = Path(output).resolve(); plan = binding(output)
    if (output/'analysis-intent.json').exists():
        raise FileExistsError('No automatic analysis retry')
    write_new_json(output/'analysis-intent.json', {'started_unix': time.time(), 'plan_sha256': file_hash(output/'plan.json')})
    tokens = joined_tokens(output, plan)
    new_scores = aggregate_scores(tokens)
    prior, source = Path(plan['prior']), Path(plan['source'])
    scores = pd.concat([read_csv(prior/'skill_scores.csv'), new_scores], ignore_index=True)
    metadata = metadata_for(plan)
    if set(scores.score) != {m['score'] for m in metadata}:
        raise ValueError('Registry and score columns differ')
    write_new_bytes(output/'token_signals.parquet', tokens.to_parquet(index=False))
    csv(output/'skill_scores.csv', scores); csv(output/'registry.csv', pd.DataFrame(metadata))
    write_new_json(output/'score-commit.json', {'created_unix': time.time(), 'scores_sha256': file_hash(output/'skill_scores.csv'),
        'tokens_sha256': file_hash(output/'token_signals.parquet'), 'gold_read_for_new_scoring': False,
        'historical_labels_previously_seen': True, 'not_prospective': True,
        'tokens_with_controls': len(tokens), 'decisions': tokens.decision_id.nunique()})
    # Gold is first opened here, after every new score has been published.
    units = pd.read_parquet(source/'reused-labels/utility_units.parquet')
    pools = read_json(source/'ranking-snapshots.json')['candidate_pools']
    diag, budgets = point_tables(scores, units, pools, metadata, plan['event_thresholds'])
    oldcheck = verify_old_metrics(diag, prior)
    independent = independent_metrics(scores, units, pools, metadata, diag)
    csv(output/'ranking_diagnostics.csv', diag); csv(output/'ranking_budgets.csv', budgets)
    csv(output/'direction-confusions.csv', direction_table(scores, units, pools, metadata, plan['event_thresholds']))
    csv(output/'skill_scores_and_gold.csv', scores.merge(units,
        on=['control', 'context_id', 'skill_id', 'phase'], how='left', validate='many_to_one'))
    summary, gains = compare_bootstrap(scores, units, pools, metadata, plan)
    csv(output/'bootstrap_summary.csv', summary); csv(output/'paired_direction_gains.csv', gains)
    null, draws, maxima = sign_null(tokens, scores, units, pools, metadata, plan)
    csv(output/'sign_null_summary.csv', null); csv(output/'sign_null_draws.csv', draws); csv(output/'sign_null_maxima.csv', maxima)
    coverage = tokens.groupby(['control', 'skill_id']).agg(tokens=('decision_id', 'size'),
        decisions=('decision_id', 'nunique'), trajectories=('trajectory_id', 'nunique'), games=('game_id', 'nunique'),
        zero_advantage_tokens=('advantage', lambda x: int((x == 0).sum())),
        tiny_centered_update_tokens=('real_small_u', 'sum'),
        min_u_centered_norm_sq=('real_u_centered_norm_sq', 'min'),
        max_abs_projection=('real_projection_coefficient', lambda x: float(x.abs().max())),
        mean_q=('real_q', 'mean')).reset_index()
    csv(output/'geometry-and-support-audit.csv', coverage)
    write_new_bytes(output/'coverage_and_effects.csv', (source/'reports/coverage_and_effects.csv').read_bytes())
    csv(output/'reports/primary-all-aggregations.csv', diag[(diag.control == 'placebo')&(diag.phase == 'all')&(diag.threshold == 0)])
    text = report(diag, gains, null, plan).encode()
    write_new_bytes(output/'reports/realized-reward-analysis.md', text)
    write_new_bytes(output/'reports/phase2-results-expanded.md',
        (prior/'reports/phase2-results-expanded-v2.md').read_bytes()+b'\n\n---\n\n'+text)
    binding(output, full=True)
    write_new_json(output/'independent-verification.json', {**oldcheck, **independent,
        'all_original_hashes_unchanged': True, 'score_columns': len(metadata), 'metric_rows': len(diag)})
    files = sealed_artifacts(output)
    write_new_json(output/'provenance.json', {'plan_sha256': file_hash(output/'plan.json'),
        'scientific_status': 'ANALYZED_exploratory_single_seed',
        'files': [{'path': str(p.relative_to(output)), 'sha256': file_hash(p)} for p in files]})
    write_new_json(output/'complete.json', {'status': 'complete', 'seed': 404,
        'provenance_sha256': file_hash(output/'provenance.json'), 'finished_unix': time.time(),
        'new_training': False, 'new_environment_rollouts': 0, 'new_api_calls': 0,
        'old_reports_preserved': True, 'new_formulas': 1, 'new_score_columns_with_controls': len(registry()),
        'all_score_columns': len(metadata), 'metric_rows': len(diag)})
    progress('COMPLETE', output=str(output), report=str(output/'reports/realized-reward-analysis.md'))
