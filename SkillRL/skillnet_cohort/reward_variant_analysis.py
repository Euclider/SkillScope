"""One-shot, no-clobber seed404 exploratory scalar analysis; no GPU or API."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .common import REPO, file_hash, read_json, write_new_bytes, write_new_json
from .reward_variants import (AGGREGATIONS, EPS, MAGNITUDES, RECIPES, aggregation_weights,
    label_blind_scales, magnitude_matrix, registry, score_id, token_matrix, validate_tokens)
from .reward_variant_statistics import (bootstrap_targets, budget_arrays, metric_arrays,
    point_metrics_many)

COHORT = REPO/'artifacts/phase12/skillnet37-independent-s404-505-606-v4'
SOURCE = COHORT/'numerical-readout-release-v1/seed-404'
UTILITY = COHORT/'all-first-calls-v1/seed-404'
DEFAULT_OUTPUT = COHORT/'reward-variants-s404-v1'
CODE = ['skillnet_cohort/reward_variants.py', 'skillnet_cohort/reward_variant_statistics.py',
        'skillnet_cohort/reward_variant_analysis.py', 'tests/skillnet_cohort/test_reward_variants.py']
REFERENCE_COLUMNS = ['activation_l8_norm', 'activation_l16_norm', 'activation_l24_norm',
                     'activation_l32_norm', 'old_margin', 'random_expected']
SIGN_METRICS = ['sign_accuracy_called', 'sign_coverage', 'sign_accuracy_abstention_wrong']


def progress(stage, **details):
    print(json.dumps({'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **details}), flush=True)


def csv(path, frame):
    write_new_bytes(path, frame.to_csv(index=False).encode())


def verify_records(records):
    for record in records:
        if file_hash(record['path']) != record['sha256']:
            raise ValueError('Immutable input/source changed: '+record['path'])


def check_origin():
    complete, provenance, commit = (read_json(SOURCE/f) for f in ('complete.json', 'report-provenance.json', 'committed.json'))
    if complete['status'] != 'complete' or complete['provenance_sha256'] != file_hash(SOURCE/'report-provenance.json'):
        raise ValueError('Incomplete or changed numerical source')
    verify_records([{'path': str(SOURCE/r['path']), 'sha256': r['sha256']} for r in provenance['files']])
    if commit['features_sha256'] != file_hash(SOURCE/'skill_context_features.parquet') or commit['token_signals_sha256'] != file_hash(SOURCE/'token_signals.parquet'):
        raise ValueError('Readout is not the committed numerical correction')
    if not commit['all_legacy_scalar_signals_exact']:
        raise ValueError('Legacy readout comparison failed')


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Prepare requires a NEW output directory')
    if output.parent != COHORT or not output.name.startswith('reward-variants-s404-'):
        raise ValueError('Only scoped seed404 scalar output is authorized')
    check_origin()
    old_plan = read_json(SOURCE.parent/'plan.json')
    frozen = [{'path': str(REPO/p), 'sha256': h} for p, h in old_plan['source_sha256'].items()]
    verify_records(frozen)
    inputs = [SOURCE/n for n in ('token_signals.parquet', 'skill_context_features.parquet',
        'committed.json', 'complete.json', 'report-provenance.json', 'ranking-snapshots.json')]
    inputs += sorted(p for p in (SOURCE/'reports').iterdir() if p.is_file())
    inputs += sorted(p for p in (SOURCE/'reused-labels').iterdir() if p.is_file())
    inputs += [UTILITY/'protocol.json', UTILITY/'support/coverage.csv', SOURCE.parent/'plan.json']
    plan = {'version': 'reward_variants_seed404_exploratory_v1', 'approved': True, 'seed': 404,
        'user_authority': '2026-09-22: 扩展reward相关D变式，在404评估并扩充分析报告',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'output': str(output),
        'source': str(SOURCE), 'prior_404_and_505_labels_seen': True,
        'scientific_status': 'exploratory_not_confirmatory', 'automatic_retry': False,
        'new_training': False, 'new_model_forward': False, 'new_environment_rollouts': 0,
        'new_api_calls': 0, 'other_seeds_evaluated': [], 'original_reports_overwritten': False,
        'bootstrap_repetitions': 2000, 'bootstrap_rng_seed': 20260922,
        'sign_null_repetitions': 512, 'sign_null_rng_seed': 20260923,
        'event_thresholds': [0., .05], 'primary': {'control': 'placebo', 'phase': 'all', 'threshold': 0.},
        'registry': registry(), 'reference_columns': REFERENCE_COLUMNS,
        'inputs': [{'path': str(p), 'sha256': file_hash(p)} for p in inputs],
        'preserved_runtime_sources': frozen,
        'analysis_sources': [{'path': str(REPO/p), 'sha256': file_hash(REPO/p)} for p in CODE],
        'setting_document': {'path': str(REPO/'docs/experiments/phase12-independent-v4/REWARD-VARIANTS-SEED404-20260922-v1.md'),
            'sha256': file_hash(REPO/'docs/experiments/phase12-independent-v4/REWARD-VARIANTS-SEED404-20260922-v1.md')},
    }
    write_new_json(output/'plan.json', plan)
    csv(output/'registry.csv', pd.DataFrame(plan['registry']))
    progress('PREPARED_NOT_SCORED', candidates=len(plan['registry']), reward_recipes=len(RECIPES), output=str(output))
    return plan


def check_plan(output):
    plan = read_json(output/'plan.json')
    if plan['output'] != str(output) or not plan['approved'] or plan['seed'] != 404 or plan['registry'] != registry():
        raise ValueError('Wrong scope or changed candidate registry')
    verify_records(plan['inputs']+plan['preserved_runtime_sources']+plan['analysis_sources']+[plan['setting_document']])
    return plan


def compute_scores(t, scales):
    """No utility input. Every score is generated before the runner opens labels."""
    rows = []; blocks = {}
    for control in ('placebo', 'null'):
        part = t[t.control == control].reset_index(drop=True)
        matrices = {'reward': token_matrix(part, scales[control]),
                    'unsigned': token_matrix(part, scales[control], 'unsigned'),
                    'magnitude': magnitude_matrix(part)}
        flipped = token_matrix(part, scales[control], flip=-1.)
        trajectories = sorted(part.trajectory_id.unique())
        trajectory_index = {key: i for i, key in enumerate(trajectories)}
        for phase in ('all', 'initial', 'early', 'middle', 'late'):
            subset = part if phase == 'all' else part[part.phase == phase]
            for (context, skill), group in subset.groupby(['context_id', 'skill_id'], sort=True):
                indices = group.index.to_numpy()
                for aggregation in AGGREGATIONS:
                    weights = aggregation_weights(group, aggregation)
                    for mode, matrix in matrices.items():
                        names = [r.name for r in RECIPES] if mode != 'magnitude' else list(MAGNITUDES)
                        values = weights @ matrix[indices]
                        for name, value in zip(names, values):
                            rows.append({'control': control, 'context_id': context, 'phase': phase,
                                'skill_id': skill, 'score': score_id(name, aggregation, mode),
                                'value': float(value), 'token_count': len(group),
                                'decision_count': group.decision_id.nunique(),
                                'trajectory_count': group.trajectory_id.nunique(), 'game_count': group.game_id.nunique()})
                    if phase == 'all':
                        plus = np.zeros((len(trajectories), len(RECIPES)))
                        minus = np.zeros_like(plus)
                        ids = group.trajectory_id.map(trajectory_index).to_numpy(int)
                        np.add.at(plus, ids, matrices['reward'][indices]*weights[:, None])
                        np.add.at(minus, ids, flipped[indices]*weights[:, None])
                        blocks[(control, context, skill, aggregation)] = (trajectories, plus, minus)
        progress('SCORED_CONTROL', control=control, rows=len(rows))
    return pd.DataFrame(rows), blocks


def add_references(scores, metadata, old_scores):
    stable = old_scores[old_scores.variant == 'stable_raw']
    rows = []
    for name in REFERENCE_COLUMNS:
        sid = 'B_'+name
        metadata.append({'score': sid, 'name': name, 'family': 'retained_reference',
            'aggregation': 'original', 'mode': 'reference', 'signed': False, 'geometry': False})
        for row in stable.itertuples():
            value = getattr(row, name)
            if np.isfinite(value):
                rows.append({'control': row.control, 'context_id': row.context_id, 'phase': row.phase,
                    'skill_id': row.skill_id, 'score': sid, 'value': value})
    return pd.concat([scores, pd.DataFrame(rows)], ignore_index=True)


def audit_legacy(scores, source_features):
    mapping = {'D_original': ('D_contribution', 1), 'D_centered_gate': ('D_centered_contribution', 1),
        'D_ungated': ('D_ungated_contribution', 1), 'D_signed': ('P_int', -1),
        'C_raw': ('C_upd', 1), 'C_centered': ('C_upd_centered', 1)}
    mapping.update({k: (v, 1) for k, v in MAGNITUDES.items() if not v.startswith('abs(')})
    records = []; keys = ['control', 'context_id', 'skill_id', 'phase']
    for name, (column, sign) in mapping.items():
        mode = 'magnitude' if name in MAGNITUDES else 'reward'
        q = scores[scores.score == score_id(name, 'token', mode)].merge(source_features[keys+[column]], on=keys, validate='one_to_one')
        q = q[np.isfinite(q[column])]
        if not np.allclose(q.value, sign*q[column], rtol=1e-11, atol=1e-12):
            raise ValueError('Existing scalar aggregation changed: '+name)
        records.append({'name': name, 'matched_rows': len(q), 'max_abs_error': float(np.max(np.abs(q.value-sign*q[column]))), 'passed': True})
    return records


def pool_data(scores, units, pool, control, metadata):
    keys = pool['shared_skill_ids']
    q = scores[(scores.control == control)&(scores.context_id == pool['context_id'])&(scores.phase == pool['phase'])]
    names = [m['score'] for m in metadata]
    matrix = q.pivot(index='score', columns='skill_id', values='value').reindex(index=names, columns=keys).to_numpy(float)
    gold = units[(units.control == control)&(units.context_id == pool['context_id'])&(units.phase == pool['phase'])].set_index('skill_id').reindex(keys)
    if not np.isfinite(matrix).all() or gold.delta_utility.isna().any():
        raise ValueError('Registered identical candidate pool lost a score/label')
    return matrix, -gold.delta_utility.to_numpy(float), gold


def point_tables(scores, units, pools, metadata, thresholds):
    diagnostics, budgets = [], []
    for control in ('placebo', 'null'):
        for pool in pools[control+'/stable_raw']:
            if not pool['shared_skill_ids']:
                continue
            matrix, target, _ = pool_data(scores, units, pool, control, metadata)
            n = len(target)
            for threshold in thresholds:
                metrics = metric_arrays(matrix, target[None, :], threshold)
                base = {'control': control, 'context_id': pool['context_id'], 'phase': pool['phase'],
                    'threshold': threshold, 'candidates': n, 'declines': int((target > threshold+EPS).sum()),
                    'increases': int((target < -threshold-EPS).sum()), 'unchanged_or_small': int((np.abs(target) <= threshold+EPS).sum())}
                for i, meta in enumerate(metadata):
                    values = {k: float(v[0, i]) for k, v in metrics.items()}
                    if not meta['signed']:
                        values.update({key: np.nan for key in SIGN_METRICS})
                    values['kendall'] = (float(kendalltau(matrix[i], target).statistic)
                        if len(np.unique(matrix[i])) > 1 and len(np.unique(target)) > 1 else np.nan)
                    diagnostics.append({**base, **meta, **values})
                for k in sorted({min(n, 1), min(n, 2), int(np.ceil(n*.25)), int(np.ceil(n*.5))}):
                    for i, meta in enumerate(metadata):
                        values = budget_arrays(matrix[i:i+1], target[None, :], k, threshold)
                        budgets.append({**base, **meta, 'k': k, **{key: float(value[0, 0]) for key, value in values.items()}})
    return pd.DataFrame(diagnostics), pd.DataFrame(budgets)


def interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return {'low': float(np.quantile(finite, .025)) if len(finite) else np.nan,
            'high': float(np.quantile(finite, .975)) if len(finite) else np.nan,
            'mean': float(finite.mean()) if len(finite) else np.nan, 'valid_draws': len(finite)}


def bootstrap_analysis(scores, units, margins, pools, metadata, plan):
    rows, comparisons, receipts, draws_out = [], [], {}, {}
    lookup = {r['score']: i for i, r in enumerate(metadata)}
    for control in ('placebo', 'null'):
        pool, = [p for p in pools[control+'/stable_raw'] if p['phase'] == 'all']
        matrix, target, _ = pool_data(scores, units, pool, control, metadata)
        draws, point, receipt = bootstrap_targets(margins[margins.context_id == pool['context_id']],
            pool['shared_skill_ids'], control, plan['bootstrap_repetitions'], plan['bootstrap_rng_seed'])
        if not np.allclose(target, point, atol=1e-14, rtol=1e-14):
            raise ValueError('Paired retained labels do not reproduce original game means')
        receipts[control] = receipt
        draws_out[control] = pd.DataFrame(draws, columns=pool['shared_skill_ids'])
        valid = draws[np.isfinite(draws).all(axis=1)]
        for threshold in plan['event_thresholds']:
            arrays = metric_arrays(matrix, valid, threshold)
            observed = metric_arrays(matrix, target[None, :], threshold)
            for i, meta in enumerate(metadata):
                base = {'control': control, 'phase': 'all', 'threshold': threshold, 'score': meta['score']}
                for key, vals in arrays.items():
                    if key in SIGN_METRICS and not meta['signed']:
                        continue
                    rows.append({**base, 'metric': key, 'point': float(observed[key][0, i]), **interval(vals[:, i])})
                if meta['mode'] != 'reward':
                    continue
                references = [('original_D', score_id('D_original', 'token')),
                    ('same_aggregation_centered_magnitude', score_id('M_delta_centered', meta['aggregation'], 'magnitude')),
                    ('matched_unsigned', score_id(meta['name'], meta['aggregation'], 'unsigned'))]
                for ref_kind, reference in references:
                    for key in ('average_precision', 'auroc_decline_vs_rest', 'spearman'):
                        j = lookup[reference]; diff = arrays[key][:, i]-arrays[key][:, j]
                        comparisons.append({**base, 'reference_kind': ref_kind, 'reference': reference, 'metric': key,
                            'point_difference': float(observed[key][0, i]-observed[key][0, j]), **interval(diff)})
            for k in sorted({1, 2, int(np.ceil(len(target)*.25)), int(np.ceil(len(target)*.5))}):
                values = budget_arrays(matrix, valid, k, threshold)
                points = budget_arrays(matrix, target[None, :], k, threshold)
                for key, vals in values.items():
                    for i, meta in enumerate(metadata):
                        rows.append({'control': control, 'phase': 'all', 'threshold': threshold,
                            'score': meta['score'], 'metric': key, 'k': k,
                            'point': float(points[key][0, i]), **interval(vals[:, i])})
        progress('BOOTSTRAP_CONTROL', control=control, **receipt)
    return pd.DataFrame(rows), pd.DataFrame(comparisons), receipts, draws_out


def sign_null_analysis(blocks, scores, units, pools, metadata, plan):
    reward = [m for m in metadata if m['mode'] == 'reward']
    trajectory_lists = [v[0] for v in blocks.values()]
    trajectories = trajectory_lists[0]
    if any(v != trajectories for v in trajectory_lists):
        raise ValueError('Control blocks do not share trajectory identities')
    rng = np.random.default_rng(plan['sign_null_rng_seed'])
    flipped = rng.integers(0, 2, size=(plan['sign_null_repetitions'], len(trajectories)))
    summaries, outputs, family_max = [], [], []
    for control in ('placebo', 'null'):
        pool, = [p for p in pools[control+'/stable_raw'] if p['phase'] == 'all']
        _, target, _ = pool_data(scores, units, pool, control, metadata)
        cube = np.empty((len(flipped), len(reward), len(target)))
        for si, skill in enumerate(pool['shared_skill_ids']):
            for ai, aggregation in enumerate(AGGREGATIONS):
                _, plus, minus = blocks[(control, pool['context_id'], skill, aggregation)]
                cube[:, ai*len(RECIPES):(ai+1)*len(RECIPES), si] = plus.sum(axis=0)+flipped @ (minus-plus)
        observed_matrix = scores[(scores.control == control)&(scores.phase == 'all')&(scores.context_id == pool['context_id'])].pivot(index='score', columns='skill_id', values='value').reindex(index=[m['score'] for m in reward], columns=pool['shared_skill_ids']).to_numpy()
        for threshold in plan['event_thresholds']:
            simulated = point_metrics_many(cube.reshape(-1, len(target)), target, threshold)
            observed = point_metrics_many(observed_matrix, target, threshold)
            arrays = {key: vals.reshape(len(flipped), len(reward)) for key, vals in simulated.items()}
            max_ap = np.max(arrays['average_precision'], axis=1)
            for draw, value in enumerate(max_ap):
                family_max.append({'control': control, 'threshold': threshold, 'draw': draw, 'maximum_reward_candidate_AP': value})
            for i, meta in enumerate(reward):
                for key, values in arrays.items():
                    v = values[:, i]; obs = observed[key][i]
                    summaries.append({'control': control, 'phase': 'all', 'threshold': threshold,
                        'score': meta['score'], 'metric': key, 'observed': float(obs),
                        **interval(v), 'reference_fraction_ge_observed': float(np.mean(v >= obs-EPS)) if np.isfinite(obs) else np.nan,
                        'family_max_fraction_ge_observed_AP': float(np.mean(max_ap >= obs-EPS)) if key == 'average_precision' and np.isfinite(obs) else np.nan,
                        'is_calibrated_p_value': False})
                outputs.extend({'control': control, 'threshold': threshold, 'score': meta['score'],
                    'draw': j, **{key: float(vals[j, i]) for key, vals in arrays.items()}}
                    for j in range(len(flipped)))
        progress('SIGN_NULL_CONTROL', control=control, draws=len(flipped), reward_scores=len(reward))
    masks = pd.DataFrame(1-2*flipped, columns=trajectories)
    return pd.DataFrame(summaries), pd.DataFrame(outputs), pd.DataFrame(family_max), masks


def markdown_report(diag, budgets, comparison, bootstrap, null, metadata, audit, plan, receipts):
    q = diag[(diag.control == 'placebo')&(diag.phase == 'all')&(diag.threshold == 0)]
    reward = q[q['mode'] == 'reward'].sort_values(['average_precision', 'score'], ascending=[False, True])
    baseline = q[q['mode'].isin(['magnitude', 'reference'])]
    # Display winners as explicitly retrospective summaries; keep ALL candidates below/on disk.
    leaders = reward.groupby('family', sort=False).head(1)
    main_cols = ['score', 'average_precision', 'auroc_decline_vs_rest', 'auroc_decline_vs_increase',
                 'spearman', 'sign_accuracy_called', 'sign_coverage']
    token = q[(q['mode'] == 'reward')&(q.aggregation == 'token')]
    head = ('## Material Passport\n\n- Origin Skill: experiment-agent\n- Origin Mode: run / validate\n'
        f'- Origin Date: {datetime.now(timezone.utc).isoformat()}\n- Verification Status: ANALYZED\n'
        '- Version Label: reward_variants_seed404_exploratory_v1\n\n')
    body = head+'# seed404：reward-directed 读出变式扩展分析\n\n'
    body += (f'共 {len(RECIPES)} 个奖励相关公式 × 3 种聚合 = {len(reward)} 个 reward 分数，'
        f'另有相应去奖励对照、幅度和旧基线，共 {len(metadata)} 列。主池 '
        f'{int(q.iloc[0].candidates)} 技能，0pp 点估计下降 {int(q.iloc[0].declines)} 种；'
        '不是独立重复实验的数量。所有公式和方向在本次实际计算前登记，但历史标签已见，属于事后探索。\n\n'
        '目标是检验 reward 是否提供增量信息，而不是必须找出赢家。最高 AP 仅为描述性最优，'
        '不能把同一 seed 上大量尝试后的改善写成已证实泛化或 Phase3 编辑收益。原 C/中心化 C 本身含 reward。\n\n')
    body += '## 1. 相同候选池：各家族探索性最高 AP（不是确认性选型）\n\n'+leaders[main_cols].to_markdown(index=False, floatfmt='.5f')+'\n\n'
    body += '所有家族、聚合和负对照均在完整 CSV；这里的最佳值包含公式/聚合选择偏差。\n\n'
    body += '## 2. 无 reward 幅度与原有参考\n\n'+baseline[main_cols[:5]].to_markdown(index=False, floatfmt='.5f')+'\n\n'
    body += '## 3. 全部奖励公式：固定 token 等权\n\n'+token[main_cols].to_markdown(index=False, floatfmt='.5f')+'\n\n'
    body += '## 4. reward 的配对增益，而非仅看最高 AP\n\n'
    ids = leaders.score.tolist()
    z = comparison[(comparison.control == 'placebo')&(comparison.threshold == 0)&(comparison.metric == 'average_precision')&comparison.score.isin(ids)]
    body += z[['score', 'reference_kind', 'point_difference', 'low', 'high', 'valid_draws']].to_markdown(index=False, floatfmt='.5f')+'\n\n'
    body += ('差值是 reward 版本 AP 减去参考 AP；区间为固定读出下配对 gold-label bootstrap，'
        '未校正候选选择，也未包含 RL seed/训练读出不确定性。P/C 几何去奖励版沿用原 A!=0 支持，'
        '只移除该支持上的符号/强度，不能作为全集完全 reward-free 的证据。采样动作类 A=+1 对照覆盖全部 token。\n\n')
    body += '## 5. trajectory-block reward 符号负对照\n\n'
    z = null[(null.control == 'placebo')&(null.threshold == 0)&(null.metric == 'average_precision')&null.score.isin(ids)]
    body += z[['score', 'observed', 'mean', 'low', 'high', 'reference_fraction_ge_observed',
               'family_max_fraction_ge_observed_AP']].to_markdown(index=False, floatfmt='.5f')+'\n\n'
    body += ('512 次随机化保留轨迹内相关性与 |A|，重新计算 gate；最后一列用每次随机化的所有 reward 候选最高 AP 做参照，'
        '用于显示 look-elsewhere 风险。比例不是校准 p 值，不能据此宣称显著。\n\n')
    body += '## 6. 方向分类、编辑预算与 NULL 稳健性\n\n'
    z = budgets[(budgets.control == 'placebo')&(budgets.phase == 'all')&(budgets.threshold == 0)&budgets.score.isin(ids)]
    body += z[['score', 'k', 'precision_at_k', 'recall_at_k', 'captured_decline_mass']].to_markdown(index=False, floatfmt='.5f')+'\n\n'
    z = diag[(diag.control == 'null')&(diag.phase == 'all')&(diag.threshold == 0)&diag.score.isin(ids)]
    body += z[['score', 'candidates', 'declines']+main_cols[1:]].to_markdown(index=False, floatfmt='.5f')+'\n\n'
    body += ('带符号方法记录非零 ΔM 的符号准确率、弃权和覆盖；完整 CSV 同时给出弃权计错的准确率。'
        '非负 D 与作为风险分数使用的 C 不提供双向分类准确率。AP/AUROC 衡量排序，不等于零阈值方向判断。'
        '0/5pp 标签阈值、全部 phase 及所有方法的预算均保留；phase 不是额外独立窗口。\n\n')
    body += '## 7. 原指标复现、样本与不确定性边界\n\n'+pd.DataFrame(audit).to_markdown(index=False)+'\n\n'
    body += pd.DataFrame([{'control': key, **value} for key, value in receipts.items()]).to_markdown(index=False)+'\n\n'
    body += ('全部效用标签复用已保存的 U0/U5 配对结果；旧区间和报告原文不改。新增 bootstrap 在任一技能缺少抽中 game 时整池记缺失，'
        '不为不同方法挑选不同技能。置信区间条件于当前自然出现技能和训练读出，不支持跨 seed 泛化。'
        '无额外 RL、模型前向、环境 rollout 或 API；505/606 未进行本轮变式分析。\n\n')
    body += '## 8. 统计误用核查：11/11\n\n'
    body += ('- Simpson：control/phase 分层，不用合并结果替代分层。\n'
        '- 生态谬误：skill-window 单位，不把 token/anchor/重复当独立 seed。\n'
        '- Berkson：旧共同池保持，完整自然支持及缺失可查。\n'
        '- Collider：未按结果新增支持筛选或拟合控制变量。\n'
        '- 基率：每表保留候选/下降/上升数，单类或零事件指标 NA。\n'
        '- 均值回归：不按极端 ΔM 选技能、检查点或重采样。\n'
        '- 幸存者：全部候选和 NA 记录，不仅报告赢家。\n'
        '- 多重检验：完整枚举与 family-max 符号负对照；不作未校正显著性声明。\n'
        '- 分析路径：已见标签的事后探索，代码/输入/候选先锁定，未伪装前瞻验证。\n'
        '- 相关非因果：预测关联不证明编辑收益。\n'
        '- 反向因果/泄漏：评分函数不读效用标签；已有标签知识仍限制确认性解释。\n\n')
    body += ('## 9. 文件与后续\n\n'
        '`registry.csv` 列出公式与全部口径；`skill_scores.csv` 为逐技能原分数；'
        '`ranking_diagnostics.csv`、`ranking_budgets.csv` 是全分层结果；'
        '`paired_reward_gains.csv`、`bootstrap_summary.csv`、`sign_null_summary.csv` 为配对/不确定性对照。'
        '不要从本次多个赢家中逐 seed 切换指标；如继续验证，需先冻结有限候选，再用于新的独立窗口/seed。\n')
    return body


def run(output):
    output = Path(output).resolve()
    if (output/'run-intent.json').exists():
        raise FileExistsError('No automatic rerun/retry of an attempted analysis')
    plan = check_plan(output)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicit CPU-only execution required')
    started = time.time()
    write_new_json(output/'run-intent.json', {'started_unix': started, 'pid': os.getpid(),
        'plan_sha256': file_hash(output/'plan.json'), 'python': platform.python_version(), 'gpu_used': False})
    t = pd.read_parquet(SOURCE/'token_signals.parquet'); validate_tokens(t)
    scales = {control: label_blind_scales(t[t.control == control]) for control in ('placebo', 'null')}
    write_new_json(output/'label-blind-scales.json', scales)
    scores, blocks = compute_scores(t, scales)
    audit = audit_legacy(scores, pd.read_parquet(SOURCE/'skill_context_features.parquet'))
    metadata = list(plan['registry'])
    # Pandas' default NA vocabulary treats the literal control name "null" as
    # missing. Preserve arm identities and treat only empty cells as missing.
    scores = add_references(scores, metadata, pd.read_csv(SOURCE/'reports/scores_and_gold.csv',
        keep_default_na=False, na_values=['']))
    csv(output/'skill_scores.csv', scores)
    write_new_json(output/'score-commit.json', {'sha256': file_hash(output/'skill_scores.csv'),
        'created_unix': time.time(), 'new_formulas_computed_without_gold_input': True,
        'prior_labels_already_seen': True, 'not_prospective': True, 'scales_sha256': file_hash(output/'label-blind-scales.json')})
    # Only now open outcomes for new method comparisons. Retained baseline file
    # contains historical labels, but add_references copies SCORE columns only.
    units = pd.read_parquet(SOURCE/'reused-labels/utility_units.parquet')
    margins = pd.read_parquet(SOURCE/'reused-labels/anchor_margins.parquet')
    pools = read_json(SOURCE/'ranking-snapshots.json')['candidate_pools']
    diag, budgets = point_tables(scores, units, pools, metadata, plan['event_thresholds'])
    csv(output/'ranking_diagnostics.csv', diag); csv(output/'ranking_budgets.csv', budgets)
    joined = scores.merge(units, on=['control', 'context_id', 'skill_id', 'phase'],
        how='left', validate='many_to_one')
    joined['utility_available'] = joined.delta_utility.notna()
    csv(output/'skill_scores_and_gold.csv', joined)
    progress('POINT_METRICS', methods=len(metadata), rows=len(diag))
    bootstrap, comparison, receipts, draws = bootstrap_analysis(scores, units, margins, pools, metadata, plan)
    csv(output/'bootstrap_summary.csv', bootstrap); csv(output/'paired_reward_gains.csv', comparison)
    write_new_json(output/'bootstrap-receipts.json', receipts)
    for control, frame in draws.items():
        write_new_bytes(output/f'bootstrap-label-draws-{control}.parquet', frame.to_parquet(index=False))
    null, null_draws, maxima, masks = sign_null_analysis(blocks, scores, units, pools, metadata, plan)
    csv(output/'sign_null_summary.csv', null); csv(output/'sign_null_draws.csv', null_draws)
    csv(output/'sign_null_maxima.csv', maxima)
    write_new_bytes(output/'sign-null-trajectory-masks.parquet', masks.to_parquet(index=False))
    write_new_bytes(output/'coverage_and_effects.csv', (SOURCE/'reports/coverage_and_effects.csv').read_bytes())
    body = markdown_report(diag, budgets, comparison, bootstrap, null, metadata, audit, plan, receipts)
    write_new_bytes(output/'reports/reward-variants-analysis.md', body.encode())
    expanded = (SOURCE/'reports/phase2-results.md').read_bytes()+b'\n\n---\n\n'+body.encode()
    write_new_bytes(output/'reports/phase2-results-expanded.md', expanded)
    csv(output/'reports/all-primary-scores.csv', diag[(diag.phase == 'all')&(diag.threshold == 0)])
    verify_records(plan['inputs']+plan['preserved_runtime_sources']+plan['analysis_sources']+[plan['setting_document']])
    provenance = {'plan_sha256': file_hash(output/'plan.json'), 'source_audit': audit,
        'preserved_inputs': len(plan['inputs']), 'preserved_runtime_sources': len(plan['preserved_runtime_sources']),
        'all_preserved_hashes_match': True, 'same_candidate_pools': pools,
        'scientific_status': 'ANALYZED_exploratory_single_seed',
        'files': [{'path': str(p.relative_to(output)), 'sha256': file_hash(p)}
            for p in sorted(output.rglob('*')) if p.is_file()]}
    write_new_json(output/'provenance.json', provenance)
    write_new_json(output/'complete.json', {'status': 'complete', 'elapsed_seconds': time.time()-started,
        'seed': 404, 'reward_formulas': len(RECIPES), 'reward_scores': len(RECIPES)*len(AGGREGATIONS),
        'all_score_columns': len(metadata), 'provenance_sha256': file_hash(output/'provenance.json'),
        'new_training': False, 'new_environment_rollouts': 0, 'new_model_forward': False,
        'old_reports_preserved': True, 'input_hashes_unchanged': True})
    progress('COMPLETE', seconds=time.time()-started, report=str(output/'reports/phase2-results-expanded.md'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
    else:
        try:
            run(args.output)
        except Exception as error:
            if (args.output/'run-intent.json').exists() and not (args.output/'complete.json').exists():
                write_new_json(args.output/'failed.json', {'error': repr(error), 'time_unix': time.time(), 'automatic_retry': False})
            raise


if __name__ == '__main__':
    main()
