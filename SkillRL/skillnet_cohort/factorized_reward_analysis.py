"""No-clobber, CPU-only seed404 validation of one factorized reward recipe."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import REPO, file_hash, read_json, write_new_bytes, write_new_json
from .factorized_reward import VERSION, GROUP_KEYS, aggregate_scores, registry, sign_null_group
from .reward_variants import AGGREGATIONS, EPS, aggregation_weights, validate_tokens
from .reward_variant_analysis import (COHORT, SOURCE, SIGN_METRICS, check_origin, csv,
    interval, point_tables, pool_data, progress, verify_records)
from .reward_variant_statistics import metric_arrays
from .realized_reward_analysis import (direction_table, independent_metrics, read_csv,
    sealed_artifacts, verify_old_metrics)

PRIOR = COHORT/'realized-reward-s404-v2'
DRAWS = COHORT/'reward-variants-s404-v2'
DEFAULT_OUTPUT = COHORT/'factorized-reward-s404-v1'
DOC = REPO/'docs/experiments/phase12-independent-v4/FACTORIZED-REWARD-SEED404-20260922-v1.md'
NEW_FILES = [Path(__file__), REPO/'skillnet_cohort/factorized_reward.py',
             REPO/'tests/skillnet_cohort/test_factorized_reward.py', DOC]
BALANCED = 'balanced_accuracy_abstention_wrong'


def record(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_hash(path)}


def deduplicate(records):
    result = {}
    for item in records:
        if item['path'] in result and result[item['path']] != item:
            raise ValueError('Conflicting historical hashes: '+item['path'])
        result[item['path']] = item
    return list(result.values())


def scope(output):
    output = Path(output).resolve()
    if output.parent != COHORT or not output.name.startswith('factorized-reward-s404-'):
        raise PermissionError('Only a new factorized-reward seed404 directory is authorized')
    return output


def prepare(output, tests):
    output = scope(output)
    if output.exists():
        raise FileExistsError('A NEW output directory is required')
    suites = list(ET.parse(tests).getroot().iter('testsuite'))
    if (not suites or sum(int(s.get('tests', 0)) for s in suites) < 20
            or any(int(s.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped'))):
        raise ValueError('At least20 passing synthetic tests and no skips required')
    check_origin()
    complete = read_json(PRIOR/'complete.json')
    if complete['status'] != 'complete' or complete['provenance_sha256'] != file_hash(PRIOR/'provenance.json'):
        raise ValueError('Prior analysis is incomplete or changed')
    prior_plan = read_json(PRIOR/'plan.json')
    inputs = [record(PRIOR/n) for n in ('complete.json', 'provenance.json')]
    inputs += [{'path': str(PRIOR/r['path']), 'sha256': r['sha256']}
               for r in read_json(PRIOR/'provenance.json')['files']]
    # Bind original committed scalars/gold AND the exact historical resampling.
    inputs += [record(SOURCE/n) for n in ('token_signals.parquet', 'ranking-snapshots.json',
        'reused-labels/utility_units.parquet', 'reports/coverage_and_effects.csv')]
    inputs += [record(DRAWS/n) for n in ('bootstrap-label-draws-placebo.parquet',
        'bootstrap-label-draws-null.parquet', 'sign-null-trajectory-masks.parquet')]
    inputs += [record(tests)]
    sources = deduplicate(prior_plan['sources']+[record(p) for p in NEW_FILES])
    inputs = deduplicate(inputs)
    verify_records(inputs+sources)
    plan = {'version': VERSION, 'approved': True, 'seed': 404, 'output': str(output),
        'source': str(SOURCE), 'prior': str(PRIOR), 'draws': str(DRAWS),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'authority': '用户：按照上述中心化幅度乘reward定向因子方案开始验证',
        'new_training': False, 'new_model_forward': False, 'new_environment_rollouts': 0,
        'new_api_calls': 0, 'other_seeds': [], 'automatic_retry': False,
        'hard_timeout_seconds': None, 'historical_labels_seen': True,
        'scientific_status': 'exploratory_single_seed_not_confirmatory',
        'primary': {'control': 'placebo', 'phase': 'all', 'aggregation': 'token',
                    'threshold': 0., 'score': 'D_factor::token::reward'},
        'event_thresholds': [0., .05], 'bootstrap_draws': 2000, 'sign_null_draws': 512,
        'epsilon': EPS, 'registry': registry(), 'inputs': inputs, 'sources': sources,
        'preserved_prior_source_count': len(prior_plan['sources']),
        'expected_token_control_rows': 155638, 'expected_decisions': 5511,
        'command': [sys.executable, '-B', '-m', 'skillnet_cohort.factorized_reward_analysis',
                    'run', '--output', str(output)],
        'environment': {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                        'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
        'working_directory': str(REPO), 'original_reports_overwritten': False}
    write_new_json(output/'plan.json', plan)
    csv(output/'new-registry.csv', pd.DataFrame(registry()))
    for path in NEW_FILES:
        write_new_bytes(output/'source-snapshot'/path.relative_to(REPO), path.read_bytes())
    progress('PREPARED_NOT_SCORED', output=str(output), new_columns=len(registry()),
             plan_sha256=file_hash(output/'plan.json'))


def binding(output):
    output = scope(output); plan = read_json(output/'plan.json')
    if (plan['version'] != VERSION or plan['registry'] != registry()
            or plan['output'] != str(output) or plan['seed'] != 404
            or not plan['approved'] or plan['new_training'] or plan['new_model_forward']
            or plan['new_environment_rollouts'] or plan['new_api_calls'] or plan['other_seeds']
            or plan['source'] != str(SOURCE) or plan['prior'] != str(PRIOR)
            or plan['draws'] != str(DRAWS) or plan['automatic_retry']):
        raise PermissionError('Changed formula or experiment scope')
    verify_records(plan['inputs']+plan['sources'])
    return plan


def metric_arrays_extended(scores, targets, threshold):
    result = metric_arrays(scores, targets, threshold)
    positive, negative = targets > threshold+EPS, targets < -threshold-EPS
    npos, nneg = positive.sum(1), negative.sum(1)
    correct_pos = positive.astype(float) @ (scores > EPS).astype(float).T
    correct_neg = negative.astype(float) @ (scores < -EPS).astype(float).T
    result[BALANCED] = .5*(
        np.divide(correct_pos, npos[:, None], out=np.full(correct_pos.shape, np.nan), where=npos[:, None] > 0)
        + np.divide(correct_neg, nneg[:, None], out=np.full(correct_neg.shape, np.nan), where=nneg[:, None] > 0))
    return result


def audit_factors(tokens, components, old_scores):
    """Separate fsum implementation, plus identity checks against archived B/b."""
    keys = ['control', 'context_id', 'phase', 'skill_id', 'score']
    lookup = old_scores.set_index(keys).value
    maxima = {'B': 0., 'R': 0., 'D_factor': 0., 'D_orientation': 0.,
              'D_factor_A1': 0., 'D_orientation_A1': 0.}
    baseline_max = 0.
    for row in components.to_dict('records'):
        g = tokens[(tokens.control == row['control']) & (tokens.context_id == row['context_id'])
                   & (tokens.skill_id == row['skill_id'])]
        if row['phase'] != 'all':
            g = g[g.phase == row['phase']]
        w = aggregation_weights(g, row['aggregation'])
        b = g.advantage.to_numpy(float)*g.chosen_delta.to_numpy(float)
        B = math.fsum(float(x*y) for x, y in zip(w, g.delta_centered_norm))
        numerator = math.fsum(float(x*y) for x, y in zip(w, b))
        denominator = math.fsum(float(x*abs(y)) for x, y in zip(w, b))+EPS
        r = numerator/denominator
        numerator1 = math.fsum(float(x*y) for x, y in zip(w, g.chosen_delta))
        denominator1 = math.fsum(float(x*abs(y)) for x, y in zip(w, g.chosen_delta))+EPS
        r1 = numerator1/denominator1
        expected = {'B': B, 'R': r, 'D_factor': -B*r, 'D_orientation': -r,
                    'D_factor_A1': -B*r1, 'D_orientation_A1': -r1}
        for key, value in expected.items():
            maxima[key] = max(maxima[key], abs(value-row[key]))
            if not np.isclose(row[key], value, rtol=1e-12, atol=1e-12):
                raise ValueError('Independent factor arithmetic mismatch: '+key)
        index = tuple(row[k] for k in keys[:-1])
        for name, mode, value in (('M_delta_centered', 'magnitude', row['B']),
                                  ('D_action_adv', 'reward', -row['b_mean'])):
            old = lookup.loc[index+(f'{name}::{row["aggregation"]}::{mode}',)]
            baseline_max = max(baseline_max, abs(old-value))
            if not np.isclose(old, value, rtol=1e-12, atol=1e-12):
                raise ValueError('The existing magnitude or direction numerator changed')
    if not components.factor_sign_matches_action_adv.all():
        raise ValueError('Mathematical direction sign identity violated')
    return {'independent_fsum_component_rows': len(components), 'max_absolute_error': maxima,
            'existing_B_and_action_adv_max_absolute_error': baseline_max,
            'factor_sign_identity_passed': True,
            'epsilon_abstention_differences': int((~components.epsilon_abstention_matches_action_adv).sum())}


def paired_analysis(scores, units, pools, metadata, plan):
    rows, comparisons = [], []
    names = [m['score'] for m in metadata]; methods = {m['score']: m for m in metadata}
    for control in ('placebo', 'null'):
        pool, = [p for p in pools[control+'/stable_raw'] if p['phase'] == 'all']
        matrix, target, _ = pool_data(scores, units, pool, control, metadata)
        draws = pd.read_parquet(DRAWS/f'bootstrap-label-draws-{control}.parquet')
        if list(draws.columns) != pool['shared_skill_ids'] or len(draws) != plan['bootstrap_draws']:
            raise ValueError('Changed paired bootstrap pool/draws')
        good = np.isfinite(draws.to_numpy()).all(1)
        for threshold in plan['event_thresholds']:
            arrays = metric_arrays_extended(matrix, draws.to_numpy()[good], threshold)
            point = metric_arrays_extended(matrix, target[None], threshold)
            for method in registry():
                i = names.index(method['score'])
                base = {'control': control, 'phase': 'all', 'threshold': threshold, 'score': method['score']}
                for key, values in arrays.items():
                    if key in SIGN_METRICS+[BALANCED] and not method['signed']:
                        continue
                    rows.append({**base, 'metric': key, 'point': float(point[key][0, i]),
                                 **interval(values[:, i]), 'complete_pool_draws': int(good.sum()),
                                 'missing_any_skill_draws': int((~good).sum())})
                if method['name'] not in ('D_factor', 'D_orientation'):
                    continue
                agg = method['aggregation']
                refs = [f'M_delta_centered::{agg}::magnitude', f'D_action_adv::{agg}::reward',
                        f'{method["name"]}_A1::{agg}::reward_free']
                if method['name'] == 'D_factor':
                    refs += [f'D_orientation::{agg}::reward', f'D_original::{agg}::reward',
                             f'D_signed::{agg}::reward', f'C_centered::{agg}::reward',
                             f'D_policy_action_adv::{agg}::reward', f'D_real::{agg}::reward']
                for ref in refs:
                    j = names.index(ref)
                    for key in arrays:
                        if key in SIGN_METRICS+[BALANCED] and not methods[ref]['signed']:
                            continue
                        diff = arrays[key][:, i]-arrays[key][:, j]
                        valid = np.isfinite(diff)
                        comparisons.append({**base, 'reference': ref, 'metric': key,
                            'point_difference': float(point[key][0, i]-point[key][0, j]), **interval(diff),
                            'fraction_positive': float(np.mean(diff[valid] > 0)) if valid.any() else np.nan,
                            'not_a_confirmatory_p_value': True})
        progress('PAIRED_LABEL_UNCERTAINTY', control=control, complete_draws=int(good.sum()),
                 incomplete_pool_draws=int((~good).sum()))
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


def null_analysis(tokens, scores, units, pools, plan):
    masks = pd.read_parquet(DRAWS/'sign-null-trajectory-masks.parquet')
    if masks.shape != (512, 128) or set(masks.columns) != set(tokens.trajectory_id):
        raise ValueError('Changed whole-trajectory sign masks')
    methods = [m for m in registry() if m['signed']]
    summaries, draws = [], []
    for control in ('placebo', 'null'):
        pool, = [p for p in pools[control+'/stable_raw'] if p['phase'] == 'all']
        matrix, target, _ = pool_data(scores, units, pool, control, methods)
        cube = np.empty((len(masks), len(methods), len(target)))
        for si, skill in enumerate(pool['shared_skill_ids']):
            g = tokens[(tokens.control == control)&(tokens.context_id == pool['context_id'])&(tokens.skill_id == skill)]
            for ai, agg in enumerate(AGGREGATIONS):
                cube[:, 2*ai:2*ai+2, si] = sign_null_group(g, agg, list(masks.columns), masks.to_numpy())
        for threshold in plan['event_thresholds']:
            point = metric_arrays_extended(matrix, target[None], threshold)
            simulated = metric_arrays_extended(cube.reshape(-1, len(target)), target[None], threshold)
            for key in ('auroc_decline_vs_increase', 'spearman', 'average_precision', BALANCED):
                values = simulated[key][0].reshape(len(masks), len(methods))
                for i, m in enumerate(methods):
                    same_name = [j for j, other in enumerate(methods) if other['name'] == m['name']]
                    maxima = np.max(values[:, same_name], axis=1)
                    v = values[:, i]; valid = np.isfinite(v); valid_max = np.isfinite(maxima)
                    observed = float(point[key][0, i])
                    summaries.append({'control': control, 'phase': 'all', 'threshold': threshold,
                        'score': m['score'], 'metric': key, 'observed': observed, **interval(v),
                        'reference_fraction_ge_observed': float(np.mean(v[valid] >= observed-EPS)) if valid.any() and np.isfinite(observed) else np.nan,
                        'three_aggregation_max_fraction_ge_observed': float(np.mean(maxima[valid_max] >= observed-EPS)) if valid_max.any() and np.isfinite(observed) else np.nan,
                        'not_a_confirmatory_p_value': True, 'historical_search_not_adjusted': True})
                    draws.extend({'control': control, 'threshold': threshold, 'score': m['score'],
                                  'metric': key, 'draw': d, 'value': float(value)} for d, value in enumerate(v))
        progress('TRAJECTORY_BLOCK_SIGN_NULL', control=control, draws=len(masks))
    return pd.DataFrame(summaries), pd.DataFrame(draws)


def verify_direction(scores, units, pools, metadata, table):
    """Independent scalar confusion counting, including unsigned exclusion."""
    checks = 0
    for control in ('placebo', 'null'):
        for pool in pools[control+'/stable_raw']:
            if not pool['shared_skill_ids']:
                continue
            matrix, target, _ = pool_data(scores, units, pool, control, metadata)
            for threshold in (0., .05):
                q = table[(table.control == control)&(table.context_id == pool['context_id'])
                          &(table.phase == pool['phase'])&(table.threshold == threshold)].set_index('score')
                for method, values in zip(metadata, matrix):
                    r = q.loc[method['score']]
                    if not method['signed']:
                        if not pd.isna(r[BALANCED]):
                            raise ValueError('Unsigned score received a direction interpretation')
                        continue
                    pos = [(x, y) for x, y in zip(values, target) if y > threshold+EPS]
                    neg = [(x, y) for x, y in zip(values, target) if y < -threshold-EPS]
                    cp = sum(x > EPS for x, _ in pos); cn = sum(x < -EPS for x, _ in neg)
                    balanced = .5*(cp/len(pos)+cn/len(neg)) if pos and neg else np.nan
                    if cp != r.correct_declines or cn != r.correct_increases or not np.isclose(balanced, r[BALANCED], equal_nan=True):
                        raise ValueError('Independent direction-confusion mismatch')
                    checks += 1
    return {'independent_direction_rows': checks, 'passed': True}


def magnitude_diagnostic(scores, units, pools, metadata):
    rows = []
    for control in ('placebo', 'null'):
        pool, = [p for p in pools[control+'/stable_raw'] if p['phase'] == 'all']
        for agg in AGGREGATIONS:
            sid = f'M_delta_centered::{agg}::magnitude'
            matrix, target, _ = pool_data(scores, units, pool, control, [m for m in metadata if m['score'] == sid])
            rows.append({'control': control, 'aggregation': agg, 'skills': len(target),
                         'rho_B_vs_absolute_delta_utility': spearmanr(matrix[0], abs(target)).statistic,
                         'rho_B_vs_signed_decline': spearmanr(matrix[0], target).statistic})
    return pd.DataFrame(rows)


def report(diag, direction, paired, null, audit, magnitude, elapsed):
    chosen = ['M_delta_centered', 'D_orientation', 'D_factor', 'D_factor_A1',
              'D_orientation_A1', 'D_action_adv', 'D_signed', 'D_original', 'C_centered', 'D_real']
    q = diag[(diag.phase == 'all')&(diag.threshold == 0)&diag.name.isin(chosen)
             &~diag['mode'].eq('unsigned')].copy()
    q = q.merge(direction[['control', 'context_id', 'phase', 'threshold', 'score', BALANCED]],
                on=['control', 'context_id', 'phase', 'threshold', 'score'], validate='one_to_one')
    columns = ['name', 'aggregation', 'candidates', 'declines', 'increases',
               'auroc_decline_vs_increase', 'spearman', 'average_precision',
               'sign_accuracy_called', BALANCED]
    primary = q[(q.control == 'placebo')&(q.aggregation == 'token')]
    gains = paired[(paired.control == 'placebo')&(paired.threshold == 0)
                   &(paired.score == 'D_factor::token::reward')
                   &paired.metric.isin(['auroc_decline_vs_increase', 'spearman', BALANCED])]
    negative = null[(null.control == 'placebo')&(null.threshold == 0)
                    &null.score.isin(['D_factor::token::reward', 'D_orientation::token::reward'])]
    risks = [
        ('Simpson / 分层反转', 'CAUTION', '保留全部阶段、control、聚合；不以单个有利分层替代主分析，稀疏分层不做总体外推。'),
        ('Ecological / 层级外推', 'CAUTION', '分析单位为skill-window，token不是独立验证样本；不能推出逐token因果credit。'),
        ('Berkson / 支持集选择', 'CAUTION', '固定18技能共同池；37技能coverage另存，不外推未自然支持的技能。'),
        ('Collider / 条件选择', 'CAUTION', '方向AUC只含非零点标签；另报包含零标签的AP/AUROC，不按结果加筛选。'),
        ('Base-rate neglect', 'CHECKED', '列出下降/上升数量、恒预测类别基线和平衡准确率。'),
        ('Regression to mean', 'CAUTION', '不按极端效用或新score筛选样本；配对标签仍存在抽样误差。'),
        ('Survivorship', 'CAUTION', '零A、失败轨迹及不支持技能不隐去；bootstrap缺任何技能则整池NA并报告。'),
        ('Look-elsewhere', 'CAUTION', '历史273列已看，新组合仍是事后探索；无确认性p值。'),
        ('Forking paths', 'CAUTION', '本次运行前冻结一个组合、两种对照分解；不改变符号或择优聚合。'),
        ('Correlation vs causation', 'CAUTION', 'R是reward交互关联，不是动作真实贡献；B不是校准后的效用变化大小。'),
        ('Reverse causality / 标签泄漏', 'CAUTION', '新分数函数不接收gold且先commit；历史标签已知，不能宣称前瞻验证。'),
    ]
    return ('# Seed404：中心化幅度 × reward 定向因子验证\n\n'
        '## Material Passport\n\n'
        '- Origin Skill: academic-research-suite / experiment-agent\n'
        '- Origin Mode: run / validate\n- Verification Status: ANALYZED（单seed事后探索）\n'
        '- Version Label: factorized_centered_reward_v1\n\n'
        '## 1. 冻结公式和范围\n\n'
        '`b=A*delta(a); B=Agg(||Hdelta||); R=Agg(b)/(Agg(abs(b))+1e-12); D_factor=-B*R`。'
        '`-R`单独列出；A1在全部原token上置A=1，不继承零A屏蔽。主口径PLACEBO/all/token/0pp。'
        '没有新训练、模型前向、效用rollout或API；505/606未启动；旧报告不覆盖。\n\n'
        'R只是正负证据净比例，不是置信度；B>0时新D符号与D_action_adv相同。'
        '重排不等于改善方向判别。范数预测的是策略响应变化量，不能直接认定已预测效用变化绝对值。\n\n'
        '## 2. 主口径结果\n\n'+primary[columns].to_markdown(index=False, floatfmt='.6f')+'\n\n'
        '无符号基线不解释为正负方向；方向AUC仅下降vs上升，AP包含全部固定技能池。\n\n'
        '## 3. 主口径配对差值与95%描述区间\n\n'
        +gains[['reference', 'metric', 'point_difference', 'low', 'high', 'valid_draws']].to_markdown(index=False, floatfmt='.6f')+'\n\n'
        '复用2000组game×continuation抽样；完整/缺技能draw数见bootstrap_summary.csv。部分指标还因无类别而NA。'
        '仅涵盖固定训练读出下的gold-label不确定性，不包含训练seed/读出抽样方差、不校正历史探索。\n\n'
        '## 4. 轨迹块reward符号对照\n\n'
        +negative[['score', 'metric', 'observed', 'reference_fraction_ge_observed',
                   'three_aggregation_max_fraction_ge_observed']].to_markdown(index=False, floatfmt='.6f')+'\n\n'
        '512组共同轨迹块翻号，保留A幅度与局内A变化。参考比例不是校准p值，不校正此前多公式搜索。\n\n'
        '## 5. 完整敏感性（不择优替代主分析）\n\n'
        +q[['control']+columns].to_markdown(index=False, floatfmt='.6f')+'\n\n'
        '阶段/5pp结果、top-k、所有285列见ranking_diagnostics.csv与ranking_budgets.csv。\n\n'
        '## 6. 幅度与效用变化大小的核对\n\n'
        +magnitude.to_markdown(index=False, floatfmt='.6f')+'\n\n'
        '## 7. 工程验收及证据边界\n\n'
        f'全部155638个token×control行、5511个决策复用；新因子{audit["independent_fsum_component_rows"]}个汇总单元独立fsum核验。'
        f'符号恒等通过；epsilon弃权差异{audit["epsilon_abstention_differences"]}个单元。'
        f'本轮CPU统计耗时约{elapsed:.1f}秒（不含开发测试）。\n\n'
        '273个旧指标列和5460条旧分层指标保留并复现；全部新旧指标另用sklearn/scipy、方向混淆手算核验。'
        '这是数值实现交叉核验，不是独立RL重复实验或跨seed可复现性证明；Verification Status保持ANALYZED。\n\n'
        '## 8. 统计解释风险检查\n\nCoverage: 11/11 checked（不是11项显著性检验）。\n\n'
        +pd.DataFrame(risks, columns=['risk', 'status', 'boundary']).to_markdown(index=False)+'\n')


def run(output):
    output = scope(output)
    if (output/'run-intent.json').exists():
        raise FileExistsError('No automatic retry or rerun of an attempted analysis')
    plan = binding(output)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise PermissionError('Explicit CPU-only environment required')
    started = time.time()
    write_new_json(output/'run-intent.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'started_unix': started, 'pid': os.getpid(), 'python': platform.python_version(),
        'plan_sha256': file_hash(output/'plan.json'), 'gpu_used': False})
    progress('STARTED_CPU_ONLY', pid=os.getpid(), output=str(output))
    try:
        tokens = pd.read_parquet(SOURCE/'token_signals.parquet'); validate_tokens(tokens)
        if len(tokens) != plan['expected_token_control_rows'] or tokens.decision_id.nunique() != plan['expected_decisions']:
            raise ValueError('Actual-token cohort differs from the fixed scope')
        new_scores, components = aggregate_scores(tokens)
        old_scores = read_csv(PRIOR/'skill_scores.csv')
        audit = audit_factors(tokens, components, old_scores)
        scores = pd.concat([old_scores, new_scores], ignore_index=True)
        metadata = read_csv(PRIOR/'registry.csv').to_dict('records')+registry()
        if len(metadata) != 285 or set(scores.score) != {m['score'] for m in metadata}:
            raise ValueError('All old273 plus new12 columns required')
        csv(output/'skill_scores.csv', scores); csv(output/'factor-components.csv', components)
        csv(output/'registry.csv', pd.DataFrame(metadata))
        write_new_json(output/'score-commit.json', {'scores_sha256': file_hash(output/'skill_scores.csv'),
            'components_sha256': file_hash(output/'factor-components.csv'), 'created_unix': time.time(),
            'new_scoring_received_gold': False, 'historical_labels_seen': True, 'not_prospective': True})
        progress('SCORES_COMMITTED', columns=len(metadata), component_rows=len(components))
        units = pd.read_parquet(SOURCE/'reused-labels/utility_units.parquet')
        pools = read_json(SOURCE/'ranking-snapshots.json')['candidate_pools']
        diag, budgets = point_tables(scores, units, pools, metadata, plan['event_thresholds'])
        directions = direction_table(scores, units, pools, metadata, plan['event_thresholds'])
        oldcheck = verify_old_metrics(diag, PRIOR)
        independent = independent_metrics(scores, units, pools, metadata, diag)
        directioncheck = verify_direction(scores, units, pools, metadata, directions)
        csv(output/'ranking_diagnostics.csv', diag); csv(output/'ranking_budgets.csv', budgets)
        csv(output/'direction-confusions.csv', directions)
        csv(output/'skill_scores_and_gold.csv', scores.merge(units,
            on=['control', 'context_id', 'skill_id', 'phase'], how='left', validate='many_to_one'))
        progress('POINT_METRICS_INDEPENDENTLY_CHECKED', rows=len(diag))
        summary, paired = paired_analysis(scores, units, pools, metadata, plan)
        csv(output/'bootstrap_summary.csv', summary); csv(output/'paired_direction_gains.csv', paired)
        null, draws = null_analysis(tokens, scores, units, pools, plan)
        csv(output/'sign_null_summary.csv', null); csv(output/'sign_null_draws.csv', draws)
        magnitude = magnitude_diagnostic(scores, units, pools, metadata)
        csv(output/'magnitude-absolute-utility.csv', magnitude)
        write_new_bytes(output/'coverage_and_effects.csv', (SOURCE/'reports/coverage_and_effects.csv').read_bytes())
        body = report(diag, directions, paired, null, audit, magnitude, time.time()-started)
        write_new_bytes(output/'reports/factorized-reward-analysis.md', body.encode())
        write_new_bytes(output/'reports/phase2-results-expanded.md',
            (PRIOR/'reports/phase2-results-expanded.md').read_bytes()+b'\n\n---\n\n'+body.encode())
        binding(output)
        write_new_json(output/'independent-verification.json', {**audit, **oldcheck,
            'ranking_metrics': independent, 'direction_metrics': directioncheck,
            'all_bound_sources_and_inputs_unchanged': True, 'score_columns': len(metadata)})
        write_new_json(output/'provenance.json', {'plan_sha256': file_hash(output/'plan.json'),
            'scientific_status': 'ANALYZED_exploratory_single_seed',
            'files': [{'path': str(p.relative_to(output)), 'sha256': file_hash(p)} for p in sealed_artifacts(output)]})
        write_new_json(output/'complete.json', {'status': 'complete', 'seed': 404,
            'finished_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.time()-started,
            'provenance_sha256': file_hash(output/'provenance.json'), 'new_training': False,
            'new_model_forward': False, 'new_environment_rollouts': 0, 'new_api_calls': 0,
            'other_seeds': [], 'old_reports_preserved': True, 'score_columns': len(metadata),
            'metric_rows': len(diag), 'preserved_prior_source_count': plan['preserved_prior_source_count']})
        progress('COMPLETE', seconds=time.time()-started, report=str(output/'reports/factorized-reward-analysis.md'))
    except Exception as error:
        write_new_json(output/'failed.json', {'error': repr(error), 'utc': datetime.now(timezone.utc).isoformat(),
                                             'automatic_retry': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--tests', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if args.tests is None:
            parser.error('--tests is required before preparation')
        prepare(args.output, args.tests)
    else:
        run(args.output)


if __name__ == '__main__':
    main()
