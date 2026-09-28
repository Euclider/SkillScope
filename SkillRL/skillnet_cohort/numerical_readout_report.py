"""Matched-pool, immutable legacy/stable/centered numerical audit reports."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .common import file_hash, read_json, write_new_bytes, write_new_json
from .numerical_readout import binding, SOURCES, FEATURE_KEYS
from .reports import passport, ranking_diagnostics
from phase2.ranking import evaluate_snapshot, score_snapshot

VARIANTS = ('legacy_recorded', 'stable_raw', 'stable_centered_gate')
ID_KEYS = ['skill_id', 'context_id', 'phase']


def variant_features(legacy, stable, variant):
    if variant not in VARIANTS:
        raise ValueError('No unregistered variant selection')
    frame = (legacy if variant == 'legacy_recorded' else stable).copy()
    if variant == 'stable_centered_gate':
        frame['C_upd'] = frame['C_upd_centered']
        frame['D_contribution'] = frame['D_centered_contribution']
        frame['gate_coverage'] = frame['gate_centered_coverage']
        # P is intentionally the SAME stable projection, not a new score.
    return frame


def attach_old_margin(features, margins, control, phases):
    evidence = margins[(margins['update'] == 0) & (margins.purpose == 'evidence')]
    rows = []
    for (skill, context), group in evidence.groupby(ID_KEYS[:2]):
        for phase in phases:
            q = group if phase == 'all' else group[group.phase == phase]
            values = q.groupby('game_id')['M_'+control].mean()
            rows.append({'skill_id': skill, 'context_id': context, 'phase': phase,
                         'old_margin': float(values.mean()) if len(values) else np.nan})
    return features.merge(pd.DataFrame(rows, columns=ID_KEYS+['old_margin']),
                          on=ID_KEYS, how='left', validate='one_to_one')


def comparison_tables(legacy, stable, units, margins, config):
    diagnostics, budgets, scores, snapshots, pools_all = [], [], [], {}, {}
    phases = config.get('prediction', {}).get('phases', ['all'])
    for control in ('placebo', 'null'):
        expected_pool = None
        for variant in VARIANTS:
            frame = variant_features(legacy, stable, variant)
            frame = frame[(frame.control == control) & frame.phase.isin(phases)]
            frame = attach_old_margin(frame, margins, control, phases)
            snapshot = score_snapshot(frame, config['ranking'])
            metrics, pools, joined = evaluate_snapshot(snapshot, units[units.control == control], config['ranking'])
            canonical = [(p['context_id'], p['phase'], sorted(p['shared_skill_ids'])) for p in pools]
            if expected_pool is not None and canonical != expected_pool:
                raise ValueError('Variants must use the identical skill candidate pool')
            expected_pool = canonical
            diagnosis = ranking_diagnostics(joined, {'candidate_pools': pools}, config['ranking'])
            diagnostics.append(diagnosis.assign(variant=variant, control=control))
            budgets.append(metrics.assign(variant=variant, control=control))
            scores.append(joined.assign(variant=variant, control=control))
            snapshots[control+'/'+variant] = snapshot
            pools_all[control+'/'+variant] = pools
    return (pd.concat(diagnostics, ignore_index=True), pd.concat(budgets, ignore_index=True),
            pd.concat(scores, ignore_index=True), snapshots, pools_all)


def report(path, seed):
    plan = binding(path, seed); root = SOURCES[seed]
    output = Path(plan['root'])/f'seed-{seed}'; reports = output/'reports'
    if (output/'complete.json').exists():
        raise FileExistsError('No duplicate report publication')
    commit = read_json(output/'committed.json')
    if (commit['features_sha256'] != file_hash(output/'skill_context_features.parquet')
            or commit['token_signals_sha256'] != file_hash(output/'token_signals.parquet')
            or commit['all_legacy_scalar_signals_exact'] is not True):
        raise ValueError('Corrected scores are not completely audited')
    completion = read_json(root/'analysis-complete.json')
    if completion['status'] != 'complete' or completion['continuations'] != 7272:
        raise ValueError('Wait for complete retained all-first-call utility labels')
    config = read_json(root/'protocol.json')
    legacy = pd.read_parquet(root/'window_signals/u0000-u0005/skill_context_features.parquet')
    stable = pd.read_parquet(output/'skill_context_features.parquet')
    units_path, margins_path = root/'window_metrics/utility_units.parquet', root/'window_metrics/anchor_margins.parquet'
    units, margins = pd.read_parquet(units_path), pd.read_parquet(margins_path)
    diagnostics, budgets, scores, snapshots, pools = comparison_tables(legacy, stable, units, margins, config)
    # Check the newly generated legacy report against the original report as
    # well as checking token arithmetic; this detects changed reporting pools.
    old_diagnostics = pd.read_csv(root/'reports/ranking_diagnostics.csv')
    keys = ['context_id', 'phase', 'score', 'threshold']
    left = old_diagnostics.sort_values(keys).reset_index(drop=True)
    right = diagnostics[(diagnostics.control == 'placebo') & (diagnostics.variant == 'legacy_recorded')].sort_values(keys).reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError('Original ranking population changed')
    for col in ('candidates', 'declines', 'average_precision', 'auroc_decline_vs_rest', 'spearman', 'kendall'):
        if not np.allclose(left[col], right[col], rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError('Original ranking report was not reproduced: '+col)
    write_new_json(output/'ranking-snapshots.json', {'snapshots': snapshots, 'candidate_pools': pools,
        'readout_commit_sha256': file_hash(output/'committed.json'),
        'scope': 'posthoc comparison using frozen U0 baseline formulas; not a new prospective preregistration'})
    for name, frame in (('ranking_diagnostics', diagnostics), ('ranking_budgets', budgets), ('scores_and_gold', scores)):
        write_new_bytes(reports/(name+'.csv'), frame.assign(seed=seed).to_csv(index=False).encode())
    for name in ('utility_units.parquet', 'utility_games.parquet', 'anchor_margins.parquet'):
        # Reuse exact compact labels. No bootstrap or environment resampling.
        write_new_bytes(output/'reused-labels'/name, (root/'window_metrics'/name).read_bytes())
    for name in ('performance.csv',):
        write_new_bytes(reports/name, (root/'reports'/name).read_bytes())
    coverage = pd.read_csv(root/'support/coverage.csv')
    complete = coverage.merge(stable[stable.control == 'placebo'][ID_KEYS+[
        'supported', 'P_int', 'P_int_centered', 'C_upd', 'C_upd_centered', 'D_contribution',
        'D_centered_contribution', 'direction_coverage', 'gate_coverage', 'gate_centered_coverage']],
        on=ID_KEYS, how='left', validate='one_to_one').merge(
        units[units.control == 'placebo'].drop(columns=['quantity_filter_applied']),
        on=ID_KEYS, how='left', validate='one_to_one')
    write_new_bytes(reports/'coverage_and_effects.csv', complete.to_csv(index=False).encode())
    boundary = ('这是已见旧结果后获批的数值实现修正，不追溯改写为新的预登记实验。'
        '同一训练批次、U0/U5、动作token、advantage、完整37技能库、router、首调用锚点、'
        '阈值、候选池与聚合权重保持。模型前向仍为原BF16；只对读出算术使用FP64。'
        '旧版源码/报告和所有RL/效用轨迹保留；没有重新采样RL、性能或O/P/N效用。\n\n'
        'legacy_recorded为原报告；stable_raw为规范化OLD概率和稳定零和投影的FP64修正；'
        'stable_centered_gate使用同一稳定P，只将C及D门控切换为中心化量。'
        'P_centered仅作恒等式检查，不当作新预测器。原版在同次前向上的所有标量读出逐值一致，'
        '原版PLACEBO排序报告也已复现；因此不是把后端前向变化误称为数值改进。'
        'KL/JS及activation基线保留原定义，不拟合阈值或挑选赢家。\n\n'
        '所有自然出现技能与全部首次调用锚点保留；未出现/没有训练读出的NA仍明确记录。'
        '有效方向比例改变可能来自数值纠错，不能解释为新轨迹支持。'
        '点估计、配对game/continuation区间和历史U0权重精度/恢复边界全部继承原报告。\n\n')
    performance = pd.read_csv(reports/'performance.csv')
    p1 = passport('numerical_readout_v1')+f'# Phase1：seed {seed}，效用标签复用\n\n'+boundary
    p1 += performance[performance.task == 'all'].to_markdown(index=False)+'\n\n'
    q = complete[(complete.phase == 'all') & complete.utility_evaluable]
    cols = ['skill_id', 'source_calls', 'source_games', 'train_decisions', 'anchor_count',
            'utility_old', 'utility_new', 'delta_utility', 'ci_low', 'ci_high', 'interval_status']
    p1 += q[cols].to_markdown(index=False)+'\n'
    p2 = passport('numerical_readout_v1')+f'# Phase2：seed {seed}，原版与稳定数值实现对照\n\n'+boundary
    p2 += ('## 数值一致性\n\n'+f"token×control行数：{commit['tokens_with_controls']}；"
        f"P中心化最大绝对误差：{commit['max_P_centering_error']:.6g}；"
        f"原始/中心化稳定门控差异：{commit['raw_centered_gate_changes']}；"
        f"稳定版相对旧版门控差异：{commit['legacy_gate_changes']}；"
        f"有效方向差异：{commit['legacy_direction_valid_changes']}。\n\n")
    primary = diagnostics[(diagnostics.phase == 'all') & (diagnostics.threshold == 0)
        & diagnostics.score.isin(['C_upd', 'P_int', 'D_contribution', 'D_ungated_contribution'])]
    p2 += '## 相同候选池的下降排序（0pp点估计阈值）\n\n'
    p2 += primary[['control', 'variant', 'score', 'candidates', 'declines', 'average_precision',
                   'auroc_decline_vs_rest', 'spearman']].to_markdown(index=False)+'\n\n'
    p2 += ('NULL/PLACEBO、所有阶段、0/5pp阈值、全部基线及同池预算见CSV。'
        '差异是描述性数值敏感性，不意味着显著改善；单seed不能证明跨seed泛化。'
        'D是单侧风险，D=0不能判断稳定/改善；P符号准确率与排序相关性分别记录。\n\n'
        '## 统计误用检查\n\n覆盖11/11类（适用性与证据边界）：\n\n'
        '- Simpson：control/phase分层保留，不用合并方向代替分层。\n'
        '- 生态谬误：分析单位为skill窗口，不能下推单token或外推独立seed。\n'
        '- Berkson：自然调用与可计算性限定候选池，完整coverage保留NA。\n'
        '- Collider：无新增按目标结果筛选或协变量控制。\n'
        '- 基率忽略：每组明确candidates与declines，零事件指标NA。\n'
        '- 均值回归：不按极端效用选择技能或宣布改善。\n'
        '- 幸存者偏差：完整效用标签是验收条件，未观测技能仍在coverage。\n'
        '- 多重检验：全部预先登记变体/层次展示，不据此宣布显著赢家。\n'
        '- 分析路径：事后数值修正明确披露，原版与全部固定变体保留。\n'
        '- 相关不等于因果：读出关联不证明技能编辑收益。\n'
        '- 反向因果：读出公式不使用U5效用，但历史标签已可见，不宣称新前瞻研究。\n')
    write_new_bytes(reports/'phase1-results.md', p1.encode())
    write_new_bytes(reports/'phase2-results.md', p2.encode())
    provenance = {'seed': seed, 'plan_sha256': file_hash(path), 'readout_commit_sha256': file_hash(output/'committed.json'),
        'utility_labels_reused_bitwise': True, 'original_ranking_report_reproduced': True,
        'utility_source': str(root), 'original_reports_preserved': True,
        'scientific_status': 'ANALYZED_posthoc_numerical_sensitivity_not_cross_seed_validation',
        'files': [{'path': str(p.relative_to(output)), 'sha256': file_hash(p)}
                  for folder in (reports, output/'reused-labels') for p in sorted(folder.iterdir()) if p.is_file()]}
    write_new_json(output/'report-provenance.json', provenance)
    write_new_json(output/'complete.json', {'status': 'complete', 'seed': seed, 'finished_unix': time.time(),
        'reports': str(reports), 'provenance_sha256': file_hash(output/'report-provenance.json'),
        'all_comparators_present': True, 'original_files_unchanged': True, 'new_environment_rollouts': 0})
    print(primary.to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(404, 505), required=True)
    a = p.parse_args(); report(a.plan, a.seed)


if __name__ == '__main__':
    main()
