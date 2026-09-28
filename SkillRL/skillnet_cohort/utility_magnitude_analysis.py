"""CPU-only, no-clobber prediction audit for updated seed404 utility labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

from .common import file_hash, read_json, write_new_bytes, write_new_json
from .reward_variant_analysis import csv, interval, point_tables, pool_data
from .reward_variant_statistics import EPS, metric_arrays, selection_matrix
from .realized_reward_analysis import read_csv
from .utility_precision import COHORT, NUMERICAL

SOURCE = COHORT / 'utility-precision-s404-v1'
DEFAULT_OUTPUT = COHORT / 'utility-precision-predictivity-s404-v2'
POINT_FIELDS = ('average_precision', 'auroc_decline_vs_rest',
                'auroc_decline_vs_increase', 'spearman', 'kendall')
MAG_FIELDS = ('spearman', 'average_precision', 'auroc_large_change_vs_rest',
              'top_quartile_mean_absolute_delta', 'top_quartile_lift')
DISPLAY = ('D_original::token::reward', 'D_signed::token::reward',
           'D_factor::token::reward', 'D_real::token::reward',
           'C_centered::token::reward', 'M_delta_centered::token::magnitude',
           'M_delta_raw::token::magnitude')
CONTRASTS = (
    ('D_original::token::reward', 'D_original::token::unsigned'),
    ('D_original::token::reward', 'M_delta_centered::token::magnitude'),
    ('D_real::token::reward', 'D_real_absA::token::reward_sign_removed'),
)


def verify_source(source: Path) -> None:
    complete = read_json(source / 'complete.json')
    if complete['status'] != 'complete' or complete['gold_repeats'] != 8:
        raise ValueError('The eight-repeat utility panel is not complete')
    if complete['provenance_sha256'] != file_hash(source / 'provenance.json'):
        raise ValueError('Changed source provenance')
    provenance = read_json(source / 'provenance.json')
    for item in provenance['files']:
        if file_hash(source / item['path']) != item['sha256']:
            raise ValueError('Changed sealed input: ' + item['path'])
    if file_hash(source / 'skill_scores.csv') != file_hash(COHORT / 'factorized-reward-s404-v1/skill_scores.csv'):
        raise ValueError('Frozen predictors changed')


def magnitude_arrays(matrix: np.ndarray, risk: np.ndarray, threshold: float = .05):
    """Fixed scores; the gold target is |delta utility|, not |readout|."""
    matrix = np.asarray(matrix, float)
    risk = np.asarray(risk, float)
    if matrix.ndim != 2 or risk.ndim != 2 or matrix.shape[1] != risk.shape[1]:
        raise ValueError('Matched [method, skill] and [draw, skill] arrays required')
    if not np.isfinite(matrix).all() or not np.isfinite(risk).all():
        raise ValueError('Incomplete score or paired gold panel')
    absolute_change = np.abs(risk)
    metrics = metric_arrays(matrix, absolute_change, threshold)
    n, draws = matrix.shape[1], len(risk)
    k = max(1, int(np.ceil(n * .25)))
    weights = selection_matrix(matrix, k)
    mean_selected = absolute_change @ weights.T / k
    mean_pool = absolute_change.mean(axis=1)
    lift = np.divide(mean_selected, mean_pool[:, None],
                     out=np.full((draws, len(matrix)), np.nan),
                     where=mean_pool[:, None] > EPS)
    return {
        'spearman': metrics['spearman'],
        'average_precision': metrics['average_precision'],
        'auroc_large_change_vs_rest': metrics['auroc_decline_vs_rest'],
        'top_quartile_mean_absolute_delta': mean_selected,
        'top_quartile_lift': lift,
    }


def magnitude_points(scores, units, pools, metadata):
    rows = []
    names = [m['score'] for m in metadata]
    for control in ('placebo', 'null'):
        for pool in pools[control + '/stable_raw']:
            if not pool['shared_skill_ids']:
                continue
            matrix, risk, _ = pool_data(scores, units, pool, control, metadata)
            absolute_change = np.abs(risk)
            event_count = int((absolute_change > .05 + EPS).sum())
            for transform, predictors in (('raw', matrix), ('absolute_readout', np.abs(matrix))):
                arrays = magnitude_arrays(predictors, risk[None, :])
                for i, meta in enumerate(metadata):
                    tau = kendalltau(predictors[i], absolute_change).statistic
                    rows.append({**meta, 'control': control, 'context_id': pool['context_id'],
                        'phase': pool['phase'], 'transform': transform,
                        'candidates': len(risk), 'large_change_gt_5pp': event_count,
                        'top_k': max(1, int(np.ceil(len(risk) * .25))),
                        'mean_absolute_delta': float(absolute_change.mean()),
                        'kendall': float(tau) if np.isfinite(tau) else np.nan,
                        **{field: float(arrays[field][0, i]) for field in MAG_FIELDS}})
    return pd.DataFrame(rows)


def magnitude_bootstrap(scores, units, pools, metadata, source):
    rows = []
    for control in ('placebo', 'null'):
        pool, = [p for p in pools[control + '/stable_raw'] if p['phase'] == 'all']
        matrix, risk, _ = pool_data(scores, units, pool, control, metadata)
        draws_frame = pd.read_parquet(source / f'bootstrap-label-draws-{control}.parquet')
        if list(draws_frame.columns) != pool['shared_skill_ids'] or len(draws_frame) != 2000:
            raise ValueError('Bootstrap gold draws do not match frozen common pool')
        draws = draws_frame.to_numpy(float)
        valid = np.isfinite(draws).all(axis=1)
        for transform, predictors in (('raw', matrix), ('absolute_readout', np.abs(matrix))):
            observed = magnitude_arrays(predictors, risk[None, :])
            resampled = magnitude_arrays(predictors, draws[valid])
            for i, meta in enumerate(metadata):
                for field in MAG_FIELDS:
                    rows.append({'control': control, 'context_id': pool['context_id'],
                        'phase': 'all', 'transform': transform, 'score': meta['score'],
                        'metric': field, 'point': float(observed[field][0, i]),
                        **interval(resampled[field][:, i]),
                        'complete_pool_draws': int(valid.sum()),
                        'missing_any_skill_draws': int((~valid).sum())})
    return pd.DataFrame(rows)


def paired_contrasts(scores, units, pools, metadata, source):
    """Descriptive paired score differences on the same resampled gold labels."""
    names = [m['score'] for m in metadata]
    rows = []
    for control in ('placebo', 'null'):
        pool, = [p for p in pools[control + '/stable_raw'] if p['phase'] == 'all']
        matrix, risk, _ = pool_data(scores, units, pool, control, metadata)
        draws = pd.read_parquet(source / f'bootstrap-label-draws-{control}.parquet')
        if list(draws.columns) != pool['shared_skill_ids'] or len(draws) != 2000:
            raise ValueError('Changed paired label draws')
        values = draws.to_numpy(float)
        values = values[np.isfinite(values).all(axis=1)]
        observed = magnitude_arrays(matrix, risk[None, :])
        resampled = magnitude_arrays(matrix, values)
        for score, reference in CONTRASTS:
            a, b = names.index(score), names.index(reference)
            for field in MAG_FIELDS:
                diff = resampled[field][:, a] - resampled[field][:, b]
                rows.append({'control': control, 'score': score, 'reference': reference,
                    'metric': field, 'point_difference': float(observed[field][0, a]
                                                              - observed[field][0, b]),
                    **interval(diff)})
    return pd.DataFrame(rows)


def precision_curve(scores, pools, metadata, source):
    rows = []
    for panel in ('gold2', 'gold4', 'gold8', 'added6_only'):
        units = read_csv(source / 'precision' / (panel + '-utility.csv'))
        for control in ('placebo', 'null'):
            pool, = [p for p in pools[control + '/stable_raw'] if p['phase'] == 'all']
            matrix, risk, _ = pool_data(scores, units, pool, control, metadata)
            metrics = magnitude_arrays(matrix, risk[None, :])
            names = [m['score'] for m in metadata]
            for score in DISPLAY + ('D_original::token::unsigned',
                                    'D_real_absA::token::reward_sign_removed'):
                i = names.index(score)
                rows.append({'panel': panel, 'control': control, 'score': score,
                    'candidates': len(risk), 'large_change_gt_5pp': int((np.abs(risk) > .05 + EPS).sum()),
                    **{field: float(metrics[field][0, i]) for field in MAG_FIELDS}})
    return pd.DataFrame(rows)


def check_direction_reproduction(recomputed, prior):
    keys = ['control', 'context_id', 'phase', 'threshold', 'score']
    joined = recomputed.merge(prior, on=keys, validate='one_to_one',
                              suffixes=('_new', '_old'), indicator=True)
    if len(joined) != len(prior) or len(joined) != len(recomputed) or not joined._merge.eq('both').all():
        raise ValueError('Directional prediction grid changed')
    for field in POINT_FIELDS:
        if not np.allclose(joined[field + '_new'], joined[field + '_old'],
                           rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError('Directional prediction did not reproduce: ' + field)
    return len(joined)


def make_report(points, bootstrap, direction, contrasts, curve, output, source, checked):
    main = points[(points.control == 'placebo') & (points.phase == 'all')
                  & points.score.isin(DISPLAY)]
    direct = direction[(direction.control == 'placebo') & (direction.phase == 'all')
                       & (direction.threshold == 0) & direction.score.isin(DISPLAY)]
    direct = direct[['score', 'average_precision', 'auroc_decline_vs_increase', 'spearman']]
    table = main.merge(direct, on='score', validate='many_to_one', suffixes=('_magnitude', '_decline'))
    table = table[['score', 'transform', 'candidates', 'large_change_gt_5pp',
                   'spearman_decline', 'spearman_magnitude', 'kendall',
                   'auroc_large_change_vs_rest', 'average_precision_magnitude',
                   'top_quartile_lift']]
    comparison = bootstrap[(bootstrap.control == 'placebo') & bootstrap.score.isin(DISPLAY)
                           & bootstrap.metric.eq('spearman')]
    comparison = comparison[['score', 'transform', 'point', 'low', 'high', 'valid_draws']]
    compared = contrasts[(contrasts.control == 'placebo') & contrasts.metric.eq('spearman')]
    compared = compared[['score', 'reference', 'point_difference', 'low', 'high', 'valid_draws']]
    precision = curve[(curve.control == 'placebo')
                      & curve.score.isin(('D_original::token::reward',
                                           'D_original::token::unsigned',
                                           'M_delta_centered::token::magnitude'))]
    precision = precision[['panel', 'score', 'large_change_gt_5pp', 'spearman']]
    return ('## Material Passport\n\n'
        '- Origin Skill: academic-research-suite / experiment-agent\n'
        '- Origin Mode: validate\n'
        '- Verification Status: ANALYZED\n'
        '- Version Label: utility_precision_predictivity_seed404_v1\n\n'
        '# Seed404：更新效用标签下的方向与幅度预测\n\n'
        f'原始精度扩展目录：`{source}`；本分析目录：`{output}`。'
        '冻结37技能库、U0→U5、404个首次调用锚点、每个arm/endpoint 8组配对续跑；'
        '不重跑训练、模型前向或环境轨迹。285列冻结readout全部进入同一分析，'
        '方向表与原精度报告逐项复算一致（'+str(checked)+'行）。\n\n'
        '目标严格区分：`m=ΔM=M_U5−M_U0`，下降排序是`−m`；变化幅度是`|m|`，'
        '不是`|M_U0|`。`raw`表示原始读出值，`absolute_readout`表示`|读出值|`；'
        '后者为事后诊断变换，不是独立冻结方法。预测分数越大，预期变化越大。'
        '大变化事件固定定义为`|m|>0.05`，top-k为共同池前25%（18技能时k=5），'
        'ties等权；升降方向不混入幅度标签。\n\n'
        '## 主口径：placebo、all、18技能\n\n'
        +table.to_markdown(index=False, floatfmt='.4f')+'\n\n'
        '其中下降Spearman对`−m`，幅度Spearman/Kendall对`|m|`；AP/AUROC对应`|m|>5pp`，'
        'top-k lift是选中技能平均`|m|`除以全池平均`|m|`。'
        '方向任务的全部AP/AUROC/相关系数见`direction-recomputed.csv`，'
        '幅度任务的全部raw/absolute、placebo/null和阶段结果见`magnitude-points.csv`。\n\n'
        '## 幅度相关的配对标签不确定性\n\n'
        +comparison.to_markdown(index=False, floatfmt='.4f')+'\n\n'
        '区间为复用原2000组game×continuation配对抽样的2.5%–97.5%描述分位数；'
        '仅完整保留18技能的1717组进入主口径。'
        '全部幅度指标区间见`magnitude-bootstrap.csv`。这些区间只覆盖固定读出下的gold标签抽样，'
        '不覆盖RL种子、训练批次、任务域或285种公式搜索。\n\n'
        '## 匹配对照与重复数敏感性\n\n'
        +compared.to_markdown(index=False, floatfmt='.4f')+'\n\n'
        '差值顺序为左侧分数减去reference；每组都在相同技能、相同gold抽样上计算，'
        '区间未经285公式搜索校正。\n\n'
        +precision.to_markdown(index=False, floatfmt='.4f')+'\n\n'
        '`gold2/gold4/gold8/added6_only`共用冻结读出，仅续跑标签精度变化；'
        '重复组不是独立RL seed，不可用最有利的子集替换8组主结果。'
        '全部对照和精度表见`paired-magnitude-contrasts.csv`与`precision-curve-magnitude.csv`。\n\n'
        '## 解释边界\n\n'
        '这次是已经观察过seed404/505旧标签后的事后分析，不能从285列中挑最大相关值作确认性结论。'
        '研究单位是同一U0→U5窗口内的技能，不是token或独立训练seed；'
        '18技能共同池不能代表完整37技能库。稀少anchor使单技能标签区间仍宽；'
        '大变化事件的基率和阈值敏感，不能把相关性当作编辑收益或逐步因果归因。'
        '统计误用检查11/11：Simpson、生态谬误、Berkson/支持集、collider、基率、'
        '均值回归、幸存者、多重比较、分析路径、相关非因果、反向因果/泄漏。'
        '读出分数先于新增效用续跑冻结，但公式族与原标签已有交互，属于探索性证据。\n')


def run(output: Path = DEFAULT_OUTPUT):
    output = Path(output).resolve()
    if output.parent != COHORT or not output.name.startswith('utility-precision-predictivity-s404-v'):
        raise ValueError('Only a new scoped seed404 analysis directory is allowed')
    if output.exists():
        raise FileExistsError('Never overwrite an analysis attempt')
    verify_source(SOURCE)
    scores = read_csv(SOURCE / 'skill_scores.csv')
    metadata = read_csv(SOURCE / 'registry.csv').to_dict('records')
    if len(metadata) != 285 or scores.score.nunique() != 285:
        raise ValueError('Expected all 285 frozen readouts')
    units = pd.read_parquet(SOURCE / 'window_metrics/utility_units.parquet')
    pools = read_json(NUMERICAL / 'ranking-snapshots.json')['candidate_pools']
    direction, _ = point_tables(scores, units, pools, metadata, [0., .05])
    checked = check_direction_reproduction(direction, read_csv(SOURCE / 'reports/ranking_diagnostics.csv'))
    points = magnitude_points(scores, units, pools, metadata)
    bootstrap = magnitude_bootstrap(scores, units, pools, metadata, SOURCE)
    contrasts = paired_contrasts(scores, units, pools, metadata, SOURCE)
    curve = precision_curve(scores, pools, metadata, SOURCE)
    report = make_report(points, bootstrap, direction, contrasts, curve, output, SOURCE, checked)
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'seed404 existing updated labels; CPU-only post-hoc analysis',
            'score_columns': 285, 'gold_repeats': 8, 'new_rollouts': 0,
            'direction_target': '-delta_utility', 'magnitude_target': 'abs(delta_utility)',
            'large_change_threshold_exclusive': .05, 'top_k_rule': 'ceil(25% of common pool)',
            'analysis_code_sha256': file_hash(Path(__file__)),
            'inputs': [{'path': str(path), 'sha256': file_hash(path)} for path in (
                SOURCE / 'complete.json', SOURCE / 'provenance.json', SOURCE / 'skill_scores.csv',
                SOURCE / 'registry.csv', SOURCE / 'window_metrics/utility_units.parquet',
                SOURCE / 'reports/ranking_diagnostics.csv',
                SOURCE / 'bootstrap-label-draws-placebo.parquet',
                SOURCE / 'bootstrap-label-draws-null.parquet',
                NUMERICAL / 'ranking-snapshots.json')]}
    write_new_json(output / 'plan.json', plan)
    csv(output / 'direction-recomputed.csv', direction)
    csv(output / 'magnitude-points.csv', points)
    csv(output / 'magnitude-bootstrap.csv', bootstrap)
    csv(output / 'paired-magnitude-contrasts.csv', contrasts)
    csv(output / 'precision-curve-magnitude.csv', curve)
    write_new_bytes(output / 'report.md', report.encode())
    write_new_json(output / 'complete.json', {'status': 'complete',
        'direction_rows_reproduced': checked, 'magnitude_rows': len(points),
        'bootstrap_rows': len(bootstrap),
        'files': [{'path': path.name, 'sha256': file_hash(path)} for path in sorted(output.iterdir())
                  if path.name != 'complete.json']})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(args.output), flush=True)


if __name__ == '__main__':
    main()
