"""Audit and publish completed seed404 v2 results without rescoring or rerunning.

Adds only a versioned cover/report, a v1-v2 metric comparison, and a publication
receipt. All experiment plans, sealed outputs, and source evidence are read-only.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from skillnet_cohort.common import REPO, file_hash, read_json, write_new_bytes, write_new_json
from skillnet_cohort.reward_variant_analysis import COHORT, check_plan, verify_records


def read(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[''], float_precision='round_trip')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_run(output):
    plan = check_plan(output)
    complete = read_json(output/'complete.json')
    require(complete['status'] == 'complete', 'Incomplete experiment')
    require(complete['provenance_sha256'] == file_hash(output/'provenance.json'), 'Changed provenance')
    provenance = read_json(output/'provenance.json')
    require(provenance['plan_sha256'] == file_hash(output/'plan.json'), 'Changed plan')
    verify_records([{'path': str(output/r['path']), 'sha256': r['sha256']} for r in provenance['files']])
    verification = read_json(output/'independent-verification.json')
    require(verification['status'] == 'PASS', 'Independent verification failed')
    require(verification['analysis_complete_sha256'] == file_hash(output/'complete.json'), 'Verification not bound to completion')
    require(verification['verification_source_sha256'] == file_hash(REPO/'scripts/verify_reward_variants_s404.py'), 'Changed independent verifier')
    verify_records(verification['additional_files'])
    print(f'preserved plan and sealed files: {output.name}', flush=True)
    return plan, complete, verification


def publish():
    prior = COHORT/'reward-variants-s404-v1'
    output = COHORT/'reward-variants-s404-v2'
    wrapper = output/'reports/phase2-results-expanded-v2.md'
    receipt_path = output/'publication-verification.json'
    comparison_path = output/'publication-v1-v2-metric-comparison.csv'
    require(not any(p.exists() for p in (wrapper, receipt_path, comparison_path)), 'Publication requires new paths')
    p1, _, _ = audit_run(prior)
    p2, complete, verified = audit_run(output)
    require(p1['registry'] == p2['registry'], 'Candidate set changed')
    require(len(p1['inputs']) == 19 and len(p2['inputs']) == 25, 'Unexpected input coverage')
    require(p1['preserved_runtime_sources'] == p2['preserved_runtime_sources'], 'Runtime source set changed')
    correction = read_json(output/'constant-correction-receipt.json')
    require(correction['cell_changes_sha256'] == file_hash(output/'constant-correction-cells.csv'), 'Changed correction cells')
    require(correction['source_sha256'] == file_hash(REPO/'skillnet_cohort/reward_variant_constant_fix.py'), 'Changed correction code')
    cells = read(output/'constant-correction-cells.csv')
    keys = ['control', 'context_id', 'phase', 'skill_id', 'score']
    s1, s2 = read(prior/'skill_scores.csv'), read(output/'skill_scores.csv')
    joined = s1.merge(s2, on=keys, suffixes=('_v1', '_v2'), how='outer', validate='one_to_one', indicator=True)
    require(joined._merge.eq('both').all(), 'Skill/score identities changed')
    changed = joined[(joined.value_v1 != joined.value_v2) & ~(joined.value_v1.isna() & joined.value_v2.isna())]
    require(len(changed) == len(cells) == correction['changed_cells'] == 366, 'Unexpected scalar corrections')
    checked = changed.merge(cells, on=keys, how='outer', validate='one_to_one', indicator='cell_match')
    require(checked.cell_match.eq('both').all(), 'Unrecorded scalar correction')
    require(np.array_equal(checked.value_v1, checked.old) and np.array_equal(checked.value_v2, checked.new), 'Correction values not reproduced')
    require(cells.reason.eq('all contributing token values exactly equal').all(), 'Nonconstant change recorded')
    require(correction['nonconstant_reductions_unchanged'], 'Nonconstant arithmetic changed')

    d1, d2 = read(prior/'ranking_diagnostics.csv'), read(output/'ranking_diagnostics.csv')
    metric_keys = ['control', 'context_id', 'phase', 'threshold', 'score']
    metrics = ['average_precision', 'auroc_decline_vs_rest', 'auroc_decline_vs_increase', 'spearman', 'kendall']
    comparisons = d1[metric_keys+metrics].merge(d2[metric_keys+metrics], on=metric_keys,
        suffixes=('_v1', '_v2'), how='outer', validate='one_to_one', indicator=True)
    require(comparisons._merge.eq('both').all(), 'Metric pools changed')
    require(len(d2) == verified['primary_and_stratified_metric_rows'] == 5220, 'Missing metrics')
    primary = comparisons[(comparisons.control == 'placebo') & (comparisons.phase == 'all') & (comparisons.threshold == 0)]
    reward = primary[primary.score.str.endswith('::reward')]
    require(len(reward) == 117, 'Reward candidates missing')
    require(np.allclose(reward.average_precision_v1, reward.average_precision_v2, atol=1e-14, rtol=0), 'Primary reward AP changed')
    primary_changed = primary[~np.isclose(primary.average_precision_v1, primary.average_precision_v2, atol=1e-14, rtol=0, equal_nan=True)]
    constants = {'D_reward_only::'+a+'::unsigned' for a in ('token', 'decision', 'game')}
    require(set(primary_changed.score) == constants, 'Primary AP changed outside constant negative controls')
    require(s2[s2.score.isin(constants)].value.eq(-1.).all(), 'Constant score not exact')
    constant_metrics = d2[d2.score.isin(constants)]
    with_events = constant_metrics[constant_metrics.declines > 0]
    require(np.allclose(with_events.average_precision, with_events.declines/with_events.candidates, atol=1e-14, rtol=0), 'Constant AP is not event prevalence')
    two_classes = constant_metrics[(constant_metrics.declines > 0)&(constant_metrics.declines < constant_metrics.candidates)]
    require(np.allclose(two_classes.auroc_decline_vs_rest, .5, atol=1e-14, rtol=0), 'Constant AUC is not chance')
    require(constant_metrics.spearman.isna().all(), 'Constant correlation must be NA')
    require(len(read(output/'ranking_budgets.csv')) == 19836, 'Missing budget diagnostics')
    require(s2.score.nunique() == complete['all_score_columns'] == 261, 'Missing score columns')

    xml = REPO/'artifacts/code_checks/reward-variants-s404-20260922-v1/constant-fixed-regression-tests.xml'
    suites = ET.parse(xml).getroot().findall('.//testsuite')
    totals = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    require(totals == {'tests': 287, 'failures': 0, 'errors': 0, 'skipped': 0}, 'Regression suite did not pass')
    root = REPO.parent
    main = root/'phase2-complete-analysis.md'
    interpretation = root/'2026-09-22-seed404-reward-readout-variants-analysis.md'
    backup = prior/'publication-original-phase2-complete-analysis.md'
    original = backup.read_bytes()
    require(len(original) == 29970, 'Unexpected original report length')
    require(main.read_bytes().startswith(original), 'Historical report prefix modified')
    require(file_hash(backup) == 'b13d3960b808af3c96b20bc5618c4e764da283146f75350e04704dc711ca88a4', 'Historical backup changed')
    planned = {wrapper, receipt_path}
    for path, body in ((main, main.read_bytes()[len(original):].decode()), (interpretation, interpretation.read_text())):
        for target in re.findall(r'\]\(([^\s)]+)\)', body):
            if '://' in target or target.startswith('#'):
                continue
            local = (path.parent/target.split('#')[0]).resolve()
            require(local.exists() or local in planned, 'Broken new report link: '+str(local))

    cover = '''## 最终发布说明：seed404 reward-directed 变式 v2

- Verification Status: ANALYZED — 单 seed、已见历史标签后的探索，不是确认性证据。
- Version Label: reward_variants_seed404_exploratory_v2_constant_preserving
- 本页先说明最终 v2 数值修正，再逐字节保留旧数值报告及本次自动扩展报告。

候选仍为 39 个 reward 公式 × 3 聚合，共 117 个 reward 分数；加对照共 261 列。
v1 成功完成后发现严格常数负对照的浮点伪排序，v2 仅保持全体输入严格相等时的精确常数，
所有非恒定列、公式、标签与阈值不变。共修正 366 个标量单元。
主 PLACEBO/all/0pp 的 117 个 reward AP 均未改变；3 个无奖励常数分数恢复
AP=5/18、AUROC=0.5、Spearman=NA。主结论仍是不足以确认稳健 reward 增益。

287 项 CPU 回归通过，5,220 行指标已独立复核，9 个原指标复现。
未新增 RL、模型前向、环境续跑、API 或其他 seed 的变式评估。
后文保留的旧 Material Passport / v1 标签只属于各段原历史文件或公式族；
本发布新增分析采用本目录 v2 数值。旧页自身的验证状态不被追溯改写。

[修正逐单元对照](../constant-correction-cells.csv)、
[独立核验](../independent-verification.json)、
[最终发布验收](../publication-verification.json)、
[全部候选指标](../ranking_diagnostics.csv)。

---

'''
    write_new_bytes(wrapper, cover.encode()+(output/'reports/phase2-results-expanded.md').read_bytes())
    write_new_bytes(comparison_path, comparisons.drop(columns='_merge').to_csv(index=False).encode())
    files = [main, interpretation, root/'HANDOFF.md', root/'HANDOFF-DETAILS.md', backup,
        prior/'publication-initial-interpretation.md', prior/'publication-initial-appended-report.md',
        prior/'NUMERICAL-CONSTANT-WARNING.md', wrapper, comparison_path, xml, Path(__file__),
        output/'constant-correction-receipt.json', output/'constant-correction-cells.csv',
        output/'direction-confusions.csv', output/'independent-verification.json']
    receipt = {'status': 'PASS', 'published_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_status': 'ANALYZED_single_seed_posthoc_not_confirmatory',
        'final_result_version': p2['version'], 'plan_sha256': file_hash(output/'plan.json'),
        'complete_sha256': file_hash(output/'complete.json'),
        'provenance_sha256': file_hash(output/'provenance.json'),
        'original_core_inputs_unchanged': len(p1['inputs']), 'final_bound_inputs': len(p2['inputs']),
        'original_runtime_sources_unchanged': len(p2['preserved_runtime_sources']),
        'v1_sealed_outputs_preserved': True, 'v2_sealed_outputs_preserved': True,
        'same_candidate_registry': True, 'score_columns': 261, 'reward_scores': 117,
        'ranking_diagnostic_rows': 5220, 'ranking_budget_rows': 19836,
        'constant_scalar_cells_corrected': len(cells), 'all_scalar_changes_match_constant_receipt': True,
        'primary_reward_ap_unchanged': True, 'constant_negative_controls_are_chance': True,
        'primary_ap_changes': primary_changed[['score', 'average_precision_v1', 'average_precision_v2']].to_dict('records'),
        'old_main_report_prefix_bytes_preserved': len(original), 'new_report_links_checked': True,
        'regression': totals, 'independent_metrics_verified': verified['primary_and_stratified_metric_rows'],
        'new_training': False, 'new_model_forwards': False, 'new_rollouts': 0, 'new_api_calls': 0,
        'other_seeds_evaluated_with_new_variants': [], 'phase3_metric_changed': False,
        'publication_files': [{'path': str(p.resolve()), 'sha256': file_hash(p)} for p in files]}
    write_new_json(receipt_path, receipt)
    print(f'PASS: final v2 publication, 117 reward APs unchanged, 366 constant cells corrected, {len(original)} historical bytes preserved', flush=True)


if __name__ == '__main__':
    publish()
