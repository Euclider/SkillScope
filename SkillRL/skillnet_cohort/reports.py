"""New immutable reports for independent local windows, never historical reports."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import file_hash, read_json, write_new_bytes, write_new_json


def passport(label):
    from phase1.archive import utc_now
    return ('## Material Passport\n\n- Origin Skill: experiment-agent\n- Origin Mode: run\n'
            f'- Origin Date: {utc_now()}\n- Verification Status: UNVERIFIED\n- Version Label: {label}\n\n')


def csv(path, frame):
    write_new_bytes(path, frame.to_csv(index=False).encode())


def ranking_diagnostics(scores, support, settings):
    from sklearn.metrics import average_precision_score, roc_auc_score
    from scipy.stats import spearmanr, kendalltau
    rows = []
    for pool in support['candidate_pools']:
        frame = scores[(scores.context_id == pool['context_id']) & (scores.phase == pool['phase'])
                       & scores.skill_id.isin(pool['shared_skill_ids'])]
        if frame.empty:
            continue
        target = -frame.delta_utility.to_numpy(float)
        for threshold in settings['event_thresholds']:
            labels = target > threshold + 1e-12
            for name in list(settings['scores']) + ['random_expected']:
                values = frame[name].to_numpy(float)  # Already signed BEFORE gold by score_snapshot.
                variable = len(frame) > 1 and np.unique(values).size > 1 and np.unique(target).size > 1
                signed = (np.abs(target) > 1e-12) & (np.abs(values) > 1e-12) if name == 'P_int' else np.zeros(len(frame), dtype=bool)
                rows.append({'context_id': pool['context_id'], 'phase': pool['phase'], 'score': name,
                    'threshold': threshold, 'candidates': len(frame), 'declines': int(labels.sum()),
                    'average_precision': float(average_precision_score(labels, values)) if labels.any() else None,
                    'auroc_decline_vs_rest': float(roc_auc_score(labels, values)) if 0 < labels.sum() < len(labels) else None,
                    'spearman': float(spearmanr(values, target).statistic) if variable else None,
                    'kendall': float(kendalltau(values, target).statistic) if variable else None,
                    'P_sign_scored_units': int(signed.sum()) if name == 'P_int' else None,
                    'P_sign_agreement_on_nonzero_point_delta': float((np.sign(values[signed]) == np.sign(target[signed])).mean()) if signed.any() else None,
                    'tie_handling_ap': 'threshold-grouped standard AP', 'zero_event_is_undefined': not labels.any()})
    return pd.DataFrame(rows)


def seed_reports(root, spec):
    root = Path(root)
    output = root / 'reports'
    performance, provenance, margins_all, diagnostics_all, budgets_all = [], [], [], [], []
    for update in (0, 5):
        for split in spec['evaluation']['splits']:
            path = root / 'evaluations' / f'u{update:04d}-{split}-performance' / 'completion.json'
            summary = read_json(path)
            if summary['status'] != 'complete' or summary['unique_games'] != summary['expected_games']:
                raise ValueError('Cannot publish incomplete performance as full traversal')
            performance.append({'seed': spec['seed'], 'update': update, 'split': split, 'task': 'all',
                                'episodes': summary['episodes'], 'success_rate': summary['success_rate']})
            for task, item in summary['per_task'].items():
                performance.append({'seed': spec['seed'], 'update': update, 'split': split, 'task': task, **item})
            provenance.append({'path': str(path.relative_to(root)), 'sha256': file_hash(path)})
    for window in sorted((root / 'windows').glob('u*-u*-*')):
        if not (window / 'sealed.json').is_file():
            raise ValueError('Window report must be sealed before composing seed report')
        config = read_json(window / 'protocol.json')
        directory = window / 'window_metrics'
        margins_all.append(pd.read_parquet(directory / 'utility_units.parquet').assign(seed=spec['seed']))
        support = read_json(directory / 'ranking_support.json')
        scores = pd.read_csv(directory / 'locked_scores_and_gold.csv')
        diagnostics_all.append(ranking_diagnostics(scores, support, config['ranking']).assign(seed=spec['seed']))
        budgets_all.append(pd.read_csv(directory / 'ranking_metrics.csv').assign(seed=spec['seed']))
        provenance.append({'path': str((window / 'sealed.json').relative_to(root)), 'sha256': file_hash(window / 'sealed.json')})
    perf = pd.DataFrame(performance)
    margins = pd.concat(margins_all, ignore_index=True) if margins_all else pd.DataFrame()
    diagnostics = pd.concat(diagnostics_all, ignore_index=True) if diagnostics_all else pd.DataFrame()
    budgets = pd.concat(budgets_all, ignore_index=True) if budgets_all else pd.DataFrame()
    for name, frame in [('performance', perf), ('utility_units', margins), ('ranking_diagnostics', diagnostics), ('ranking_budgets', budgets)]:
        csv(output / f'{name}.csv', frame)
    p1 = passport('phase12_independent_seed_v4') + f'# Phase1：seed {spec["seed"]}，冻结 SkillNet-37，U0→U5\n\n'
    p1 += '训练为五次 GRPO 迭代，不是五次 optimizer.step。所有任务共用完整 37 技能库及独立冻结 router。\n\n'
    p1 += perf[perf.task == 'all'].to_markdown(index=False) + '\n\n'
    if not margins.empty:
        q = margins[(margins.control == 'placebo') & (margins.phase == 'all')]
        columns = ['skill_id', 'anchor_count', 'game_count', 'utility_old', 'utility_new', 'delta_utility', 'ci_low', 'ci_high']
        p1 += '## 同锚点、同 continuation seed 的边际效用变化\n\n'
        p1 += q[columns].to_markdown(index=False) + '\n\n'
    p1 += ('M=success_ORIGINAL−success_PLACEBO，ΔM=M_U5−M_U0；NULL 为次要对照。'
           'CI 使用配对 game/continuation bootstrap。点估计有变化不等于已统计确证；'
           '不以正负翻转为必要成功标准。原始配对结果、动作序列和逐技能支持记录保留于各 window。\n')
    p2 = passport('phase12_readout_seed_v4') + f'# Phase2：seed {spec["seed"]}，U0→U5 预测性\n\n'
    p2 += ('实际第一轮训练 batch 提供固定状态、动作 token、mask 和 advantage；U0 与 U5 在相同输入上读出。'
           'gated D 为主，−P、C 及同源幅度指标为预登记诊断；不按 gold 选方向或窗口。'
           'readout 无额外环境 rollout，但验证标签来自独立 O/P/N 续跑，两者成本不混淆。\n\n')
    if not diagnostics.empty:
        q = diagnostics[(diagnostics.phase == 'all') & (diagnostics.threshold == 0)]
        p2 += q.drop(columns=['tie_handling_ap', 'zero_event_is_undefined']).to_markdown(index=False) + '\n\n'
    p2 += ('各分数使用相同的自然支持候选池；缺支持为弃权，零下降事件时 AP/Recall 不定义。'
           'D 是单侧下降风险，不能把 D=0 解释为稳定或上升；−P 的排序关联也不等于方向分类准确率。'
           '完整 top-k/比例预算、5pp 阈值、NULL、分阶段敏感性见 CSV/window 报告。'
           '一个 seed 不支持跨 seed 泛化结论，所有阶段分层和 continuation repeats 不是独立 RL seed。\n')
    recovery = root.parent / 'recovery-v1/cohort-recovery.json'
    if spec['seed'] == 404 and recovery.is_file():
        note = ('\n\n恢复说明：seed404 在首轮 rollout 完成、首次 optimizer.step 之前发生进度文件扫描竞态。'
                '原128条轨迹及动作token被复用，未重新采样；已有OLD概率与恢复后的前向结果逐bit核对。'
                'vLLM/worker随机状态在登记seed上重新初始化，不声称与不中断运行逐bit等价。'
                '首轮遗失的vLLM采样logprob仅用于后端差异诊断，未伪造；训练使用重新完成的native OLD。'
                '累计30小时包含首次失败运行，扣除故障停机；详见recovery-v1及其保全证据。\n')
        p1 += note
        p2 += note
        provenance.append({'path': str(recovery), 'sha256': file_hash(recovery), 'kind': 'explicit_pre_optimizer_recovery'})
    recovery_v2 = root.parent / 'recovery-v2/cohort-recovery.json'
    if spec['seed'] == 404 and recovery_v2.is_file():
        note = ('\n第二次恢复补充：首次恢复已完成5512行native OLD和reference前向，'
                '随后Python bool/NumPy标量兼容错误在首次优化前中断。'
                '本次复用全部已存的实际trainer-chosen OLD概率；每rank一个原状态经native前向精确核验，'
                '并验证所有有效token在恢复native dtype后保持精确。reference因未持久化而重算。'
                '首轮OLD entropy日志缺失，未伪造；优化器内entropy正则及其余RL参数完全保留。'
                'U1没有重新采样环境轨迹；累计30小时计入此前两个attempt。详见recovery-v2。\n')
        p1 += note
        p2 += note
        provenance.append({'path': str(recovery_v2), 'sha256': file_hash(recovery_v2),
                           'kind': 'explicit_complete_old_pre_optimizer_recovery'})
    recovery_v3 = root.parent / 'recovery-v3/cohort-recovery.json'
    if spec['seed'] == 404 and recovery_v3.is_file():
        note = ('\n第三次恢复补充：五轮RL、640条训练轨迹及每rank的204次Adam更新已完成，'
                '仅训练后导出子进程因本地verl导入路径中断。以模块入口修复并直接导出原U5，'
                '没有重跑训练、重采样训练轨迹或改变原生检查点。'
                '新导出在发布前与八rank原生FP32分片逐张量、逐bit核验；'
                '旧失败目录及记录保留。累计30小时计入此前所有实际运行，故障停机不计；详见recovery-v3。\n')
        p1 += note
        p2 += note
        provenance.append({'path': str(recovery_v3), 'sha256': file_hash(recovery_v3),
                           'kind': 'explicit_post_training_export_continuation'})
    recovery_v4 = root.parent / 'recovery-v4/cohort-recovery.json'
    if recovery_v4.is_file():
        note = ('\n运行时限修订：用户在2026-09-20明确取消三个seed累计时间上限；'
                '以上恢复说明中的30小时为历史规则，不再约束本次接续。'
                '磁盘保护、科学设置、404→505→606顺序不变，仍记录累计实际运行时间。\n')
        if spec['seed'] == 404:
            note += ('第四次恢复仅修复窗口汇总对逐轮NEW概率的过时依赖，'
                     '复用原5轮RL、U5导出、U0全量评估/anchors/540条续跑及8个读出分片。'
                     '窗口协议和C/P/D定义未改；起点OLD为实际训练概率，终点为相同输入的U5前向。'
                     'U0 HF快照物理权重为BF16，不能把元数据的FP32当成原生master精度；'
                     '参数范数诊断使用已登记原模型按原生加载步骤在CPU重建的FP32 B0，'
                     '不将有舍入的U0导出上转后冒充原生master。无额外RL或环境rollout。'
                     '原U0 BF16运行值核对为PASS，FP32导出一致性检查为FAIL，两者证据均保留。\n')
        p1 += note
        p2 += note
        provenance.append({'path': str(recovery_v4), 'sha256': file_hash(recovery_v4),
                           'kind': 'explicit_readout_continuation_and_wallclock_limit_removal'})
    write_new_bytes(output / 'phase1-results.md', p1.encode())
    write_new_bytes(output / 'phase2-results.md', p2.encode())
    write_new_json(output / 'provenance.json', {'seed': spec['seed'], 'inputs': provenance,
        'verification_status': 'UNVERIFIED', 'readout_uses_post_update_rollout_labels': False})


def cohort_report(root, plan, completed):
    root = Path(root)
    out = root / 'reports'
    files = []
    for name in ('performance', 'utility_units', 'ranking_diagnostics', 'ranking_budgets'):
        frames = [pd.read_csv(root / f'seed-{seed}' / 'reports' / f'{name}.csv') for seed in completed]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        csv(out / f'{name}-all-completed-seeds.csv', frame)
        files.append(name)
    text = passport('phase12_independent_cohort_v4') + '# Phase1–2 独立窗口汇总\n\n'
    text += f'预登记 seeds：{plan["seeds"]}；完成：{completed}；未完成：{[s for s in plan["seeds"] if s not in completed]}。\n\n'
    text += ('顺序由登记决定，不按 outcome 筛 seed。所有 seed 从共同 B0 独立训练五轮，技能内容冻结。'
             '评估 seeds 为共同随机数，因此不将重复的 U0/评估样本当成额外独立重复。'
             '跨 seed 比较先看各 seed 的同池排名和效应；不把所有 skill×seed 或阶段行视为独立样本。\n\n')
    if completed:
        rank = pd.read_csv(out / 'ranking_diagnostics-all-completed-seeds.csv')
        if not rank.empty:
            q = rank[(rank.phase == 'all') & (rank.threshold == 0)]
            text += q[['seed', 'score', 'candidates', 'declines', 'average_precision', 'spearman']].to_markdown(index=False) + '\n\n'
    text += '本文件为运行结果汇总，不预先断言 idea 成立；预算停止、支持不足或无下降事件必须一并报告。\n'
    write_new_bytes(out / 'phase12-cohort-summary.md', text.encode())
