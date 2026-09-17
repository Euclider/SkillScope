"""Final Phase2 synthesis, generated only from fully verified evaluations."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase2.measure import phase
from phase2.protocol import signal_directory
from phase2.utilities import read_evaluations

PLAN = "complete_report_plan_20260914.json"


def table(frame, columns=None, pp=()):
    q = frame.copy()
    for name in pp:
        if name in q: q[name] *= 100
    if columns is not None: q = q[columns]
    return q.to_markdown(index=False, floatfmt=".6g") + "\n" if len(q) else "无符合支持条件的结果。\n"


def behavior(evaluations):
    rows = []
    gold = evaluations[evaluations.purpose == "gold"]
    keys = ["skill_id", "anchor_id", "continuation_seed"]
    for update, q in gold.groupby("update"):
        original = q[q.arm == "original"]
        for control in ("placebo", "null"):
            paired = original.merge(q[q.arm == control], on=keys, suffixes=("_o", "_c"), validate="one_to_one")
            if len(paired) != len(original): raise ValueError("Behavior requires every matched arm pair")
            rows.append({"update": update, "control": control, "paired_anchor_repeats": len(paired),
                "first_action_flip": (paired.first_action_o != paired.first_action_c).mean(),
                "suffix_action_divergence": np.mean([a != b for a, b in zip(paired.action_sequence_o, paired.action_sequence_c)]),
                "reward_disagreement": (paired.success_o != paired.success_c).mean(),
                "original_suffix_unique_skills": original.selected_skill_ids.map(lambda v: len(set(v))).mean(),
                "original_suffix_steps": original.suffix_trajectory_length.mean()})
    return pd.DataFrame(rows)


def rank_summary(metrics, pool):
    primary = metrics[(metrics.phase == "all") & (metrics.context_id == pool["context_id"]) &
                      (metrics.start_update == pool["start_update"]) & (metrics.global_update == pool["global_update"]) &
                      (metrics.target == "decline") & (metrics.threshold == 0)]
    paragraphs = [f"主共同支持池为 {pool['shared_supported_units']} 个 Skill：{', '.join(pool['shared_skill_ids']) or '无'}；"
                  f"另有 {pool['excluded_units']} 个单元不进入统一方法比较。"]
    for k, group in primary.groupby("k"):
        chosen = group[group.score.isin(["D_contribution", "P_int", "delta_centered_norm", "random_expected"])]
        descriptions = [f"{r.score}：Precision@{k}={r.precision_at_k:.3f}，下降量覆盖率={r.captured_change_mass:.3f}"
                        for r in chosen.itertuples()]
        paragraphs.append("；".join(descriptions) + "。")
        d = group[group.score == "D_contribution"]
        norm = group[group.score == "delta_centered_norm"]
        if len(d) == 1 and len(norm) == 1:
            dp = d.precision_at_k.iloc[0] - norm.precision_at_k.iloc[0]
            dc = d.captured_change_mass.iloc[0] - norm.captured_change_mass.iloc[0]
            paragraphs.append(f"固定 k={k} 时，D 相比 centered norm 的命中率差为 {dp:+.3f}、下降量覆盖率差为 {dc:+.3f}；"
                              "正值表示该预算下 D 更高，零表示持平，负值表示 D 更低，nan 表示该量未定义。这不是统计显著性检验。")
    paragraphs.append("这描述同一更新窗口内的定位/排序，不是效用数值回归，也不是跨 seed 的显著性结论。"
                      "下降量为零时覆盖率未定义；点估计下降命中不等于命中统计确证的下降。")
    return "\n\n".join(paragraphs) + "\n"


def build(root):
    root = Path(root).resolve()
    proof_path = root/"evaluation_completion.json"
    if not proof_path.exists(): raise ValueError("Full report requires completed verified evaluation")
    proof = json.loads(proof_path.read_text())
    plan = json.loads((root/PLAN).read_text())
    digest = sha256_file(root/"protocol.json")
    if proof["status"] != "complete_and_verified" or proof["protocol_sha256"] != digest or plan["protocol_sha256"] != digest:
        raise ValueError("Completion/report protocol mismatch")
    for source in plan["frozen_sources"]:
        if sha256_file(Path(source["path"])) != source["sha256"]: raise ValueError("Historical report/source changed")
    config = json.loads((root/"protocol.json").read_text())
    legacy = Path(plan["legacy_root"])
    archive = legacy/"reports/2026-09-12-observation-audit-v1"
    comparisons = pd.read_csv(archive/"comparisons.csv")
    old_supported = pd.read_csv(archive/"all_supported.csv")
    old_test = pd.read_csv(archive/"heldout.csv")
    performance = pd.DataFrame(json.loads((legacy/"metrics/prospective_performance.json").read_text())["results"])
    wm = root/"window_metrics"
    raw_path = wm/"raw_features_and_semantic_utility.csv"
    ranking_path = wm/"ranking_metrics.csv"
    if sha256_file(raw_path) != proof["raw_features_and_utility_sha256"] or sha256_file(ranking_path) != proof["ranking_metrics_sha256"]:
        raise ValueError("Verified result files changed")
    raw = pd.read_csv(raw_path)
    units = pd.read_parquet(wm/"utility_units.parquet")
    ranking = pd.read_csv(ranking_path)
    pools = json.loads((wm/"ranking_support.json").read_text())["candidate_pools"]
    evaluations = read_evaluations(root)
    b = behavior(evaluations)
    b.to_csv(wm/"complete_report_behavior.csv", index=False)
    updates = []
    for cohort, directory, values in (("pilot", legacy, range(31, 36)), ("expanded", root, config["post_updates"])):
        for update in values:
            sd = directory/"signals"/f"u{update:04d}"
            c = json.loads((sd/"committed.json").read_text())
            parameter = json.loads((sd/"parameter_delta.json").read_text())
            updates.append({"cohort": cohort, "update": update, "rollouts": 32,
                "Adam_steps": c["optimizer_steps"], "Adam_before": c["adam_step_before"], "Adam_after": c["adam_step_after"],
                "parameter_L2": parameter["delta_l2"], "relative_L2_percent": parameter["relative_delta_l2"]*100})
    update_frame = pd.DataFrame(updates)
    update_frame.to_csv(wm/"complete_report_updates.csv", index=False)
    anchors = []
    for item in config["evaluation"]["anchor_sets"]:
        records = [json.loads(s) for s in Path(item["anchors_path"]).read_text().splitlines() if s.strip()]
        counts = Counter(phase(r["trigger_step"]) for r in records)
        anchors.append({"skill": item["skill_id"], "anchors": len(records), "games": len({r["game_id"] for r in records}),
                        "first_step_min": min(r["trigger_step"] for r in records), "first_step_max": max(r["trigger_step"] for r in records),
                        **{p: counts[p] for p in ("initial", "early", "middle", "late")}})
    primary = raw[(raw.phase == "all") & raw.gold_evaluation_available]
    down = int((primary.delta_utility < -1e-12).sum()); up = int((primary.delta_utility > 1e-12).sum())
    ci_down = int((primary.ci_high < 0).sum()); ci_up = int((primary.ci_low > 0).sum())
    old_nonzero = old_test[old_test.delta_utility.abs() > 1e-12]
    counts = old_nonzero.groupby("global_update").delta_utility.agg(
        positive=lambda x: int((x > 0).sum()), negative=lambda x: int((x < 0).sum())).reset_index()
    text = ["# Phase2 完整实验分析：Policy update 是否包含 Skill 效用变化的预测信息\n",
        f"生成时间（UTC）：{utc_now()}。本报告整合已完成的旧单步 pilot 与新 U35→U40 窗口；新批次全部轨迹及预测时间隔离已通过核验。\n",
        "## 1. 结论摘要\n",
        "研究目标是以实际 policy update 的低成本读出定位需要更新的 Skill；不要求准确拟合效用数值，也不以正→负 flip 作为必要条件。D 是下降风险分数，P 是有符号投影，幅度指标回答的是变化程度。\n",
        f"新窗口四个 Skill 中，语义效用点估计下降 {down} 个、上升 {up} 个、不变 {len(primary)-down-up} 个；"
        f"ΔM 的配对 95% CI 完全小于 0 的为 {ci_down} 个、完全大于 0 的为 {ci_up} 个。CI 跨 0 不等于各 Skill 都必须朝同一方向变化，而是该单元当前测量不确定。\n"]
    for pool in pools:
        if pool["phase"] == "all": text.append(rank_summary(ranking, pool))
    text += ["旧单步 pilot 中 D 对下降的 pooled 排序相关为 +0.417（19 个支持单元）及 +0.556（7 个时间外单元），"
             "但非零测试变化方向与 update 身份混杂；旧等容量回归未显示 P/D 的额外数值预测收益。该事实不否定无需回归的直接排序目标，也不能被用来宣布方向预测成立。\n",
        "## 2. 两个批次的证据范围与可比性\n",
        "| 项目 | 旧单步 pilot | 新扩大窗口 |\n|---|---|---|\n"
        "| 同一 Seed303 延续路径 | U30→U35，5 个相邻窗口 | U35→U40，1 个五步窗口 |\n"
        "| 评估端点 | U30–U35，共 6 个 | U35/U40，共 2 个 |\n"
        "| anchors | 旧 B0 的 200 个首次调用 anchors | U35 自然 rollout 新采集的 200 个 anchors |\n"
        "| 每 anchor–arm–端点 | 1 evidence + 1 gold | 2 evidence + 4 gold |\n"
        "| 三臂 suffix 总数 | 7,200 | 7,200（U35 导入 3,600；U40 新增 3,600） |\n"
        "| 主要比较 | 时间外 readout/等容量 probe | 预锁定同窗口直接 top-k 排序 |\n",
        "两个批次合计 14,400 条独立归档的实验 suffix，但不是 14,400 个独立预测单元。新旧 U35 的支持 state 和 continuation seeds 不同，"
        "不能拼接成一条未经控制的效用时间曲线，也不能把差异全部归因于窗口变大。旧 pilot 原计划的 U36 未执行；新批次 U36–U39 仅归档单步信号，没有新设中间 gold。\n",
        "Phase1 的多 seed 结果只作为研究动机：支持效用会变化及 S_int 与变化的关联。Phase2 当前仍是一条 seed303 路径，不能将 Phase1 的跨 seed 泛化直接转写为 P/D 的跨 seed 泛化。\n",
        "## 3. 实际 RL 更新和测量协议\n",
        "Qwen3.5-4B、ALFWorld clean、GRPO；8 games × 4 rollouts=32 条/update，LR=1e-6，KL=0.01；decision minibatch=32，microbatch=1，"
        "prompt/response=2048/64，最多 30 环境步。model、Adam 状态和 scheduler 连续恢复；global update 不等于一次 Adam step。硬件/分片数随资源改变，不声称固定并行布局的逐 bit 随机路径复现。\n",
        table(update_frame),
        "参数 L2 是去除 tied-head 重复计数后的 FP32 差分；累计净位移不是各单步 L2 相加。新窗口共 160 条训练 rollout、70 次 Adam steps（611→681），U35→U40 净参数 L2=0.517594766。所有 intermediate FP32 policy、实际 batch、advantage/mask 和 old/new 全词表概率保留。\n",
        "Bank 冻结为 12 general + 32 task-specific；clean 候选为 12+6。每步按 state 路由一个 Skill，不是一次把整个 bundle 喂给 policy。本轮评估固定 4 个自然路由 Skill，不宣称全 18/44 条 Skill 已评估。新 anchor 覆盖：\n",
        table(pd.DataFrame(anchors)),
        "initial=0、early=1–4、middle=5–14、late≥15。信号支持另要求至少 20 个非零 advantage decisions、4 个非零支持训练 games、8 条非零支持轨迹。gold anchors 多不代表训练方向支持充分；unsupported 不编码为零风险。\n",
        "## 4. 干预、效用和指标的精确定义\n",
        "在目标 Skill 第一次调用前精确重放同一 prefix，随后 ORIGINAL/模板与 token 长度匹配的 PLACEBO/目标 payload 为空的 NULL 自由续跑。"
        "后续每次路由到目标 Skill 都继续相同干预；其他 Skill、候选 ID/描述和 Router 不变。NULL 不是删除该条目重新检索，也不是禁用整个 Skill Bank。\n",
        "主效用 M_sem=E[success_O−success_P]，ΔM_sem=M_new−M_old；O−NULL 为次要对照。每个 game 内平均 anchors/repeats，再对 games 等权；"
        "95% CI 使用 10,000 次配对 game/continuation bootstrap。效用表单位 pp；环境终局成功奖励为 10，return 差为 success 差乘 10。\n",
        "设 u_O=logπ_new(·|z,s)−logπ_old(·|z,s)，u_ctl 为相同 teacher-forced response 前缀下的对照变化，δ_int=u_O−u_ctl；"
        "d=A(e_a−π_old)，C_upd=cos(d,u_O)，P_int=〈d,δ_int〉/(||d||+ε)，D=E[有效方向及 C_upd≥0、||δ||≥τδ 时的 max(0,−P)]。"
        "词表全量 248,320，W=I，ε=1e−12，τδ=1e−8；零 advantage/未通过 gate 的 token 仍在 D 聚合分母。mean(P)>0 与 D>0 可同时成立。\n",
        "新窗口将 U36 旧 batch 的实际 state/token/advantage 固定，在 U35 和 U40 端点重放。它是累计 interaction 对起始 reward direction 的投影，"
        "不是五个 D 相加，也不假定中间 reward direction 不变。d 是 outcome-consistent 局部方向，不是包含 KL/Adam/clipping 的完整参数梯度。\n",
        "P>0 不数学保证长期 ΔM>0；D≥0，无符号，小 D 不证明效用上升。centered/raw norm、KL/JS 和 activation norm 是幅度读出。"
        "全局参数范数同一窗口内对所有 Skill 相同，不能据此排序。JVP 和线性化误差未实现，不能用参数范数冒充。\n",
        "## 5. 旧单步结果：保留但不混入新测试\n",
        table(comparisons[comparisons.signal.isin(["D_contribution", "P_int", "delta_centered_norm", "delta_norm", "C_upd_centered", "old_margin"])],
              ["scope", "signal", "risk_orientation", "units", "updates", "rho_raw_magnitude", "rho_risk_decline", "ap_any_decline"]),
        "旧时间外非零变化按 update 分布如下。若不同 update 自身的标签方向不同，pooled 相关可能来自区分 update，不能据此证明同一 update 内能选对 Skill：\n",
        table(counts),
        "旧回归结果仅作为辅助，不再以预测具体 ΔM 的 MAE 作为当前主要成功标准：\n",
        table(performance, ["model", "test_units", "test_updates", "signed_MAE", "conditional_sign_accuracy"], pp=["signed_MAE"]),
        "上表 signed_MAE 已转为 pp。旧 `cle_004` U34→U35 的 ΔM=−14.29 pp、95% CI [−28.57,−1.79]，属于明确的变化样本；"
        "点估计由正到负不等价于独立 calibration 确认的 harmful sign flip。19 个支持单元及 7 个测试单元的全部原始值仍在旧 observation-audit-v1 的 CSV。\n",
        "## 6. 新窗口的边际效用变化与双轴分解\n",
        table(primary, ["skill_id", "supported", "anchor_count", "game_count", "utility_old", "utility_new", "delta_utility", "ci_low", "ci_high"],
              pp=["utility_old", "utility_new", "delta_utility", "ci_low", "ci_high"]),
        "方向不需要跨 Skill 一致。CI 跨 0 表示不确定，不等于必须让所有 Skill 效用统一变正或变负。有限生成重复下的退化 [0,0] CI 也不证明真实期望恒定。\n",
        table(primary, ["skill_id", "original_old", "original_new", "control_old", "control_new", "delta_original", "delta_control", "delta_utility"],
              pp=["original_old", "original_new", "control_old", "control_new", "delta_original", "delta_control", "delta_utility"]),
        "ΔM=ΔORIGINAL−ΔPLACEBO。ORIGINAL 改善但 ΔM 下降可表示相对增益缩小；ORIGINAL 和 PLACEBO 都退化则需考虑一般性 policy 变化。"
        "这些是固定 prefix 的锚点续跑成功率，不是全局从 reset 开始的 policy validation；PLACEBO/NULL 也不是全库 skill-free baseline。\n",
        "NULL 次要对照（pp）：\n",
        table(units[(units.control == "null") & (units.phase == "all")], ["skill_id", "utility_old", "utility_new", "delta_utility", "ci_low", "ci_high"],
              pp=["utility_old", "utility_new", "delta_utility", "ci_low", "ci_high"]),
        "## 7. D 相比幅度读出是否有下降排序优势？\n",
        "以下风险方向、共同支持池、k=1/2 与 25%/50% 预算均在目标 gold 前锁定；重合预算去重。D、−P、+norm 与旧 margin 低优先、通用对照更新、随机期望比较。"
        "并列使用随机选取的期望，不按 gold 破并列。Precision/Recall 标签阈值分别为 0 与 5 pp；下降量始终累计 max(0,−ΔM)，不随标签阈值修改。\n"]
    rank_columns = ["score", "n_candidates", "events", "k", "precision_at_k", "recall_at_k", "captured_change_mass", "spearman", "kendall"]
    for threshold in (0., .05):
        text += [f"### 7.{1 if threshold == 0 else 2} 下降事件阈值：{threshold*100:g} pp\n",
                 table(ranking[(ranking.phase == "all") & (ranking.target == "decline") & (ranking.threshold == threshold)], rank_columns)]
    text += ["无下降事件时 Recall/下降量覆盖率未定义；没有 score/gold 变异时相关未定义。主共同支持池较小时排序分辨率低，不能把表中数值当作统计显著性。"
             "同一窗口的所有候选共享 update 身份，因此该表直接检验窗口内定位，但仍不足以估计跨独立更新/seed 的稳定性。\n",
        "### 7.3 原始读出与 signed ΔM\n",
        table(primary, ["skill_id", "supported", "delta_utility", "P_int", "D_contribution", "D_ungated_contribution", "delta_centered_norm", "delta_norm", "C_upd", "gate_coverage"], pp=["delta_utility"]),
        "P 列为原始 P，不是风险排序使用的 −P。Reward projection 的优势须体现在统一池中的命中/覆盖或顺序，而非仅因 D 有符号语义就认定其更优。"
        "这里 D 本身无符号；有符号趋势应结合原始 P 和连续 ΔM 读出，不做事后反号或重新拟合。\n",
        "### 7.4 幅度目标的次要审计\n",
        table(ranking[(ranking.phase == "all") & (ranking.target == "any_change") & (ranking.threshold == 0) &
                      ranking.score.isin(["D_contribution", "P_int", "delta_centered_norm", "delta_norm", "random_expected"])], rank_columns),
        "|ΔM| 定位与下降定位是两个不同目标。各原预定 score 的方向没有为幅度目标重新调参，该表不能替代下降主分析。\n",
        "## 8. 中途调用、轨迹变化与支持不足\n",
        table(b, pp=["first_action_flip", "suffix_action_divergence", "reward_disagreement"]),
        "上述比例单位 %，是各端点 ORIGINAL 与对照臂的行为差异，不是跨 checkpoint 的 S_int；按 matched anchor/repeat 描述，不按 game 等权。"
        "distinct Skill 数和长度仅统计自由 suffix，不含重放 prefix。行为 divergence 不必产生 reward 差，更不等价于 harmful flip。\n",
        table(raw, ["skill_id", "phase", "supported", "gold_evaluation_available", "delta_utility", "P_int", "D_contribution", "delta_centered_norm"], pp=["delta_utility"]),
        "训练信号支持和自然 gold anchor 支持分开记录；没有 anchor 的阶段标签留空，不能伪造零变化。相同 anchors/repeats 的不同 phase 和 all 不是独立窗口。\n",
        "## 9. 结论边界与下一阶段的增量价值\n",
        "本批完成的是测量链路与预锁定直接排序检验。它不要求精准回归，也不要求 Skill 同向变化；但依然需要在独立更新窗口/seed 中证明排序重复性。"
        "当前只一条延续路径、4 个评估 Skill，不能声称已覆盖全 Bank、不同 RL 设置或全部 task。训练 batch 与 unseen anchors 的 state 分布不同，局部投影到长期 reward 仍是经验问题。\n",
        "预测锁定前没有使用 U40 held-out 续跑结果，符合 without post-update gold rollout 的边界；仍使用已有训练 rollout、完整新旧模型前向和旧效用 evidence。"
        "离线 gold 是验证成本，不是在线输入。尚未做 equal-budget 新 trajectory/短 behavioral probe 对比，不能宣称已验证节省多少 rollout 或端到端 FLOPs/时间优势。\n",
        "Phase3 应预注册相同 top-k 编辑预算、相同编辑器和独立编辑后评估，比较 D/−P/norm、随机、old-margin、已有 trajectory-summary，以及预算匹配的新 rollout/probe。"
        "同时扫描所需编辑比例、rollout 预算和提升/错误修改率，形成成本—收益曲线。不能用当前观测性排序直接宣称它是失效机制或 Skill 修改已有效。\n",
        "## 10. 归档与可复核性\n",
        f"科学协议 SHA-256：`{digest}`；新端点完整性和逐轨迹 SHA-256：[{proof_path.name}]({proof_path})。\n",
        f"原始指标/全部分层：[CSV]({raw_path})；预锁定排序与实际标签：[CSV]({wm/'locked_scores_and_gold.csv'})；"
        f"top-k 全结果：[CSV]({ranking_path})；轨迹目录：`{root/'evaluations/u0040/trajectories'}`。\n",
        f"旧 19 单元：[CSV]({archive/'all_supported.csv'})；旧 7 测试单元：[CSV]({archive/'heldout.csv'})；"
        f"旧多指标比较：[CSV]({archive/'comparisons.csv'})。\n",
        "报告只描述完成的实验与统计边界，不混入中断/启动失败作为实验样本。旧报告和 Phase1 结果保持不变；执行资源修订与日志独立归档。\n"]
    output = Path(plan["output"])
    if output.parent != root.parents[3]: raise ValueError("Report must stay in the skill-RL project folder")
    temporary = output.with_suffix(".md.partial")
    temporary.write_text("\n".join(text))
    os.replace(temporary, output)
    source_files = [proof_path, root/"protocol.json", raw_path, ranking_path, wm/"utility_units.parquet",
                    archive/"comparisons.csv", archive/"all_supported.csv", archive/"heldout.csv"]
    atomic_write_json(root/"full_report_completion.json", {"created_at": utc_now(), "status": "complete",
        "output": str(output), "report_sha256": sha256_file(output),
        "generator_sha256": sha256_file(Path(__file__)), "report_plan_sha256": sha256_file(root/PLAN),
        "sources": {str(p): sha256_file(p) for p in source_files},
        "scope": "Completed legacy single-update pilot plus fully verified U35-to40 window; not all proposed future Phase2/Phase3 experiments"})
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    print(build(p.parse_args().root))
