"""Produce a progressively updated, evidence-backed Phase2 meeting report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error

from phase1.archive import atomic_write_json, utc_now
from phase2.utilities import read_evaluations, units
from phase2.signed_analysis import summarize, direction_counts


def corr(x,y):
    if len(x)<3 or len(set(x))<2 or len(set(y))<2:return None
    return float(spearmanr(x,y).statistic)


def table(frame,columns):
    if frame.empty:return "尚无完整结果。\n"
    return frame[columns].to_markdown(index=False,floatfmt=".4f")+"\n"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    a.root=a.root.resolve()
    config=json.loads((a.root/"protocol.json").read_text())
    metrics=a.root/"metrics"
    metrics.mkdir(exist_ok=True)
    report=a.output or Path(__file__).resolve().parents[2]/"2026-09-10-phase2-semantic-direction-fast-results.md"
    status=json.loads((a.root/"status.json").read_text()) if (a.root/"status.json").exists() else {"stage":"启动与严格采集"}
    committed=sorted(int(x.parent.name[1:]) for x in (a.root/"signals").glob("u*/committed.json"))
    checkpoints=sorted(int(x.name.removeprefix("global_step_")) for x in (a.root/"checkpoints").glob("global_step_*") if (x/"data.pt").exists())
    e=read_evaluations(a.root)
    u,g,m=units(a.root)
    signals=[]
    for update in committed:
        f=pd.read_parquet(a.root/"signals"/f"u{update:04d}"/"skill_context_features.parquet")
        signals.append(f)
    f=pd.concat(signals,ignore_index=True) if signals else pd.DataFrame()
    text=["# Phase2：语义边际效用方向验证（首批滚动报告）\n",
          f"更新时间（UTC）：{utc_now()}。汇报目标：2026-09-11 上午（北京时间）。\n",
          f"运行：`{config['run_id']}`。当前阶段：`{status.get('stage')}`。\n",
          "## 1. 主目标与实验边界\n",
          "主分析为 `M_sem = success(ORIGINAL) − success(PLACEBO)`，以连续 signed `ΔM_sem` 为目标，重点检验 `P_intᴾ` 的方向预测增量。`Dᴾ` 对下降风险的预测为次要分析；NULL 是同步记录的次要对照。上升、下降均有研究价值，harmful sign flip 不是成立条件。\n",
          "本轮从选定的 Seed303 U30 完整恢复，固定采集 U31–U36。U31/U32 为开发，U33 为边界隔离，U34–U36 为时间外测试。首批只有一个父 seed、3个测试 update；事件不足或区间跨0时报告不确定。\n",
          "为赶汇报，先采用每 anchor 两次独立 continuation：41011 估计旧 margin，42011 构造 matched old/new gold。使用全部200个旧 B0 anchors、4个 supported Skills，覆盖中途调用。时间外测试复用固定 game/state support，不声称新的 game 或 seed 泛化。\n",
          "## 2. 实际进度与完整性\n",
          f"已保存恢复 checkpoint：{checkpoints}；已锁定信号：{committed}；完整 gold/evidence suffix：{len(e):,} / 8,400。\n",
          "旧/新 ORIGINAL 全词表概率直接来自训练 actor 的相同前向路径。PLACEBO/NULL 用原始精度端点重建并 teacher-force 同一旧 response。advantage 保存实际 actor loss tensor，decision ID 随重排携带；补齐重复行在统计中去重，并保留实际 optimizer 使用记录。\n"]
    if status.get("error"):text.append(f"当前需处理的运行错误：`{status['error']}`。\n")
    if (a.root/"attempts/u0034-failed-1789058617473162626/README.md").exists():
        text.append("9月11日00:43的U34单卡尝试在训练前被旧preflight的两卡下限拦下，未发生新更新。05:23后已适配显式卡数检查（Phase1默认不变），从U33重新按空闲资源续训；失败记录见 `attempts/u0034-failed-1789058617473162626/README.md`。\n")
    if (a.root/"attempts/u0034-failed-1789078232762546641/evidence/recovery.json").exists():
        text.append("9月11日06:10的U34五卡尝试在首次backward发生共享显存OOM，未完成Adam step或保存U34端点。13:10人工审计后，将32条rollout、750行旧概率及batch等移入 `attempts/u0034-failed-1789078232762546641/evidence/`，失败数据不进入主分析；原文件与轨迹索引备份均可恢复。从U33重新采样续训，不声称逐bit续接失败五卡随机流；具体原因见该尝试的 `README.md`。\n")
    if (a.root/"attempts/u0034-failed-1789104704603119618/evidence/recovery.json").exists():
        text.append("9月11日13:31的U34原生四卡尝试也在首次backward因另一进程占约39GiB显存而OOM，未完成Adam step。15:35将32条轨迹、676行batch/旧概率及前向进度归入 `attempts/u0034-failed-1789104704603119618/evidence/`，未删除或计入正式结果。按既有资源协议准备从U33改用当前空闲卡数续训，双卡时采用CPU AdamW；原生四卡没有elastic审计，不将其误记为丢失恢复记录。\n")
    watch=a.root/"gpu_watch_latest.json"
    if watch.exists():
        watched=json.loads(watch.read_text())
        text += [f"最近一次等待资源检查（UTC）`{watched['at']}`：即时空闲候选 `{watched['idle_gpus']}`，连续两次确认空闲 `{watched['confirmed_idle_gpus']}`。此为调度边界快照，不表示运行中GPU的实时空闲状态。\n",
                 "监控每30秒检查，显存占用<1500MiB且利用率<10%、连续两次满足才续训；使用所有确认空闲卡。U34首次初始化OOM记录已保存于 `attempts/u0034-attempt1-startup-oom/`。只在尚无任何新训练证据的初始化OOM时自动等待重试；其他异常仍停止。`gpu_watch.jsonl`保留检查记录，`status.json`为当前阶段。\n"]
    if (a.root/"runtime_resources/latest.json").exists():
        text.append("9月11日15:43起，另有独立只读监控每15秒归档物理GPU显存/利用率、compute PID与磁盘余量，见 `runtime_resources/pipeline-2142632.jsonl`（后续运行按pipeline PID另存）。该监控不预留显存或终止其他任务；不能保证共享GPU独占。\n")
    allocations=[json.loads(path.read_text()) for path in sorted((a.root/"allocations").glob("u*.json"))]
    if allocations:
        text += ["### 按可用资源续训（用户于9月10日授权）\n",
                 "下表为各update最近一次启动尝试的分配，不代表当前仍占用这些GPU；完整分配历史保存在 `allocation_history.jsonl` 与失败尝试目录。9月11日起，每次启动的preflight与不可变manifest分别保存在 `launch_manifests/`，路径由allocation记录；旧manifest不覆盖，逻辑训练ID不变。\n",
                 table(pd.DataFrame(allocations),["global_update","physical_gpus","world_size","source_world_size","optimizer_backend","training_rollouts"]),
                 "LR=1e-6、KL=0.01保持不变。卡数只在 update 边界调整；参数、Adam moments/counter 与 scheduler 保留。换 world size 后使用预定的独立 rank RNG，不能声称逐 bit 延续8卡轨迹。1–2卡采用CPU AdamW；其他卡数保留GPU AdamW。3/5/6/7卡通过零权重同步槽保持每个完整 optimizer minibatch 的32个有效 decision，而不是悄悄改变有效batch。\n",
                 "资源占用已经导致先前12–18小时排期失效；首批报告按实际完成端点发布，不以汇报截止时间冒充实验完成。恢复逐rank校验见 `elastic_restore_audits/`，资源修订见 `resource_policy.json`。\n"]
        restored=[]
        for directory in sorted((a.root/"elastic_restore_audits").glob("u*")):
            rows=[json.loads(path.read_text()) for path in directory.glob("rank-*.json")]
            if not rows:continue
            parity=[row["full_vocab_parity_max_abs"] for row in rows if row["full_vocab_parity_max_abs"] is not None]
            restored.append({"update":int(directory.name[1:]),"audited_ranks":len(rows),
                "world_size":rows[0]["world_size"],"Adam_at_restore":rows[0]["optimizer_step"],
                "all_checksums_match":all(row["checksums_match"] for row in rows),
                "parity_max_abs":max(parity) if parity else None})
        if restored:
            text += [table(pd.DataFrame(restored),list(restored[0])),
                     "该表只证明加载状态校验通过，不代表对应 update 已训练完成。U32 首次四卡尝试在恢复后 rollout 入口遇到 FSDP inference-buffer 错误，未进行新更新；已改用 no_grad 校验并通过真实双卡生命周期测试，失败日志完整保留在 `attempts/u0032-attempt1-inference-cache/`。\n"]
    evidence=[]
    for file in sorted((a.root/"batches").glob("u*/manifest.json")):
        bm=json.loads(file.read_text())
        update=bm["global_update"]
        stepfile=a.root/"optimizer_steps"/f"u{update:04d}-rank0.jsonl"
        steps=[json.loads(line) for line in stepfile.read_text().splitlines() if line.strip()] if stepfile.exists() else []
        evidence.append({"update":update,"batch_rows":bm["row_count"],"unique_decisions":bm["unique_decisions"],
                         "loss_tokens":bm["loss_tokens"],"alignment_audit":(file.parent/"alignment_audit.json").exists(),
                         "old_full_vocab_rows":len(list((a.root/"old_logprobs"/f"u{update:04d}").glob("row-*.pt"))),
                         "new_full_vocab_rows":len(list((a.root/"new_logprobs"/f"u{update:04d}").glob("row-*.pt"))),
                         "recorded_Adam_steps":len(steps)})
    if evidence:
        text.append(table(pd.DataFrame(evidence),list(evidence[0])))
        text.append("此表可含尚在进行的 update；只有 checkpoint 与 signals committed 后才视为端点测量完成。\n")
    if not f.empty:
        primary=f[(f.control=="placebo")&(f.phase=="all")]
        text += ["## 3. Reward-directed signals（读取新 gold 前锁定）\n",
                 table(primary,["global_update","skill_id","supported","nonzero_advantage_decisions","nonzero_advantage_games","P_int","D_contribution","gate_coverage"]),
                 "`D_contribution` 的 token 均值即 D；分母包括零 advantage 与未通过 gate 的 token。supported=false 的组不参与主方向预测，不能把它们解释成低风险。\n",
                 "`d=A(e_a−π_old)` 是训练 outcome-consistent direction，包含实际 GRPO normalization 与 invalid penalty。它不是逐动作正确性；C/P/D 原式、中心化坐标及 matched-backend sensitivity 全部归档。\n"]
        records=[]
        for update in committed:
            c=json.loads((a.root/"signals"/f"u{update:04d}"/"committed.json").read_text())
            records.append({"update":update,"Adam_steps":c["optimizer_steps"],"Adam_before":c["adam_step_before"],"Adam_after":c["adam_step_after"],"parameter_L2":c["raw_parameter_delta_l2"],"live_rows":c["live_old_and_new_rows"]})
        text += [table(pd.DataFrame(records),list(records[0]))]
        text += ["### 指标覆盖与解释\n",
                 "已计算逐token（保留decision ID）的 `P_int`、`C_upd`、门控/未门控D、`u_original/u_control/delta` 的范数、KL/JS，以及8/16/24/32层activation interaction范数；原始FP32参数差分另行归档。正文P是预定token等权聚合，不是每个环境step独立gold。可由token归档重建decision等权读出，但不临时选择更有利的聚合。\n",
                 "P保留正负方向；D是负向部分的门控聚合，不能以D低推断效用上升。C检查update fidelity，不是效用方向标签。action/activation/parameter范数没有符号；同一次update的全局参数范数对所有Skill相同。JVP与线性化误差尚未完成，不能用参数范数冒充。\n",
                 table(primary,["global_update","skill_id","C_upd","direction_coverage","gate_coverage"])]
    comparisons=[]
    if not u.empty:
        u.to_parquet(metrics/"utility_units.parquet",index=False)
        g.to_parquet(metrics/"utility_games.parquet",index=False)
        m.to_parquet(metrics/"anchor_margins.parquet",index=False)
        text += ["## 4. 三臂效用与变化方向\n",
                 "数值使用 success-rate 单位；例如0.05为5个百分点。CI按相同 game 的 old/new paired difference bootstrap 10,000次；预定方向阈值为±5pp。\n",
                 table(u[(u.control=="placebo")&(u.phase=="all")],["global_update","skill_id","utility_old","utility_new","delta_utility","ci_low","ci_high","direction"]),
                 "所有 initial/early/middle/late 分层、O−NULL 与 O−PLACEBO 对照保存在机器可读表，不能把多条 anchor 当作独立 update。\n"]
        text.append("这里 stable 是本批固定 continuation seed 下的 game bootstrap 分类；若所有观测差值均为0，重采样CI会退化为[0,0]，这不证明对其他生成随机种子或真实期望效用也完全不变。\n")
        text += ["### Policy 与对照臂的双轴变化\n",
                 table(u[(u.control=="placebo")&(u.phase=="all")],
                       ["global_update","skill_id","original_old","original_new","control_old","control_new","delta_original","delta_control"]),
                 "这里 control 指 PLACEBO，不是完全 skill-free Policy。NULL 也只移除目标 Skill 的 payload，其他 Skill 仍保留；两者均不能冒充全库 no-skill baseline。`ΔM_sem=ΔORIGINAL−ΔPLACEBO`。\n"]
        merged=u.merge(f,on=["global_update","control","skill_id","phase"],validate="one_to_one")
        merged.to_parquet(metrics/"signal_utility_pairs.parquet",index=False)
        primary=merged[(merged.control=="placebo")&(merged.phase=="all")]
        for scope,qs in (("all_updates_exploratory",primary),("heldout_updates",primary[primary.global_update.isin(config["test_updates"])])):
            q=qs[qs.supported]
            if q.empty:continue
            comparisons.append({"scope":scope,"units":len(q),"updates":q.global_update.nunique(),
                                "P_int_vs_signed_delta_spearman":corr(q.P_int,q.delta_utility),
                                "D_vs_negative_delta_spearman":corr(q.D_contribution,-q.delta_utility),
                                "negative_point_count":int(q.negative_point_label.sum()),
                                "negative_prevalence_AP_baseline":float(q.negative_point_label.mean()),
                                "direct_D_negative_AUPRC":float(average_precision_score(q.negative_point_label,q.D_contribution)) if q.negative_point_label.nunique()==2 else None})
        atomic_write_json(metrics/"direct_associations.json",{"created_at":utc_now(),"results":comparisons,
                                                              "interpretation":"exploratory point estimates; few update clusters; not evidence of seed generalization"})
        text += ["## 5. 关联与前瞻预测\n",table(pd.DataFrame(comparisons),list(comparisons[0])) if comparisons else "有效支持不足，暂不计算关联。\n",
                 "上表是探索性点估计，不能仅凭相关系数方向正确就宣称预测成立。跨 update 稳定性及增量预测需要结合后续表、覆盖率和极小的测试 update 数判断。\n"]
        from phase2.utilities import feature_frame
        baseline_rows=[]
        for update in sorted(primary.global_update.unique()):
            base=feature_frame(a.root,int(update),m)
            baseline_rows.append(base[["global_update","skill_id","old_margin","old_margin_se","raw_parameter_delta_l2","train_success"]])
        enriched=primary.merge(pd.concat(baseline_rows,ignore_index=True),on=["global_update","skill_id"],validate="one_to_one")
        signed_audit=summarize(enriched,config["test_updates"])
        atomic_write_json(metrics/"signed_analysis_v2.json",{"created_at":utc_now(),**signed_audit})
        associations=pd.DataFrame(signed_audit["associations"])
        associations.to_parquet(metrics/"all_signal_associations.parquet",index=False)
        text += ["### 5.1 连续方向分析补充（9月10日经用户确认）\n",
                 "本补充在已观察U31–U33、尚未取得U34–U36测试gold时归档：`analysis_amendment_v2.json`。既有公式、门控、Skill/anchor集合、预测模型特征、开发/隔离/测试划分不变。新增全部已采集指标的描述性比较和简单基线，不按这里的相关性选择特征。8月19日idea文件仍含早期harmful-flip主目标文字；本报告沿用后续已确认的连续语义效用目标，不声称验证了旧版全部强主张。\n",
                 "局部训练reward方向上的P与独立长期效用并无逐例同号保证。排序关联、逐例同号、时间外预测是三个不同层次；尤其不能把有正相关系数写成方向已预测成功。\n",
                 table(associations,["scope","signal","units","updates","rho_signed_delta","rho_absolute_delta"]),
                 "上述所有相关系数仅为探索性点估计。`rho_signed_delta`均对应ΔM；D的下降方向相关系数应取相反数。范数与signed ΔM的相关性本身不赋予范数正负方向。多指标、小样本、同一Skill重复观测及相邻差分共享端点均限制解释；不计算把12个单元当12个独立updates的显著性。\n",
                 table(pd.DataFrame(signed_audit["direction_counts"]),["scope","units","positive_points","negative_points","zero_points","nonzero_point_coverage","conditional_sign_accuracy","always_positive_conditional_accuracy","ci_excludes_zero_points"]),
                 "P同号率只对非零效用点估计且P非零的单元计算，必须结合覆盖率；不是可靠方向标签。CI排除0数量只是额外可信度描述，不取代预定±5pp标签。零变化、CI不确定样本全部保留在主连续MAE中，不能删掉或临时更换无变化Skill。\n",
                 table(pd.DataFrame(signed_audit["by_update"]),["update","units","P_vs_signed_delta","D_vs_decline","positive_points","negative_points","zero_points","conditional_sign_accuracy"]),
                 "逐update表用于检查关联是否被单次更新主导，4个Skill的相关性仍极不稳定。四个冻结Skill可用于最小可行性验证，不能据此声称全面Skill或跨seed泛化。\n"]
        text.append("### 5.2 时间外预测比较\n")
        pred=[]
        for file in sorted((a.root/"predictions").glob("u*.json")):
            x=json.loads(file.read_text())
            if x["global_update"] in config["test_updates"]:
                pred.extend({"global_update":x["global_update"],**r} for r in x["predictions"])
        performance=[]
        if pred:
            pv=pd.DataFrame(pred).merge(u[(u.control=="placebo")&(u.phase=="all")],on=["global_update","skill_id"],validate="one_to_one")
            pv=pv[pv.supported]
            pv.to_parquet(metrics/"prospective_prediction_results.parquet",index=False)
            for name in ("zero","dev_mean","old_margin","unsigned","activation","signed","opposition"):
                col=f"predicted_delta_{name}"
                if col not in pv:continue
                q=pv.dropna(subset=[col])
                if q.empty:continue
                counts=direction_counts(q,col)
                r={"model":name,"test_units":len(q),"test_updates":int(q.global_update.nunique()),
                   "signed_MAE":float(mean_absolute_error(q.delta_utility,q[col])),
                   "rho_signed_delta":corr(q[col],q.delta_utility),
                   "prediction_coverage":counts["prediction_coverage"],
                   "conditional_sign_accuracy":counts["conditional_sign_accuracy"],
                   "ci_excludes_zero_sign_accuracy":counts["ci_excludes_zero_sign_accuracy"]}
                prob=f"negative_probability_{name}"
                if prob in q and q[prob].notna().all():
                    r["negative_Brier"]=float(brier_score_loss(q.negative_point_label,q[prob]))
                    r["negative_AUPRC"]=float(average_precision_score(q.negative_point_label,q[prob])) if q.negative_point_label.nunique()==2 else None
                performance.append(r)
            if "predicted_direction_dev_majority" in pv:
                majority=direction_counts(pv,"predicted_direction_dev_majority")
                atomic_write_json(metrics/"development_majority_direction_baseline.json",majority)
                text.append("开发集多数方向（仅U31/U32非零变化、平局abstain）的时间外方向基线："+
                            json.dumps(majority,ensure_ascii=False)+"。该±1方向标签不是效用幅度预测，不参与MAE。\n")
            atomic_write_json(metrics/"prospective_performance.json",{"results":performance,"training_updates":[31,32],"test_updates":[34,35,36]})
            if performance:text.append(table(pd.DataFrame(performance),list(pd.DataFrame(performance).columns)))
        text += ["方向分类首先报告上述 CI 支持的 positive/negative/stable/uncertain。AUPRC/Brier 的负事件仅指点估计 ΔM<−0.05，并不把这些事件称作可靠下降；continuous signed MAE 是不依赖二值标签的主要预测误差。无下降事件时不计算无意义的下降分类AUPRC。新增zero-delta、development-mean基线只作参照，既有模型不重选或调参。\n",
                 "拟合器只使用 U31/U32，最多8个 Skill-update 单元；统一 StandardScaler、Ridge α=1 / Logistic C=1，不调参。支持不足6个时只保留 direct scores，不拟合。测试只有3个 updates，不能将最多12个 Skill-update 单元当成12次独立训练重复。\n"]
        if len(primary):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
            for skill,q in primary.groupby("skill_id"):
                axes[0].scatter(q.P_int,q.delta_utility*100,label=skill)
                axes[1].scatter(q.D_contribution,-q.delta_utility*100,label=skill)
            axes[0].set(xlabel="Signed P_int (PLACEBO control)",ylabel="Semantic utility change (pp)")
            axes[1].set(xlabel="Reward-opposing D (PLACEBO control)",ylabel="Semantic utility decline (pp)")
            for ax in axes:
                ax.axhline(0,color="gray",lw=.7);ax.legend(fontsize=8);ax.grid(alpha=.2)
            fig.suptitle("Exploratory single-seed measurements; all recorded Skill pairs")
            plots=a.root/"plots";plots.mkdir(exist_ok=True)
            fig.savefig(plots/"direction_associations.png",dpi=200)
            fig.savefig(plots/"direction_associations.pdf")
            plt.close(fig)
            text.append(f"\n![方向关联]({plots/'direction_associations.png'})\n")
    text += ["## 6. 尚不能据此声称的结论\n",
             "JVP 的可微实现、额外随机/打乱 reward 更新对照、更多训练 seed 和完整预算匹配基线仍需分别验证；未运行的项目不会列成已完成。第一批以精确 P/D、参数和 activation、三臂 gold 及时间外方向分析优先。\n",
             "## 7. 归档\n",
             f"运行根目录：`{a.root}`。执行协议：`protocol.json`；训练数据：`batches/`；真实新旧概率：`old_logprobs/`、`new_logprobs/`；信号：`signals/`；预测锁定记录：`predictions/`；完整三臂轨迹：`evaluations/`；统计：`metrics/`。\n",
             "Phase1 原三-seed 报告和权重未修改。当前报告按完整 endpoint 自动更新，缺失结果保持待完成。\n"]
    temporary=report.with_suffix(".md.partial")
    temporary.write_text("\n".join(text),encoding="utf-8")
    temporary.replace(report)
    print(report,flush=True)


if __name__=="__main__":main()
