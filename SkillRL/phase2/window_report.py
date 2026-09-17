"""Archive completed wider-window labels and frozen predictions in a new cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase2.protocol import signal_directory, validate_extended
from phase2.utilities import read_evaluations, window_units


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    a=p.parse_args()
    config=json.loads((a.root/"protocol.json").read_text())
    validate_extended(config,Path(__file__).resolve().parents[1])
    units,games,margins=window_units(a.root)
    directory=a.root/"window_metrics";directory.mkdir(exist_ok=True)
    units.to_parquet(directory/"utility_units.parquet",index=False)
    games.to_parquet(directory/"utility_games.parquet",index=False)
    margins.to_parquet(directory/"anchor_margins.parquet",index=False)
    text=["# Extended Phase2：按冻结窗口汇总\n",f"路径：`{config['rl_path_id']}`；更新时间：{utc_now()}。\n",
          "窗口方向为起始 batch 的 reward direction 在窗口端点 interaction 上的投影，不是逐 update D 的相加。all 与各阶段单独分析，不将重复分层当成独立样本。\n"]
    pairs=[];predictions=[];rankings=[];ranking_support=[];ranking_rows=[]
    for window in config["windows"]:
        if units.empty:break
        labels=units[(units.start_update==window["start"])&(units.global_update==window["end"])]
        if labels.empty:continue
        signal=signal_directory(a.root,window["end"],window["start"])
        keys=["start_update","global_update","control","skill_id","context_id","phase"]
        if (signal/"committed.json").exists():
            features=pd.read_parquet(signal/"skill_context_features.parquet")
            pairs.append(labels.merge(features[keys+[c for c in features if c not in labels]],on=keys,how="left",validate="one_to_one"))
        path=signal/"prediction.json"
        if path.exists():
            frozen=json.loads(path.read_text())
            prediction=pd.DataFrame(frozen["predictions"])
            prediction["start_update"]=window["start"];prediction["global_update"]=window["end"];prediction["control"]="placebo"
            predictions.append(prediction.merge(labels[labels.control=="placebo"],on=keys,validate="one_to_one"))
            if config.get("ranking") and frozen.get("ranking_scores"):
                if frozen["protocol_sha256"]!=sha256_file(a.root/"protocol.json"):
                    raise ValueError("Ranking protocol changed after scores were locked")
                if frozen["ranking_plan"]!=config["ranking"] or frozen["target_gold_read"]:
                    raise ValueError("Ranking was not frozen under the registered plan")
                from phase2.ranking import evaluate_snapshot
                metrics,support,rows=evaluate_snapshot(frozen["ranking_scores"],labels[labels.control=="placebo"],config["ranking"])
                identity={"start_update":window["start"],"global_update":window["end"],"window_role":window["role"]}
                rankings.append(metrics.assign(**identity))
                ranking_support.extend([{**identity,**row} for row in support])
                ranking_rows.append(rows.assign(**identity))
    performances=[]
    if pairs:pd.concat(pairs,ignore_index=True).assign(rl_path_id=config["rl_path_id"]).to_parquet(directory/"signal_utility_pairs.parquet",index=False)
    if predictions:
        results=pd.concat(predictions,ignore_index=True)
        results.to_parquet(directory/"prospective_predictions.parquet",index=False)
        for phase,frame in results[(results.window_role=="test")&results.supported].groupby("phase"):
            for name in [c for c in frame if c.startswith("predicted_delta_")]:
                q=frame.dropna(subset=[name,"delta_utility"])
                if not q.empty:
                    performances.append({"phase":phase,"model":name,"units":len(q),
                                         "windows":q[["start_update","global_update"]].drop_duplicates().shape[0],
                                         "mae_pp":float((q[name]-q.delta_utility).abs().mean()*100)})
    if not units.empty:
        show=units[(units.control=="placebo")&(units.phase=="all")].copy()
        for name in ["utility_old","utility_new","delta_utility","ci_low","ci_high"]:show[name]*=100
        columns=["start_update","global_update","window_role","skill_id","context_id","anchor_count","continuation_repeats","delta_utility","ci_low","ci_high"]
        text += ["## 语义效用变化（pp）\n",show[columns].to_markdown(index=False,floatfmt=".4f")+"\n"]
    if rankings:
        ranking_table=pd.concat(rankings,ignore_index=True)
        ranking_table.to_csv(directory/"ranking_metrics.csv",index=False)
        pd.concat(ranking_rows,ignore_index=True).to_csv(directory/"locked_scores_and_gold.csv",index=False)
        atomic_write_json(directory/"ranking_support.json",{"candidate_pools":ranking_support})
        primary=ranking_table[(ranking_table.phase=="all")&(ranking_table.target=="decline")&(ranking_table.threshold==0)]
        columns=["start_update","global_update","context_id","score","n_candidates","events","k","precision_at_k","recall_at_k","captured_change_mass","spearman"]
        text += ["## 主分析：同一窗口内的 Skill top-k 排序\n",
                 "分数不依赖拟合回归器；正负方向和预算先于目标 gold 固定。所有方法使用同一支持候选池；并列按随机打破的期望计分。下表下降是 gold 点估计，不代表每个 Skill 的下降已统计确证。\n",
                 primary[columns].to_markdown(index=False,floatfmt=".4f")+"\n",
                 "下降事件为零时 Recall/下降量覆盖率未定义。完整 CSV 同时包含 >5 pp 阈值、|ΔM| 审计及各阶段；这些不是额外独立窗口。首个单窗口排序只作描述，不声称跨窗口/seed 泛化或排序指标显著性。\n"]
    if performances:text += ["## 辅助：数值预测（非主要成功标准）\n",pd.DataFrame(performances).to_markdown(index=False,floatfmt=".4f")+"\n"]
    atomic_write_json(directory/"summary.json",{"created_at":utc_now(),"rl_path_id":config["rl_path_id"],
        "completed_window_units":len(units),"performance":performances,"ranking_primary":bool(config.get("ranking")),
        "ranking_candidate_pools":ranking_support,"claims_cross_seed_generalization":False})
    (a.root/"reports").mkdir(exist_ok=True)
    (a.root/"reports/window-results.md").write_text("\n".join(text))
    if not units.empty and config.get("report_path"):
        report=Path(config["report_path"])
        marker="<!-- completed-u35-to40-ranking-window -->"
        existing=report.read_text() if report.exists() else ""
        if marker not in existing:
            with report.open("a") as handle:
                handle.write("\n\n"+marker+"\n\n"+"\n".join(text))


if __name__=="__main__":main()
