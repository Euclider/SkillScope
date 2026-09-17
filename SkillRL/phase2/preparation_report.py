"""Report NEW pre-update coverage/margins; never infer a change without endpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase2.utilities import margins, mean_game, paired_bootstrap, read_evaluations


def summarize_margins(m, repetitions=10000):
    rows = []
    rng = np.random.default_rng(20260912)
    for (purpose, skill, context), group in m.groupby(["purpose", "skill_id", "context_id"]):
        for phase in ("all", "initial", "early", "middle", "late"):
            q = group if phase == "all" else group[group.phase == phase]
            if q.empty: continue
            for control in ("placebo", "null"):
                q = q.copy()
                q["delta"] = q[f"M_{control}"]  # paired-bootstrap accepts a generic paired contrast.
                draws = paired_bootstrap(q, rng, repetitions)
                low, high = np.quantile(draws, [.025, .975])
                rows.append({"purpose": purpose, "skill_id": skill, "context_id": context, "phase": phase,
                             "control": control, "anchor_count": q.anchor_id.nunique(), "game_count": q.game_id.nunique(),
                             "continuation_repeats": q.continuation_seed.nunique(),
                             "original_success": mean_game(q, "original"), "control_success": mean_game(q, control),
                             "semantic_or_total_margin": mean_game(q, f"M_{control}"), "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    config = json.loads((root/"protocol.json").read_text())
    from phase2.queued_preparation import validate_baseline
    validate_baseline(config, root, config["parent_update"])
    ev = read_evaluations(root)
    if ev.empty or set(ev["update"]) != {config["parent_update"]}:
        raise ValueError("Exactly one completed pre-update endpoint is required")
    m = margins(ev)
    results = summarize_margins(m, config["evaluation"]["bootstrap_repetitions"])
    directory = root/"reports"
    directory.mkdir(exist_ok=True)
    results.to_csv(directory/"baseline_margins.csv", index=False)
    m.to_parquet(directory/"paired_anchor_margins.parquet", index=False)
    support = json.loads((root/"support/anchor_sets.proposal.json").read_text())
    coverage = pd.DataFrame(support["coverage"])
    sources = [json.loads(line) for path in sorted((root/"coverage").glob("shard-*.jsonl"))
               for line in path.read_text().splitlines() if line.strip()]
    summary = {"created_at": utc_now(), "preupdate_only": True, "contains_delta_utility": False,
               "coverage_trajectories": len(sources), "coverage_unique_games": len({x["game_id"] for x in sources}),
               "coverage_mean_unique_skills": float(np.mean([x["unique_skill_count"] for x in sources])),
               "coverage_mean_steps": float(np.mean([x["trajectory_length"] for x in sources])),
               "coverage_success": float(np.mean([x["success"] for x in sources])),
               "supported_skills": len(support["anchor_sets"]), "baseline_suffixes": len(ev),
               "source_hashes": {str(path): sha256_file(path) for path in [root/"protocol.json", root/"support/anchor_sets.proposal.json"]}}
    atomic_write_json(directory/"baseline_summary.json", summary)
    main_table = results[(results.purpose == "gold") & (results.phase == "all") & (results.control == "placebo")].copy()
    for column in ("original_success", "control_success", "semantic_or_total_margin", "ci_low", "ci_high"):
        main_table[column] *= 100
    text = ("# U35 新支持集覆盖与多重复效用基线\n\n"
            "这是一份 pre-update 准备报告，不包含新 RL 窗口、ΔM 或排序预测结论；旧实验保持不变。\n\n"
            f"自然轨迹 {len(sources)} 条，覆盖 {summary['coverage_unique_games']} 个 game；"
            f"平均每条调用 {summary['coverage_mean_unique_skills']:.3f} 种 Skill。\n\n"
            "## 自然支持审计\n\n"+coverage.to_markdown(index=False)+"\n\n"
            "## 独立 gold 基线（成功率及效用均为百分点）\n\n"+main_table.to_markdown(index=False, floatfmt=".3f")+"\n\n"
            "全部 evidence/gold、ORIGINAL−PLACEBO / ORIGINAL−NULL 和各阶段数值见 baseline_margins.csv。"
            "多个 continuation repeats 在 anchor/game 内配对，不作为独立 games 或独立更新窗口。\n\n"
            "后续主分析为同一更新窗口内的 Skill top-k 排序；不以回归 MAE 作为必要成功条件。"
            "新窗口预算、完整训练状态和存储策略仍需单独冻结，当前队列不会自动训练。\n")
    (directory/"baseline-report.md").write_text(text)
    queue = json.loads((root/"queue.json").read_text())
    if queue.get("report_path"):
        report = Path(queue["report_path"])
        marker = "<!-- completed-preupdate-baseline -->"
        existing = report.read_text() if report.exists() else ""
        if marker not in existing:
            with report.open("a") as handle:
                handle.write("\n\n"+marker+"\n\n"+text.replace("# U35 新支持集覆盖与多重复效用基线", "## 已完成：U35 新支持集覆盖与多重复效用基线", 1))
    print(directory/"baseline-report.md")


if __name__ == "__main__":
    main()
