"""Verify complete endpoint evidence and append a raw-feature results appendix."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import psutil

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase2.protocol import parent_update, signal_directory, validate_extended
from phase2.utilities import read_evaluations


def assert_prospective(prediction, rows, protocol_sha, feature_sha):
    if prediction["target_gold_read"] or prediction["protocol_sha256"] != protocol_sha:
        raise ValueError("Prediction is not bound to the prospective frozen protocol")
    if prediction["features_sha256"] != feature_sha:
        raise ValueError("Frozen feature evidence changed")
    locked = datetime.fromisoformat(prediction["created_at"])
    if not rows or any(datetime.fromisoformat(row["created_at"]) <= locked for row in rows):
        raise ValueError("Every target trajectory must postdate prediction locking")


def finalize(root):
    root = Path(root).resolve()
    config = json.loads((root/"protocol.json").read_text())
    validate_extended(config, Path(__file__).resolve().parents[1])
    evaluations = read_evaluations(root)  # exact IDs, seeds, shard and arm coverage
    endpoints = {u for w in config["windows"] for u in (w["start"], w["end"])}
    if evaluations.empty or set(evaluations["update"]) != endpoints:
        raise ValueError("Not all registered endpoints are complete")
    digest = sha256_file(root/"protocol.json")
    locked, raw_rows = [], []
    for window in config["windows"]:
        directory = signal_directory(root, window["end"], window["start"])
        prediction = json.loads((directory/"prediction.json").read_text())
        target = evaluations[evaluations["update"] == window["end"]].to_dict("records")
        assert_prospective(prediction, target, digest, sha256_file(directory/"skill_context_features.parquet"))
        locked.append({"start": window["start"], "end": window["end"],
                       "prediction_sha256": sha256_file(directory/"prediction.json"),
                       "prediction_locked_at": prediction["created_at"],
                       "first_target_trajectory_at": min(x["created_at"] for x in target)})
        for row in prediction["ranking_scores"]:
            raw_rows.append({"start_update": window["start"], "global_update": window["end"],
                **{k: row[k] for k in ("skill_id", "context_id", "phase", "supported")}, **row["raw_features"]})
    inventory = []
    for row in evaluations[evaluations["update"] > parent_update(config)].to_dict("records"):
        path = Path(row["trajectory_path"])
        trajectory = json.loads(path.read_text())
        if trajectory["trajectory_id"] != row["trajectory_id"] or trajectory["success"] != row["success"]:
            raise ValueError("Trajectory file/index identity mismatch")
        if not isinstance(trajectory["steps"], list): raise ValueError("Missing archived trajectory steps")
        inventory.append({"trajectory_id": row["trajectory_id"], "path": str(path), "sha256": sha256_file(path)})
    subprocess.run([sys.executable, "-m", "phase2.window_report", "--root", str(root)],
                   cwd=Path(__file__).resolve().parents[1], check=True)
    units = pd.read_parquet(root/"window_metrics/utility_units.parquet")
    keys = ["start_update", "global_update", "skill_id", "context_id", "phase"]
    raw = pd.DataFrame(raw_rows).merge(units[units.control == "placebo"], on=keys, how="left", validate="one_to_one")
    raw["gold_evaluation_available"] = raw.delta_utility.notna()
    destination = root/"window_metrics/raw_features_and_semantic_utility.csv"
    raw.to_csv(destination, index=False)
    display = raw[raw.phase == "all"].copy()
    for name in ("utility_old", "utility_new", "delta_utility", "ci_low", "ci_high"): display[name] *= 100
    columns = keys[:2] + ["skill_id", "supported", "utility_old", "utility_new", "delta_utility", "ci_low", "ci_high",
                          "D_contribution", "P_int", "delta_centered_norm", "delta_norm", "C_upd", "gate_coverage"]
    text = ["## 完整评估校验及原始指标对照\n",
            f"全部注册端点评估通过 ID/seed/分片覆盖检查；新端点 {len(inventory):,} 条完整轨迹均已核对索引并留存 SHA-256。目标轨迹生成均晚于对应 prediction 锁定。\n",
            "下表效用与 CI 单位为 pp；D、P、norm 等是原始量，未作排序方向变换。正式下降排序使用 D、−P、+norm；不能把排序 CSV 中的 −P 误读为原始 P。\n",
            display[columns].to_markdown(index=False, floatfmt=".6g") + "\n",
            "包含 initial/early/middle/late 在内的全部信号支持/不支持单元及原始观测量详见 `window_metrics/raw_features_and_semantic_utility.csv`。没有自然评估 anchor 的阶段保留为空标签，并标注 `gold_evaluation_available=false`，不伪造零效用。正式方法比较仍以预先锁定的共同支持池、top-k 下降事件命中及下降量覆盖率为准，不要求精确回归 ΔM。\n",
            "本次仅一个 seed303 延续路径、一个 U35→U40 窗口、原有 4 个 Skill；分层和 continuation 重复不增加独立训练窗口数。点估计方向、CI 与排序表现需分别解读；不能据单窗口宣称跨 seed 泛化或 D 已普遍优于幅度指标。\n"]
    appendix = root/"reports/completion-appendix.md"
    appendix.write_text("\n".join(text))
    if config.get("report_path"):
        report = Path(config["report_path"])
        marker = "<!-- phase2-evaluation-completion-audit-v1 -->"
        if marker not in report.read_text():
            with report.open("a") as handle: handle.write("\n\n"+marker+"\n\n"+"\n".join(text))
    atomic_write_json(root/"evaluation_completion.json", {"completed_at": utc_now(), "protocol_sha256": digest,
        "endpoint_counts": {str(k): int(v) for k, v in evaluations.groupby("update").size().items()},
        "prospective_locks": locked, "new_trajectory_inventory": inventory,
        "raw_features_and_utility_sha256": sha256_file(destination),
        "ranking_metrics_sha256": sha256_file(root/"window_metrics/ranking_metrics.csv"),
        "report_path": config["report_path"], "status": "complete_and_verified"})
    from phase2.complete_report import PLAN, build
    if (root/PLAN).exists(): build(root)


def watch(root):
    while True:
        state = json.loads((root/"status.json").read_text())
        running = False
        try:
            process = psutil.Process(state["pipeline_pid"])
            running = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess: pass
        if not running:
            if state.get("stage") != "extended_cohort_complete":
                raise RuntimeError(f"Pipeline stopped before completion: {state}")
            break
        atomic_write_json(root/"completion_watch_status.json", {"at": utc_now(), "pid": os.getpid(),
            "stage": "waiting_for_registered_evaluations", "pipeline_stage": state["stage"]})
        time.sleep(30)
    finalize(root)
    atomic_write_json(root/"completion_watch_status.json", {"at": utc_now(), "pid": os.getpid(), "stage": "complete_and_verified"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        with (args.root/"logs/completion-watch.log").open("a") as log:
            proc = subprocess.Popen([sys.executable, "-u", "-m", "phase2.finalize_evaluation", "--root", str(args.root), "--watch"],
                cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Completion verification PID {proc.pid}")
        return
    try:
        if args.watch: watch(args.root)
        else: finalize(args.root)
    except Exception as error:
        atomic_write_json(args.root/"completion_watch_status.json", {"at": utc_now(), "pid": os.getpid(), "stage": "stopped_on_error", "error": str(error)})
        raise


if __name__ == "__main__": main()
