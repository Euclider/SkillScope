"""Read-only analysis of completed Phase2 endpoints; write a separate report snapshot.

No training, evaluation, prediction fitting, or canonical metric files are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from phase1.archive import atomic_write_json
from phase2.utilities import read_evaluations


def records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=15))


def correlation(x, y):
    return float(spearmanr(x, y).statistic) if x.nunique() > 1 and y.nunique() > 1 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-update", type=int, default=35)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Snapshot already exists: {args.output}")
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    sources = set()

    def read_json(path):
        sources.add(path)
        return json.loads(path.read_text())

    def read_table(name):
        path = root / "metrics" / f"{name}.parquet"
        sources.add(path)
        frame = pd.read_parquet(path)
        return frame[frame.global_update <= args.max_update] if "global_update" in frame else frame

    protocol = read_json(root / "protocol.json")
    evaluations = read_evaluations(root)
    evaluations = evaluations[evaluations["update"] <= args.max_update].copy()
    endpoints = sorted(int(x) for x in evaluations["update"].unique())
    assert endpoints == list(range(30, args.max_update + 1)), endpoints
    for update in endpoints:
        for pattern in ("shard-*.jsonl", "shard-*-complete.json"):
            sources.update((root / "evaluations" / f"u{update:04d}").glob(pattern))
    assert evaluations.prefix_replay_verified.all()
    assert not evaluations.duplicated(["update", "purpose", "skill_id", "anchor_id", "arm"]).any()
    assert evaluations.groupby("update").size().eq(1200).all()
    assert evaluations.trajectory_path.map(lambda x: Path(x).is_file()).all()
    assert evaluations.prefix_return.eq(0).all()

    units = read_table("utility_units")
    signals = read_table("signal_utility_pairs")
    predictions = read_table("prospective_prediction_results")
    semantic = signals[(signals.control == "placebo") & (signals.phase == "all")].copy()
    supported = semantic[semantic.supported].copy()
    scopes = {"all_completed_exploratory": supported,
              "time_holdout": supported[supported.global_update.isin(protocol["test_updates"])]}
    direct = []
    for scope, frame in scopes.items():
        nz = frame.delta_utility.abs() > 1e-12
        neg = frame.delta_utility < -.05
        direct.append({"scope": scope, "units": len(frame), "updates": frame.global_update.nunique(),
                       "rho_P_signed_delta": correlation(frame.P_int, frame.delta_utility),
                       "rho_D_decline": correlation(frame.D_contribution, -frame.delta_utility),
                       "P_sign_accuracy_nonzero": float((np.sign(frame.loc[nz, "P_int"]) == np.sign(frame.loc[nz, "delta_utility"])).mean()),
                       "nonzero_delta_units": int(nz.sum()), "negative_gt_5pp_events": int(neg.sum()),
                       "D_average_precision_negative_gt_5pp": float(average_precision_score(neg, frame.D_contribution)),
                       "negative_gt_5pp_prevalence": float(neg.mean())})
    per_update = []
    for update, frame in supported.groupby("global_update"):
        per_update.append({"update": int(update), "units": len(frame),
                           "rho_P_signed_delta": correlation(frame.P_int, frame.delta_utility),
                           "rho_D_decline": correlation(frame.D_contribution, -frame.delta_utility)})
    performance = []
    for column in predictions:
        if not column.startswith("predicted_delta_"):
            continue
        nz = predictions.delta_utility.abs() > 1e-12
        performance.append({"model": column.removeprefix("predicted_delta_"),
                            "test_units": len(predictions),
                            "mae_pp": float((predictions[column] - predictions.delta_utility).abs().mean() * 100),
                            "nonzero_prediction_fraction": float((predictions[column].abs() > 1e-12).mean()),
                            "sign_accuracy_nonzero_gold": float((np.sign(predictions.loc[nz, column]) == np.sign(predictions.loc[nz, "delta_utility"])).mean())})

    gold = evaluations[evaluations.purpose == "gold"].copy()
    keys = ["update", "skill_id", "anchor_id", "game_id", "phase", "trigger_step"]
    paired = gold[gold.arm == "original"].merge(gold[gold.arm == "placebo"], on=keys, suffixes=("_O", "_P"), validate="one_to_one")
    paired["first_action_diff"] = paired.first_action_O != paired.first_action_P
    paired["suffix_diff"] = paired.action_sequence_O != paired.action_sequence_P
    paired["reward_diff"] = paired.success_O != paired.success_P
    behaviors = []
    for skill, frame in [("all", paired), *list(paired.groupby("skill_id"))]:
        behaviors.append({"skill": skill, "anchor_endpoint_pairs": len(frame),
                          **{key: float(frame[key].mean()) for key in ["first_action_diff", "suffix_diff", "reward_diff"]}})

    margins = read_table("anchor_margins")
    margins = margins[(margins["update"] <= args.max_update) & (margins.purpose == "gold")]
    endpoint_utility = margins.groupby(["update", "skill_id", "game_id"])[["original", "placebo", "null", "M_placebo", "M_null"]].mean().groupby(["update", "skill_id"]).mean().reset_index()
    phase_units = units[(units.control == "placebo") & (units.phase != "all")]
    delta_anchors = margins.merge(margins.assign(update=margins["update"] + 1), on=["update", "skill_id", "anchor_id", "game_id", "phase", "trigger_step"], suffixes=("_new", "_old"))
    delta_anchors["delta"] = delta_anchors.M_placebo_new - delta_anchors.M_placebo_old
    anchor_delta_distribution = delta_anchors.groupby(["update", "skill_id"]).delta.agg(
        count="size", positive=lambda x: int((x > 0).sum()), negative=lambda x: int((x < 0).sum()), zero=lambda x: int((x == 0).sum())).reset_index()
    anchors = []
    for skill in protocol["evaluation"]["skills"]:
        path = repo / protocol["evaluation"]["anchors_dir"] / f"{skill}.jsonl"
        sources.add(path)
        frame = margins[(margins["update"] == 30) & (margins.skill_id == skill)]
        anchors.append({"skill": skill, "anchors": len(frame), "games": frame.game_id.nunique(),
                        "min_step": int(frame.trigger_step.min()), "max_step": int(frame.trigger_step.max()),
                        "phases": frame.phase.value_counts().to_dict()})
    training = []
    for update in endpoints[1:]:
        committed = read_json(root / f"signals/u{update:04d}/committed.json")
        prediction = read_json(root / f"predictions/u{update:04d}.json")
        parameter = read_json(root / f"signals/u{update:04d}/parameter_delta.json")
        metrics = read_json(repo / f"artifacts/training_steps/phase2-s303-fast-u{update}/step-{update:06d}.json")["metrics"]
        feature_path = root / f"signals/u{update:04d}/skill_context_features.parquet"
        sources.add(feature_path)
        feature_hash = hashlib.sha256(feature_path.read_bytes()).hexdigest()
        first_eval = evaluations.loc[evaluations["update"] == update, "created_at"].min()
        lock_ok = committed["created_at"] <= prediction["created_at"] < first_eval
        assert lock_ok and feature_hash == committed["features_sha256"] == prediction["features_sha256"]
        assert not prediction["target_gold_read"] and not committed["gold_read"]
        training.append({"update": update, "rollouts": 32, "adam_before": committed["adam_step_before"],
                         "adam_after": committed["adam_step_after"], "adam_steps": committed["optimizer_steps"],
                         "parameter_delta_l2": parameter["delta_l2"], "relative_parameter_delta_l2": parameter["relative_delta_l2"],
                         "train_rollout_success": metrics["episode/success_rate"],
                         "signals_locked_at": committed["created_at"], "predictions_locked_at": prediction["created_at"],
                         "first_target_evaluation_record": first_eval, "lock_and_hash_audit": lock_ok})

    sources.update(repo / "phase2" / name for name in ["discussion_snapshot.py", "direction.py", "aggregate.py", "utilities.py", "forecast.py", "evaluate.py"])
    sources.add(root / "analysis_amendment_v2.json")
    phase1 = repo.parent / "2026-09-04-qwen35-clean-all-skill-three-seed-utility-results.md"
    sources.add(phase1)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
               "max_update": args.max_update, "completed_endpoints": endpoints,
               "complete_suffixes": len(evaluations), "gold_suffixes": len(gold),
               "prefix_replay_verified": True, "trajectory_files_exist": True,
               "candidate_skill_update_units": len(semantic), "supported_units": len(supported),
               "anchors": anchors, "training": training, "endpoint_utility": records(endpoint_utility),
               "semantic_pairs": records(semantic), "phase_semantic_pairs": records(phase_units),
               "null_pairs": records(units[(units.control == "null") & (units.phase == "all")]),
               "direct_associations": direct, "per_update_correlations": per_update,
               "prospective_performance": performance, "locked_test_predictions": records(predictions),
               "behavior_pairs": behaviors, "anchor_delta_distribution": records(anchor_delta_distribution),
               "gold_original_suffix_mean_distinct_skills": float(gold.loc[gold.arm == "original", "selected_skill_ids"].map(len).mean()),
               "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}}
    atomic_write_json(args.output, summary)
    print(json.dumps({key: summary[key] for key in ["complete_suffixes", "gold_suffixes", "supported_units", "direct_associations", "prospective_performance", "behavior_pairs", "gold_original_suffix_mean_distinct_skills"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
