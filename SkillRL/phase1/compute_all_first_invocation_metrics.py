#!/usr/bin/env python3
"""Compute per-Skill, per-seed longitudinal utility and generalization metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase1.archive import atomic_write_json
from phase1.first_invocation import PayloadArm

ARMS = [arm.value for arm in PayloadArm]


def load_records(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return pd.DataFrame(rows)


def cluster_interval(block: pd.DataFrame, column: str, resamples: int, rng: np.random.Generator) -> tuple[float, float, float]:
    values = block.groupby("game_id", dropna=False)[column].mean().to_numpy(float)
    estimate = float(values.mean())
    if len(values) == 1:
        return estimate, estimate, estimate
    samples = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return estimate, float(lower), float(upper)


def hierarchical_seed_game_interval(block: pd.DataFrame, column: str, resamples: int, rng: np.random.Generator) -> tuple[float, float, float]:
    seed_games = {
        int(seed): values.groupby("game_id", dropna=False)[column].mean().to_numpy(float)
        for seed, values in block.groupby("rl_seed")
    }
    seeds = np.array(sorted(seed_games), dtype=int)
    estimate = float(np.mean([values.mean() for values in seed_games.values()]))
    samples = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for seed in sampled_seeds:
            games = seed_games[int(seed)]
            seed_means.append(float(rng.choice(games, size=len(games), replace=True).mean()))
        samples[index] = float(np.mean(seed_means))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return estimate, float(lower), float(upper)


def first_divergence(left: list[str], right: list[str]) -> int | None:
    for index, (left_action, right_action) in enumerate(zip(left, right)):
        if left_action != right_action:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def paired_effects(records: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "checkpoint_id", "rl_seed", "global_update", "skill_id", "anchor_id", "state_id",
        "source_eval_seed", "environment_seed", "game_id", "context_id", "trigger_step", "trigger_phase",
    ]
    duplicates = records.duplicated([*identity, "payload_arm"], keep=False)
    if duplicates.any():
        raise ValueError("duplicate checkpoint/Skill/anchor/arm records")
    for _, block in records.groupby(identity, dropna=False):
        if set(block["payload_arm"]) != set(ARMS):
            raise ValueError("incomplete original/placebo/null branch")
    values = [
        "suffix_return", "success", "invalid_action_count", "first_action", "action_sequence",
        "target_selected_count", "target_payload_injection_count", "suffix_trajectory_length",
    ]
    wide = records.pivot(index=identity, columns="payload_arm", values=values)
    wide.columns = [f"{metric}_{arm}" for metric, arm in wide.columns]
    wide = wide.reset_index()
    wide["semantic_utility"] = wide["suffix_return_original"] - wide["suffix_return_placebo"]
    wide["total_utility"] = wide["suffix_return_original"] - wide["suffix_return_null"]
    wide["prompt_nuisance"] = wide["suffix_return_placebo"] - wide["suffix_return_null"]
    wide["semantic_success_utility"] = wide["success_original"].astype(float) - wide["success_placebo"].astype(float)
    wide["total_success_utility"] = wide["success_original"].astype(float) - wide["success_null"].astype(float)
    wide["original_placebo_first_action_flip"] = wide["first_action_original"] != wide["first_action_placebo"]
    wide["original_null_first_action_flip"] = wide["first_action_original"] != wide["first_action_null"]
    wide["original_placebo_suffix_divergence"] = [
        left != right for left, right in zip(wide["action_sequence_original"], wide["action_sequence_placebo"])
    ]
    wide["original_null_suffix_divergence"] = [
        left != right for left, right in zip(wide["action_sequence_original"], wide["action_sequence_null"])
    ]
    wide["original_placebo_first_divergence_offset"] = [
        first_divergence(left, right) for left, right in zip(wide["action_sequence_original"], wide["action_sequence_placebo"])
    ]
    return wide


def summarize_block(block: pd.DataFrame, resamples: int, rng: np.random.Generator) -> dict[str, Any]:
    result: dict[str, Any] = {
        "matched_anchors": len(block),
        "distinct_games": int(block["game_id"].nunique()),
        "distinct_states": int(block["state_id"].nunique()),
    }
    for column in ("semantic_utility", "total_utility", "prompt_nuisance"):
        estimate, lower, upper = cluster_interval(block, column, resamples, rng)
        result[column] = estimate
        result[f"{column}_lcb"] = lower
        result[f"{column}_ucb"] = upper
        result[f"{column}_positive_anchors"] = int((block[column] > 0).sum())
        result[f"{column}_negative_anchors"] = int((block[column] < 0).sum())
        result[f"{column}_zero_anchors"] = int((block[column] == 0).sum())
    for arm in ARMS:
        result[f"success_rate_{arm}"] = float(block[f"success_{arm}"].astype(float).mean())
        result[f"success_count_{arm}"] = int(block[f"success_{arm}"].astype(bool).sum())
        result[f"mean_invalid_actions_{arm}"] = float(block[f"invalid_action_count_{arm}"].mean())
    for column in (
        "original_placebo_first_action_flip", "original_null_first_action_flip",
        "original_placebo_suffix_divergence", "original_null_suffix_divergence",
    ):
        result[f"{column}_rate"] = float(block[column].mean())
    divergence = block["original_placebo_first_divergence_offset"].dropna()
    result["mean_original_placebo_first_divergence_offset"] = float(divergence.mean()) if len(divergence) else None
    return result


def checkpoint_skill_metrics(paired: pd.DataFrame, resamples: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for keys, block in paired.groupby(["checkpoint_id", "rl_seed", "global_update", "skill_id"], sort=False):
        checkpoint, rl_seed, update, skill = keys
        rows.append({
            "checkpoint_id": checkpoint,
            "rl_seed": int(rl_seed),
            "global_update": int(update),
            "skill_id": skill,
            **summarize_block(block, resamples, rng),
        })
    return pd.DataFrame(rows)


def checkpoint_phase_metrics(paired: pd.DataFrame, resamples: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for keys, block in paired.groupby(["checkpoint_id", "rl_seed", "global_update", "skill_id", "trigger_phase"], sort=False):
        checkpoint, rl_seed, update, skill, phase = keys
        rows.append({
            "checkpoint_id": checkpoint,
            "rl_seed": int(rl_seed),
            "global_update": int(update),
            "skill_id": skill,
            "trigger_phase": phase,
            **summarize_block(block, resamples, rng),
        })
    return pd.DataFrame(rows)


def longitudinal_effects(paired: pd.DataFrame, base_checkpoint: str) -> pd.DataFrame:
    identity = [
        "skill_id", "anchor_id", "state_id", "source_eval_seed", "environment_seed", "game_id",
        "context_id", "trigger_step", "trigger_phase",
    ]
    base = paired[paired["checkpoint_id"] == base_checkpoint][identity + ["semantic_utility", "total_utility", "prompt_nuisance"]]
    if base.duplicated(identity).any():
        raise ValueError("base contains duplicate anchor identities")
    base = base.rename(columns={name: f"base_{name}" for name in ("semantic_utility", "total_utility", "prompt_nuisance")})
    posts = paired[paired["checkpoint_id"] != base_checkpoint]
    merged = posts.merge(base, on=identity, validate="many_to_one")
    if len(merged) != len(posts):
        raise ValueError("post checkpoints do not exactly match the fixed B0 anchor support")
    for utility in ("semantic_utility", "total_utility", "prompt_nuisance"):
        merged[f"delta_{utility}"] = merged[utility] - merged[f"base_{utility}"]
    merged["point_harmful_semantic_flip"] = (merged["base_semantic_utility"] > 0) & (merged["semantic_utility"] < 0)
    return merged


def longitudinal_metrics(longitudinal: pd.DataFrame, resamples: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for keys, block in longitudinal.groupby(["checkpoint_id", "rl_seed", "global_update", "skill_id"], sort=False):
        checkpoint, rl_seed, update, skill = keys
        row: dict[str, Any] = {
            "checkpoint_id": checkpoint,
            "rl_seed": int(rl_seed),
            "global_update": int(update),
            "skill_id": skill,
            "matched_anchors": len(block),
            "distinct_games": int(block["game_id"].nunique()),
        }
        for utility in ("semantic_utility", "total_utility", "prompt_nuisance"):
            column = f"delta_{utility}"
            estimate, lower, upper = cluster_interval(block, column, resamples, rng)
            row[column] = estimate
            row[f"{column}_lcb"] = lower
            row[f"{column}_ucb"] = upper
            row[f"{column}_positive_anchors"] = int((block[column] > 0).sum())
            row[f"{column}_negative_anchors"] = int((block[column] < 0).sum())
            row[f"{column}_zero_anchors"] = int((block[column] == 0).sum())
        row["point_harmful_semantic_flip_rate"] = float(block["point_harmful_semantic_flip"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def longitudinal_phase_metrics(longitudinal: pd.DataFrame, resamples: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    group = ["checkpoint_id", "rl_seed", "global_update", "skill_id", "trigger_phase"]
    for keys, block in longitudinal.groupby(group, sort=False):
        checkpoint, rl_seed, update, skill, phase = keys
        row: dict[str, Any] = {
            "checkpoint_id": checkpoint,
            "rl_seed": int(rl_seed),
            "global_update": int(update),
            "skill_id": skill,
            "trigger_phase": phase,
            "matched_anchors": len(block),
            "distinct_games": int(block["game_id"].nunique()),
        }
        for utility in ("semantic_utility", "total_utility", "prompt_nuisance"):
            column = f"delta_{utility}"
            estimate, lower, upper = cluster_interval(block, column, resamples, rng)
            row[column] = estimate
            row[f"{column}_lcb"] = lower
            row[f"{column}_ucb"] = upper
            row[f"{column}_positive_anchors"] = int((block[column] > 0).sum())
            row[f"{column}_negative_anchors"] = int((block[column] < 0).sum())
            row[f"{column}_zero_anchors"] = int((block[column] == 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def seed_generalization(
    longitudinal: pd.DataFrame,
    checkpoint_metrics: pd.DataFrame,
    longitudinal_summary: pd.DataFrame,
    base_checkpoint: str,
    resamples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    base_metrics = checkpoint_metrics[checkpoint_metrics["checkpoint_id"] == base_checkpoint].set_index("skill_id")
    rows = []
    for (update, skill), block in longitudinal.groupby(["global_update", "skill_id"]):
        if block["rl_seed"].nunique() < 2:
            continue
        estimate, lower, upper = hierarchical_seed_game_interval(block, "delta_semantic_utility", resamples, rng)
        per_seed = longitudinal_summary[(longitudinal_summary["global_update"] == update) & (longitudinal_summary["skill_id"] == skill)].sort_values("rl_seed")
        signs = [int(np.sign(value)) for value in per_seed["delta_semantic_utility"]]
        base_row = base_metrics.loc[skill]
        baseline_reliably_positive = bool(base_row["semantic_utility_lcb"] > 0)
        reliable_harmful_seeds = 0
        for _, post in per_seed.iterrows():
            checkpoint_row = checkpoint_metrics[
                (checkpoint_metrics["checkpoint_id"] == post["checkpoint_id"])
                & (checkpoint_metrics["skill_id"] == skill)
            ].iloc[0]
            reliable_harmful_seeds += int(baseline_reliably_positive and checkpoint_row["semantic_utility_ucb"] < 0)
        rows.append({
            "global_update": int(update),
            "skill_id": skill,
            "rl_seed_count": int(block["rl_seed"].nunique()),
            "pooled_delta_semantic_utility": estimate,
            "pooled_delta_semantic_utility_lcb": lower,
            "pooled_delta_semantic_utility_ucb": upper,
            "per_seed_delta_semantic_utility": {str(int(row.rl_seed)): float(row.delta_semantic_utility) for row in per_seed.itertuples()},
            "per_seed_delta_signs": signs,
            "same_delta_sign_across_seeds": len(set(signs)) == 1,
            "nonzero_same_direction_across_seeds": len(set(signs)) == 1 and signs[0] != 0,
            "between_seed_delta_range": float(per_seed["delta_semantic_utility"].max() - per_seed["delta_semantic_utility"].min()),
            "baseline_reliably_positive": baseline_reliably_positive,
            "reliable_harmful_seed_count": reliable_harmful_seeds,
        })
    return pd.DataFrame(rows)


def seed_phase_generalization(
    longitudinal: pd.DataFrame,
    resamples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Summarize longitudinal semantic change by invocation phase across RL seeds."""
    rows = []
    group = ["global_update", "skill_id", "trigger_phase"]
    for (update, skill, phase), block in longitudinal.groupby(group):
        if block["rl_seed"].nunique() < 2:
            continue
        estimate, lower, upper = hierarchical_seed_game_interval(
            block, "delta_semantic_utility", resamples, rng
        )
        seed_estimates = {
            str(int(seed)): float(values.groupby("game_id")["delta_semantic_utility"].mean().mean())
            for seed, values in block.groupby("rl_seed")
        }
        signs = [int(np.sign(seed_estimates[seed])) for seed in sorted(seed_estimates)]
        rows.append({
            "global_update": int(update),
            "skill_id": skill,
            "trigger_phase": phase,
            "rl_seed_count": int(block["rl_seed"].nunique()),
            "matched_anchors_per_seed": int(block.groupby("rl_seed").size().min()),
            "distinct_games": int(block["game_id"].nunique()),
            "pooled_delta_semantic_utility": estimate,
            "pooled_delta_semantic_utility_lcb": lower,
            "pooled_delta_semantic_utility_ucb": upper,
            "per_seed_delta_semantic_utility": seed_estimates,
            "per_seed_delta_signs": signs,
            "same_delta_sign_across_seeds": len(set(signs)) == 1,
            "nonzero_same_direction_across_seeds": len(set(signs)) == 1 and signs[0] != 0,
        })
    return pd.DataFrame(rows)


def audit(records: pd.DataFrame, coverage: dict[str, Any], base_checkpoint: str) -> dict[str, Any]:
    failures = []
    prompt_mismatches = 0
    router_mismatches = 0
    base_original_source_mismatches = []
    for keys, block in records.groupby(["checkpoint_id", "skill_id", "anchor_id"]):
        trajectories = {}
        for row in block.to_dict("records"):
            trajectory = json.loads(Path(row["trajectory_path"]).read_text(encoding="utf-8"))
            trajectories[row["payload_arm"]] = trajectory
            if not trajectory.get("prefix_replay_verified") or not trajectory.get("steps"):
                failures.append([*keys, row["payload_arm"], "prefix_or_steps"])
                continue
            first = trajectory["steps"][0]
            if first.get("selected_skill_id") != row["skill_id"]:
                failures.append([*keys, row["payload_arm"], "trigger_route"])
            injected = first.get("payload_injected_skill_ids", [])
            expected = [row["skill_id"]] if row["payload_arm"] in {"original", "placebo"} else []
            if injected != expected:
                failures.append([*keys, row["payload_arm"], "payload"])
            if row["checkpoint_id"] == base_checkpoint and row["payload_arm"] == "original":
                source_path = Path(trajectory["anchor"]["source_trajectory_path"])
                source = json.loads(source_path.read_text(encoding="utf-8"))
                trigger = int(row["trigger_step"])
                source_suffix = source["action_sequence"][trigger:]
                if source_suffix != trajectory["action_sequence"] or bool(source["success"]) != bool(trajectory["success"]):
                    base_original_source_mismatches.append([row["skill_id"], row["anchor_id"]])
        if set(trajectories) == set(ARMS):
            prompts = {arm: trajectories[arm]["steps"][0]["prompt_tokens"] for arm in ARMS}
            prompt_mismatches += int(prompts["original"] != prompts["placebo"])
            routers = {json.dumps(trajectories[arm]["steps"][0].get("skill_router_scores", {}), sort_keys=True) for arm in ARMS}
            router_mismatches += int(len(routers) != 1)
    supported = [row["skill_id"] for row in coverage["coverage"] if row["supported"]]
    expected_anchors = sum(row["selected_occurrences"] for row in coverage["coverage"] if row["supported"])
    return {
        "record_count": len(records),
        "checkpoint_count": int(records["checkpoint_id"].nunique()),
        "supported_skills": supported,
        "unsupported_skills": [row["skill_id"] for row in coverage["coverage"] if not row["supported"]],
        "selected_anchors_per_checkpoint": expected_anchors,
        "expected_records": expected_anchors * len(ARMS) * int(records["checkpoint_id"].nunique()),
        "duplicate_count": int(records.duplicated(["checkpoint_id", "skill_id", "anchor_id", "payload_arm"]).sum()),
        "trajectory_path_unique_count": int(records["trajectory_path"].nunique()),
        "intervention_failure_count": len(failures),
        "intervention_failures": failures,
        "original_placebo_trigger_prompt_token_mismatches": prompt_mismatches,
        "trigger_router_score_mismatches_across_arms": router_mismatches,
        "base_original_source_reproduction_mismatch_count": len(base_original_source_mismatches),
        "base_original_source_reproduction_mismatches": base_original_source_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--base-checkpoint-id", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.bootstrap_seed)
    records = load_records(args.input)
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    paired = paired_effects(records)
    checkpoint = checkpoint_skill_metrics(paired, args.bootstrap_resamples, rng)
    phases = checkpoint_phase_metrics(paired, args.bootstrap_resamples, rng)
    longitudinal = longitudinal_effects(paired, args.base_checkpoint_id)
    longitudinal_summary = longitudinal_metrics(longitudinal, args.bootstrap_resamples, rng)
    longitudinal_phases = longitudinal_phase_metrics(longitudinal, args.bootstrap_resamples, rng)
    generalization = seed_generalization(
        longitudinal, checkpoint, longitudinal_summary, args.base_checkpoint_id, args.bootstrap_resamples, rng
    )
    phase_generalization = seed_phase_generalization(longitudinal, args.bootstrap_resamples, rng)
    integrity = audit(records, coverage, args.base_checkpoint_id)
    if integrity["record_count"] != integrity["expected_records"]:
        raise ValueError(f"record count mismatch: {integrity}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_parquet(args.output_dir / "records.parquet", index=False)
    paired.to_parquet(args.output_dir / "paired_anchor_effects.parquet", index=False)
    checkpoint.to_parquet(args.output_dir / "checkpoint_skill_metrics.parquet", index=False)
    phases.to_parquet(args.output_dir / "checkpoint_skill_phase_metrics.parquet", index=False)
    longitudinal.to_parquet(args.output_dir / "longitudinal_anchor_effects.parquet", index=False)
    longitudinal_summary.to_parquet(args.output_dir / "longitudinal_skill_metrics.parquet", index=False)
    longitudinal_phases.to_parquet(args.output_dir / "longitudinal_skill_phase_metrics.parquet", index=False)
    generalization.to_parquet(args.output_dir / "seed_generalization_metrics.parquet", index=False)
    phase_generalization.to_parquet(
        args.output_dir / "seed_phase_generalization_metrics.parquet", index=False
    )
    atomic_write_json(args.output_dir / "integrity_report.json", integrity)
    atomic_write_json(args.output_dir / "summary.json", {
        "schema_version": "phase1.all_first_invocation_metrics.v1",
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "integrity": integrity,
        "checkpoint_skill_metrics": checkpoint.to_dict("records"),
        "longitudinal_skill_metrics": longitudinal_summary.to_dict("records"),
        "seed_generalization_metrics": generalization.to_dict("records"),
        "seed_phase_generalization_metrics": phase_generalization.to_dict("records"),
    })


if __name__ == "__main__":
    main()
