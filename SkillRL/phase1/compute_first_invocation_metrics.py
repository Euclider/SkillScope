#!/usr/bin/env python3
"""Compute paired payload utility and longitudinal checkpoint changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase1.archive import atomic_write_json
from phase1.first_invocation import PayloadArm


def load_records(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return pd.DataFrame(rows)


def cluster_interval(
    values: pd.DataFrame,
    column: str,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float, int]:
    clusters = values.groupby("game_id", dropna=False)[column].mean()
    data = clusters.to_numpy(dtype=float)
    estimate = float(data.mean())
    if len(data) == 1:
        return estimate, estimate, estimate, 1
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, (resamples, len(data)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return estimate, float(lower), float(upper), len(data)


def paired_effects(records: pd.DataFrame) -> pd.DataFrame:
    key = [
        "checkpoint_id", "anchor_id", "state_id", "source_eval_seed",
        "environment_seed", "game_id", "context_id", "skill_id",
        "trigger_step",
    ]
    required_arms = {arm.value for arm in PayloadArm}
    duplicates = records.duplicated([*key, "payload_arm"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate checkpoint/anchor/payload-arm records")
    for _, block in records.groupby(key, dropna=False):
        if set(block["payload_arm"]) != required_arms:
            raise ValueError("Incomplete original/placebo/null matched branch")
    value_columns = [
        "suffix_return", "success", "invalid_action_count", "first_action",
        "target_selected_count", "target_payload_injection_count",
        "suffix_trajectory_length",
    ]
    wide = records.pivot(index=key, columns="payload_arm", values=value_columns)
    wide.columns = [f"{metric}_{arm}" for metric, arm in wide.columns]
    wide = wide.reset_index()
    wide["semantic_utility"] = (
        wide["suffix_return_original"] - wide["suffix_return_placebo"]
    )
    wide["total_utility"] = (
        wide["suffix_return_original"] - wide["suffix_return_null"]
    )
    wide["prompt_nuisance"] = (
        wide["suffix_return_placebo"] - wide["suffix_return_null"]
    )
    wide["semantic_success_utility"] = (
        wide["success_original"].astype(float)
        - wide["success_placebo"].astype(float)
    )
    wide["total_success_utility"] = (
        wide["success_original"].astype(float)
        - wide["success_null"].astype(float)
    )
    wide["original_placebo_first_action_flip"] = (
        wide["first_action_original"] != wide["first_action_placebo"]
    )
    wide["original_null_first_action_flip"] = (
        wide["first_action_original"] != wide["first_action_null"]
    )
    return wide


def summarize_checkpoints(
    paired: pd.DataFrame,
    *,
    checkpoint_order: list[str],
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    order = {checkpoint: index for index, checkpoint in enumerate(checkpoint_order)}
    for checkpoint, block in paired.groupby("checkpoint_id", sort=False):
        row: dict[str, Any] = {
            "checkpoint_id": checkpoint,
            "checkpoint_order": order[checkpoint],
            "matched_anchors": len(block),
            "state_count": int(block["state_id"].nunique()),
            "game_clusters": int(block["game_id"].nunique()),
        }
        for column in (
            "semantic_utility", "total_utility", "prompt_nuisance",
            "semantic_success_utility", "total_success_utility",
        ):
            estimate, lower, upper, clusters = cluster_interval(
                block, column, resamples=resamples, seed=seed
            )
            row[column] = estimate
            row[f"{column}_lcb"] = lower
            row[f"{column}_ucb"] = upper
            row[f"{column}_clusters"] = clusters
        for arm in (arm.value for arm in PayloadArm):
            row[f"success_rate_{arm}"] = float(
                block[f"success_{arm}"].astype(float).mean()
            )
            row[f"success_count_{arm}"] = int(
                block[f"success_{arm}"].astype(bool).sum()
            )
            row[f"mean_return_{arm}"] = float(
                block[f"suffix_return_{arm}"].astype(float).mean()
            )
            row[f"mean_invalid_actions_{arm}"] = float(
                block[f"invalid_action_count_{arm}"].astype(float).mean()
            )
            row[f"mean_suffix_length_{arm}"] = float(
                block[f"suffix_trajectory_length_{arm}"].astype(float).mean()
            )
            row[f"mean_target_selections_{arm}"] = float(
                block[f"target_selected_count_{arm}"].astype(float).mean()
            )
            row[f"mean_target_payload_injections_{arm}"] = float(
                block[f"target_payload_injection_count_{arm}"].astype(float).mean()
            )
        for utility in ("semantic_utility", "total_utility", "prompt_nuisance"):
            row[f"{utility}_positive_anchors"] = int((block[utility] > 0).sum())
            row[f"{utility}_negative_anchors"] = int((block[utility] < 0).sum())
            row[f"{utility}_zero_anchors"] = int((block[utility] == 0).sum())
        row["original_placebo_first_action_flip_rate"] = float(
            block["original_placebo_first_action_flip"].mean()
        )
        row["original_null_first_action_flip_rate"] = float(
            block["original_null_first_action_flip"].mean()
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("checkpoint_order")


def longitudinal_effects(
    paired: pd.DataFrame,
    *,
    base_checkpoint: str,
    checkpoint_order: list[str],
    resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = [
        "anchor_id", "state_id", "source_eval_seed", "environment_seed",
        "game_id", "context_id", "skill_id", "trigger_step",
    ]
    value_columns = ["semantic_utility", "total_utility", "prompt_nuisance"]
    wide = paired.pivot(index=identity, columns="checkpoint_id", values=value_columns)
    rows = []
    summary = []
    for checkpoint in checkpoint_order:
        if checkpoint == base_checkpoint:
            continue
        block = pd.DataFrame(index=wide.index).reset_index()
        for value in value_columns:
            block[f"delta_{value}"] = (
                wide[(value, checkpoint)] - wide[(value, base_checkpoint)]
            ).to_numpy()
        block["pre_checkpoint_id"] = base_checkpoint
        block["post_checkpoint_id"] = checkpoint
        rows.append(block)
        result: dict[str, Any] = {
            "pre_checkpoint_id": base_checkpoint,
            "post_checkpoint_id": checkpoint,
            "matched_anchors": len(block),
            "game_clusters": int(block["game_id"].nunique()),
        }
        for value in value_columns:
            column = f"delta_{value}"
            estimate, lower, upper, _ = cluster_interval(
                block, column, resamples=resamples, seed=seed
            )
            result[column] = estimate
            result[f"{column}_lcb"] = lower
            result[f"{column}_ucb"] = upper
            result[f"{column}_positive_anchors"] = int((block[column] > 0).sum())
            result[f"{column}_negative_anchors"] = int((block[column] < 0).sum())
            result[f"{column}_zero_anchors"] = int((block[column] == 0).sum())
        summary.append(result)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summary)


def audit_trajectories(records: pd.DataFrame) -> dict[str, Any]:
    failures = []
    payload_failures = []
    first_prompt_tokens: dict[tuple[str, str], dict[str, int]] = {}
    first_router_hashes: dict[tuple[str, str], dict[str, str]] = {}
    for row in records.to_dict("records"):
        trajectory = json.loads(
            Path(row["trajectory_path"]).read_text(encoding="utf-8")
        )
        steps = trajectory["steps"]
        if not trajectory.get("prefix_replay_verified") or not steps:
            failures.append([row["checkpoint_id"], row["anchor_id"], "replay_or_steps"])
            continue
        first = steps[0]
        if first.get("selected_skill_id") != row["skill_id"]:
            failures.append([row["checkpoint_id"], row["anchor_id"], "trigger_route"])
        injected = first.get("payload_injected_skill_ids", [])
        if row["payload_arm"] in {"original", "placebo"}:
            valid_payload = (
                injected == [row["skill_id"]]
                and bool(first.get("payload_text"))
            )
        else:
            valid_payload = injected == [] and first.get("payload_text") == ""
        if not valid_payload:
            payload_failures.append([
                row["checkpoint_id"], row["anchor_id"], row["payload_arm"]
            ])
        key = (row["checkpoint_id"], row["anchor_id"])
        first_prompt_tokens.setdefault(key, {})[row["payload_arm"]] = int(
            first["prompt_tokens"]
        )
        first_router_hashes.setdefault(key, {})[row["payload_arm"]] = json.dumps(
            first.get("skill_router_scores", {}), sort_keys=True
        )
    prompt_mismatches = sum(
        values.get("original") != values.get("placebo")
        for values in first_prompt_tokens.values()
    )
    router_mismatches = sum(
        len(set(values.values())) != 1 for values in first_router_hashes.values()
    )
    return {
        "trajectory_count": len(records),
        "unique_trajectory_path_count": int(records["trajectory_path"].nunique()),
        "duplicate_checkpoint_anchor_arm_count": int(records.duplicated(
            ["checkpoint_id", "anchor_id", "payload_arm"]
        ).sum()),
        "records_per_checkpoint": {
            str(key): int(value)
            for key, value in records.groupby("checkpoint_id").size().items()
        },
        "records_per_payload_arm": {
            str(key): int(value)
            for key, value in records.groupby("payload_arm").size().items()
        },
        "trajectory_failures": failures,
        "trajectory_failure_count": len(failures),
        "payload_intervention_failures": payload_failures,
        "payload_intervention_failure_count": len(payload_failures),
        "original_placebo_trigger_prompt_token_mismatches": prompt_mismatches,
        "trigger_router_score_mismatches_across_arms": router_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint-order", nargs="+", required=True)
    parser.add_argument("--base-checkpoint-id", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.input)
    if set(records["checkpoint_id"]) != set(args.checkpoint_order):
        raise ValueError("--checkpoint-order does not match evaluator records")
    paired = paired_effects(records)
    checkpoint_metrics = summarize_checkpoints(
        paired,
        checkpoint_order=args.checkpoint_order,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    longitudinal, longitudinal_summary = longitudinal_effects(
        paired,
        base_checkpoint=args.base_checkpoint_id,
        checkpoint_order=args.checkpoint_order,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    audit = audit_trajectories(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_parquet(args.output_dir / "records.parquet", index=False)
    paired.to_parquet(args.output_dir / "paired_anchor_effects.parquet", index=False)
    checkpoint_metrics.to_parquet(
        args.output_dir / "checkpoint_utility_metrics.parquet", index=False
    )
    longitudinal.to_parquet(
        args.output_dir / "longitudinal_anchor_effects.parquet", index=False
    )
    longitudinal_summary.to_parquet(
        args.output_dir / "longitudinal_utility_metrics.parquet", index=False
    )
    atomic_write_json(args.output_dir / "integrity_report.json", audit)
    atomic_write_json(args.output_dir / "summary.json", {
        "schema_version": "phase1.first_invocation_metrics.v1",
        "record_count": len(records),
        "matched_anchor_checkpoint_rows": len(paired),
        "checkpoint_count": int(records["checkpoint_id"].nunique()),
        "anchor_occurrence_count": int(records["anchor_id"].nunique()),
        "distinct_state_count": int(records["state_id"].nunique()),
        "game_cluster_count": int(records["game_id"].nunique()),
        "trigger_steps": sorted(records["trigger_step"].unique().tolist()),
        "arms": sorted(records["payload_arm"].unique().tolist()),
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "checkpoint_metrics": checkpoint_metrics.to_dict("records"),
        "longitudinal_metrics": longitudinal_summary.to_dict("records"),
        "integrity": audit,
    })


if __name__ == "__main__":
    main()
