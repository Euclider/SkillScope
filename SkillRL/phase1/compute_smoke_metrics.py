#!/usr/bin/env python3
"""Summarize the small Phase-I model/action capability smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from phase1.archive import atomic_write_json
from phase1.metrics import matched_skill_differences


def grouped_metrics(records: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows = []
    for key, block in records.groupby(fields, dropna=False):
        steps = int(block["trajectory_length"].sum())
        invalid = int(block["invalid_action_count"].sum())
        target_retrieved = block.apply(
            lambda row: row["skill_id"] in row["retrieved_skill_ids"], axis=1
        )
        target_injected = block.apply(
            lambda row: row["skill_id"] in row["injected_skill_ids"], axis=1
        )
        row = dict(zip(fields, key if isinstance(key, tuple) else (key,)))
        row.update({
            "episodes": len(block),
            "success_rate": float(block["success"].mean()),
            "invalid_actions": invalid,
            "environment_steps": steps,
            "invalid_action_rate": invalid / max(1, steps),
            "action_format_compliance_rate": 1.0 - invalid / max(1, steps),
            "mean_trajectory_length": float(block["trajectory_length"].mean()),
            "median_trajectory_length": float(block["trajectory_length"].median()),
            "max_step_failure_rate": float(
                ((block["trajectory_length"] >= block["max_steps"]) & ~block["success"]).mean()
            ),
            "target_skill_retrieval_coverage": float(target_retrieved.mean()),
            "target_skill_injection_coverage": float(target_injected.mean()),
            "mean_unique_skill_count": float(block["unique_skill_count"].mean()),
            "mean_prompt_tokens": float(block["prompt_tokens"].mean()),
            "mean_completion_tokens": float(block["completion_tokens"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def archive_integrity(records: pd.DataFrame, expected_records: int) -> dict:
    unique_fields = [
        "checkpoint_id", "update_type", "rl_seed", "eval_seed", "game_id",
        "skill_id", "skill_condition",
    ]
    required_step_fields = {
        "prompt_text", "observation", "admissible_actions", "raw_model_output",
        "projected_action", "reward", "next_observation", "retrieved_skill_ids",
        "injected_skill_ids", "disabled_skill_ids",
    }
    missing_paths = []
    unreadable_paths = []
    missing_step_fields: dict[str, list[str]] = {}
    for path_text in records["trajectory_path"]:
        path = Path(path_text)
        if not path.exists():
            missing_paths.append(str(path))
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for index, step in enumerate(payload.get("steps", [])):
                missing = sorted(required_step_fields - set(step))
                if missing:
                    missing_step_fields[f"{path}:{index}"] = missing
        except Exception as error:
            unreadable_paths.append(f"{path}: {error!r}")
    duplicate_count = int(records.duplicated(unique_fields).sum())
    result = {
        "expected_rollout_records": expected_records,
        "actual_rollout_records": len(records),
        "duplicate_unique_keys": duplicate_count,
        "missing_trajectory_paths": missing_paths,
        "unreadable_trajectory_paths": unreadable_paths,
        "steps_with_missing_fields": missing_step_fields,
    }
    result["passed"] = (
        len(records) == expected_records
        and duplicate_count == 0
        and not missing_paths
        and not unreadable_paths
        and not missing_step_fields
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=36)
    args = parser.parse_args()

    records = pd.read_json(args.input, lines=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_context_condition = grouped_metrics(records, ["context_id", "skill_condition"])
    by_condition = grouped_metrics(records, ["skill_condition"])
    paired = matched_skill_differences(records)
    margins = (
        paired.groupby(["context_id", "skill_id"], dropna=False)["margin"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "preliminary_full_minus_skill_margin", "count": "paired_samples"})
    )
    no_skill = records.pivot(
        index=[
            "checkpoint_id", "update_type", "rl_seed", "eval_seed",
            "environment_seed", "game_id", "context_id", "skill_id",
        ],
        columns="skill_condition",
        values="success",
    )
    if {"full_bank", "no_skill"}.issubset(no_skill.columns):
        total_bank = (no_skill["full_bank"].astype(float) - no_skill["no_skill"].astype(float))
        total_bank = total_bank.groupby(["context_id", "skill_id"]).agg(["mean", "count"]).reset_index()
        total_bank = total_bank.rename(columns={"mean": "preliminary_full_minus_no_skill_margin", "count": "paired_samples"})
    else:
        total_bank = pd.DataFrame()

    integrity = archive_integrity(records, args.expected_records)
    by_context_condition.to_parquet(args.output_dir / "by_context_condition.parquet", index=False)
    by_context_condition.to_csv(args.output_dir / "by_context_condition.csv", index=False)
    by_condition.to_parquet(args.output_dir / "by_condition.parquet", index=False)
    by_condition.to_csv(args.output_dir / "by_condition.csv", index=False)
    margins.to_parquet(args.output_dir / "preliminary_skill_margins.parquet", index=False)
    margins.to_csv(args.output_dir / "preliminary_skill_margins.csv", index=False)
    total_bank.to_parquet(args.output_dir / "preliminary_total_bank_margins.parquet", index=False)
    total_bank.to_csv(args.output_dir / "preliminary_total_bank_margins.csv", index=False)
    atomic_write_json(args.output_dir / "archive_integrity.json", integrity)

    overall_steps = int(records["trajectory_length"].sum())
    overall_invalid = int(records["invalid_action_count"].sum())
    summary = {
        "schema_version": "phase1.smoke_summary.v1",
        "rollout_records": len(records),
        "independent_games": int(records["game_id"].nunique()),
        "contexts": int(records["context_id"].nunique()),
        "conditions": sorted(records["skill_condition"].unique()),
        "success_rate": float(records["success"].mean()),
        "environment_steps": overall_steps,
        "invalid_action_rate": overall_invalid / max(1, overall_steps),
        "action_format_compliance_rate": 1.0 - overall_invalid / max(1, overall_steps),
        "mean_trajectory_length": float(records["trajectory_length"].mean()),
        "archive_integrity_passed": integrity["passed"],
        "capability_gate_invalid_action_rate_lte_0_30": overall_invalid / max(1, overall_steps) <= 0.30,
        "note": "Two games and one eval seed per context: margins are diagnostic only.",
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
