#!/usr/bin/env python3
"""Create an exact paired comparison between two smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from phase1.archive import atomic_write_json

PAIR_FIELDS = [
    "eval_seed",
    "environment_seed",
    "game_id",
    "context_id",
    "skill_id",
    "skill_condition",
]


def prepare(path: Path, prefix: str) -> pd.DataFrame:
    records = pd.read_json(path, lines=True)
    if records.duplicated(PAIR_FIELDS).any():
        raise ValueError(f"Duplicate paired keys in {path}")
    records[f"{prefix}_invalid_action_rate"] = (
        records["invalid_action_count"] / records["trajectory_length"]
    )
    records[f"{prefix}_completion_tokens_per_step"] = (
        records["completion_tokens"] / records["trajectory_length"]
    )
    keep = PAIR_FIELDS + [
        "success",
        "invalid_action_count",
        "trajectory_length",
        "prompt_tokens",
        "completion_tokens",
        f"{prefix}_invalid_action_rate",
        f"{prefix}_completion_tokens_per_step",
    ]
    return records[keep].rename(columns={
        field: f"{prefix}_{field}"
        for field in (
            "success", "invalid_action_count", "trajectory_length",
            "prompt_tokens", "completion_tokens",
        )
    })


def aggregate(paired: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows = []
    for key, block in paired.groupby(fields, dropna=False):
        row = dict(zip(fields, key if isinstance(key, tuple) else (key,)))
        row.update({
            "paired_episodes": len(block),
            "baseline_success_rate": float(block["baseline_success"].mean()),
            "candidate_success_rate": float(block["candidate_success"].mean()),
            "success_rate_delta": float(block["success_delta"].mean()),
            "baseline_invalid_action_rate": float(
                block["baseline_invalid_action_count"].sum()
                / block["baseline_trajectory_length"].sum()
            ),
            "candidate_invalid_action_rate": float(
                block["candidate_invalid_action_count"].sum()
                / block["candidate_trajectory_length"].sum()
            ),
            "invalid_action_rate_delta": float(
                block["candidate_invalid_action_count"].sum()
                / block["candidate_trajectory_length"].sum()
                - block["baseline_invalid_action_count"].sum()
                / block["baseline_trajectory_length"].sum()
            ),
            "mean_completion_tokens_per_step_delta": float(
                block["completion_tokens_per_step_delta"].mean()
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = prepare(args.baseline, "baseline")
    candidate = prepare(args.candidate, "candidate")
    paired = baseline.merge(candidate, on=PAIR_FIELDS, how="outer", indicator=True)
    unmatched = paired[paired["_merge"] != "both"]
    if not unmatched.empty:
        raise ValueError(f"Smoke runs do not have identical paired keys: {len(unmatched)} unmatched")
    paired = paired.drop(columns="_merge")
    paired["success_delta"] = (
        paired["candidate_success"].astype(float) - paired["baseline_success"].astype(float)
    )
    paired["invalid_action_rate_delta"] = (
        paired["candidate_invalid_action_rate"] - paired["baseline_invalid_action_rate"]
    )
    paired["completion_tokens_per_step_delta"] = (
        paired["candidate_completion_tokens_per_step"]
        - paired["baseline_completion_tokens_per_step"]
    )
    by_condition = aggregate(paired, ["skill_condition"])
    by_context_condition = aggregate(paired, ["context_id", "skill_condition"])
    by_context = aggregate(paired, ["context_id"])
    overall = aggregate(paired.assign(_all="all"), ["_all"]).iloc[0].to_dict()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("paired_episodes", paired),
        ("by_condition", by_condition),
        ("by_context_condition", by_context_condition),
        ("by_context", by_context),
    ):
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
        frame.to_parquet(args.output_dir / f"{name}.parquet", index=False)
    atomic_write_json(args.output_dir / "summary.json", {
        "schema_version": "phase1.smoke_comparison.v1",
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "paired_episodes": len(paired),
        "identical_pair_keys": True,
        **overall,
        "by_condition": by_condition.to_dict(orient="records"),
        "by_context": by_context.to_dict(orient="records"),
    })
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
