#!/usr/bin/env python3
"""Create report-oriented diagnostics from audited three-seed utility metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase1.archive import atomic_write_json


def scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [scalar(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar(item) for item in value]
    return value


def checkpoint_aggregates(paired: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for checkpoint, block in paired.groupby("checkpoint_id", sort=False):
        flipped = block["original_placebo_first_action_flip"].astype(bool)
        nonzero = block["semantic_utility"] != 0
        rows.append({
            "checkpoint_id": checkpoint,
            "rl_seed": int(block["rl_seed"].iloc[0]),
            "global_update": int(block["global_update"].iloc[0]),
            "anchor_count": len(block),
            "first_action_flip_rate": float(flipped.mean()),
            "suffix_divergence_rate": float(block["original_placebo_suffix_divergence"].mean()),
            "nonzero_reward_contrast_rate": float(nonzero.mean()),
            "positive_anchor_count": int((block["semantic_utility"] > 0).sum()),
            "negative_anchor_count": int((block["semantic_utility"] < 0).sum()),
            "zero_anchor_count": int((block["semantic_utility"] == 0).sum()),
            "nonzero_reward_given_first_flip": (
                float(nonzero[flipped].mean()) if flipped.any() else None
            ),
        })
    return rows


def anchor_concordance(longitudinal: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (update, skill), block in longitudinal.groupby(["global_update", "skill_id"]):
        pivot = block.pivot(index="anchor_id", columns="rl_seed", values="delta_semantic_utility")
        if pivot.shape[1] < 2 or pivot.isna().any().any():
            continue
        signs = np.sign(pivot.to_numpy(float))
        nonzero_union = np.any(signs != 0, axis=1)
        conflict = (np.min(signs, axis=1) < 0) & (np.max(signs, axis=1) > 0)
        all_positive = np.all(signs > 0, axis=1)
        all_negative = np.all(signs < 0, axis=1)
        all_zero = np.all(signs == 0, axis=1)
        same_nonzero = all_positive | all_negative
        rows.append({
            "global_update": int(update),
            "skill_id": skill,
            "rl_seed_count": int(pivot.shape[1]),
            "anchor_count": int(pivot.shape[0]),
            "nonzero_union_count": int(nonzero_union.sum()),
            "all_seed_same_nonzero_count": int(same_nonzero.sum()),
            "all_seed_positive_count": int(all_positive.sum()),
            "all_seed_negative_count": int(all_negative.sum()),
            "positive_negative_conflict_count": int(conflict.sum()),
            "all_seed_zero_count": int(all_zero.sum()),
        })
    return rows


def sign_turnover(longitudinal: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    labels = {-1: "negative", 0: "zero", 1: "positive"}
    for keys, block in longitudinal.groupby(
        ["checkpoint_id", "rl_seed", "global_update", "skill_id"], sort=False
    ):
        checkpoint, seed, update, skill = keys
        transitions: dict[str, int] = {}
        for base, post in zip(block["base_semantic_utility"], block["semantic_utility"]):
            key = f"{labels[int(np.sign(base))]}_to_{labels[int(np.sign(post))]}"
            transitions[key] = transitions.get(key, 0) + 1
        rows.append({
            "checkpoint_id": checkpoint,
            "rl_seed": int(seed),
            "global_update": int(update),
            "skill_id": skill,
            "transitions": transitions,
        })
    return rows


def reliable_harmful_pairs(checkpoint: pd.DataFrame, base_id: str) -> list[dict[str, Any]]:
    base = checkpoint[checkpoint["checkpoint_id"] == base_id].set_index("skill_id")
    rows = []
    for post in checkpoint[checkpoint["checkpoint_id"] != base_id].itertuples():
        baseline = base.loc[post.skill_id]
        if baseline.semantic_utility_lcb > 0 and post.semantic_utility_ucb < 0:
            rows.append({
                "checkpoint_id": post.checkpoint_id,
                "skill_id": post.skill_id,
                "base_semantic_utility": float(baseline.semantic_utility),
                "base_lcb": float(baseline.semantic_utility_lcb),
                "base_ucb": float(baseline.semantic_utility_ucb),
                "post_semantic_utility": float(post.semantic_utility),
                "post_lcb": float(post.semantic_utility_lcb),
                "post_ucb": float(post.semantic_utility_ucb),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint-id", default="b0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paired = pd.read_parquet(args.metrics_dir / "paired_anchor_effects.parquet")
    checkpoint = pd.read_parquet(args.metrics_dir / "checkpoint_skill_metrics.parquet")
    longitudinal = pd.read_parquet(args.metrics_dir / "longitudinal_anchor_effects.parquet")
    longitudinal_summary = pd.read_parquet(args.metrics_dir / "longitudinal_skill_metrics.parquet")
    generalization = pd.read_parquet(args.metrics_dir / "seed_generalization_metrics.parquet")
    phase_generalization = pd.read_parquet(args.metrics_dir / "seed_phase_generalization_metrics.parquet")

    payload = {
        "schema_version": "phase1.three_seed_report_diagnostics.v1",
        "checkpoint_aggregates": checkpoint_aggregates(paired),
        "checkpoint_skill_metrics": checkpoint.to_dict("records"),
        "longitudinal_skill_metrics": longitudinal_summary.to_dict("records"),
        "seed_generalization_metrics": generalization.to_dict("records"),
        "seed_phase_generalization_metrics": phase_generalization.to_dict("records"),
        "anchor_cross_seed_concordance": anchor_concordance(longitudinal),
        "anchor_sign_turnover": sign_turnover(longitudinal),
        "reliable_harmful_pairs": reliable_harmful_pairs(
            checkpoint, args.base_checkpoint_id
        ),
        "overall": {
            "checkpoint_anchor_pairs": len(paired),
            "first_action_flip_rate": float(
                paired["original_placebo_first_action_flip"].mean()
            ),
            "suffix_divergence_rate": float(
                paired["original_placebo_suffix_divergence"].mean()
            ),
            "nonzero_reward_contrast_rate": float(
                (paired["semantic_utility"] != 0).mean()
            ),
        },
    }
    atomic_write_json(args.output, scalar(payload))


if __name__ == "__main__":
    main()
