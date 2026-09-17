#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from phase1.archive import atomic_write_json


def js_divergence(log_p: dict[str, float], log_q: dict[str, float]) -> float:
    actions = sorted(set(log_p) & set(log_q))
    if not actions:
        return math.nan
    p = np.array([math.exp(log_p[action]) for action in actions], dtype=float)
    q = np.array([math.exp(log_q[action]) for action in actions], dtype=float)
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / midpoint)) + 0.5 * np.sum(q * np.log(q / midpoint)))


def validate_fixed_state_matches(records: pd.DataFrame) -> None:
    """Reject pre/post comparisons whose state, prompt, or action set changed."""
    for key, block in records.groupby(["probe_id", "skill_id", "skill_condition"]):
        if block["prompt_hash"].nunique(dropna=False) != 1:
            raise ValueError(f"Prompt changed across checkpoints for fixed probe {key}")
        action_sets = block["candidate_action_log_probs"].map(
            lambda values: tuple(sorted(values))
        )
        if action_sets.nunique(dropna=False) != 1:
            raise ValueError(f"Admissible actions changed across checkpoints for fixed probe {key}")


def condition_interaction_summary(metrics: pd.DataFrame) -> dict:
    rates = metrics.groupby("skill_condition")["action_flip"].mean()
    index = ["probe_id", "skill_id", "pre_checkpoint_id", "post_checkpoint_id"]
    flips = metrics.pivot(index=index, columns="skill_condition", values="action_flip")
    summary = {
        "full_minus_minus_skill_flip_rate": float(
            rates.get("full_bank", math.nan) - rates.get("minus_skill", math.nan)
        ),
        "full_minus_no_skill_flip_rate": float(
            rates.get("full_bank", math.nan) - rates.get("no_skill", math.nan)
        ),
    }
    if {"full_bank", "minus_skill"}.issubset(flips.columns):
        summary["full_flip_minus_skill_stable_count"] = int(
            (flips["full_bank"].astype(bool) & ~flips["minus_skill"].astype(bool)).sum()
        )
    if {"full_bank", "no_skill"}.issubset(flips.columns):
        summary["full_flip_no_skill_stable_count"] = int(
            (flips["full_bank"].astype(bool) & ~flips["no_skill"].astype(bool)).sum()
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint-order", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = pd.read_json(args.input, lines=True)
    validate_fixed_state_matches(records)
    order = {checkpoint: index for index, checkpoint in enumerate(args.checkpoint_order)}
    records["checkpoint_order"] = records["checkpoint_id"].map(order)
    if records["checkpoint_order"].isna().any():
        raise ValueError("--checkpoint-order does not cover all checkpoint_id values")
    def transition_row(key, pre, post) -> dict:
        return {
            "probe_id": key[0], "skill_id": key[1], "skill_condition": key[2],
            "pre_checkpoint_id": pre["checkpoint_id"],
            "post_checkpoint_id": post["checkpoint_id"],
            "action_flip": pre["greedy_projected_action"] != post["greedy_projected_action"],
            "action_js_divergence": js_divergence(
                pre["candidate_action_log_probs"], post["candidate_action_log_probs"]
            ),
            "pre_action_valid": pre["is_action_valid"],
            "post_action_valid": post["is_action_valid"],
            "pre_action_admissible": pre["is_action_admissible"],
            "post_action_admissible": post["is_action_admissible"],
        }

    rows = []
    endpoint_rows = []
    for key, block in records.groupby(["probe_id", "skill_id", "skill_condition"]):
        ordered = block.sort_values("checkpoint_order")
        for (_, pre), (_, post) in zip(ordered.iloc[:-1].iterrows(), ordered.iloc[1:].iterrows()):
            rows.append(transition_row(key, pre, post))
        endpoint_rows.append(transition_row(key, ordered.iloc[0], ordered.iloc[-1]))
    metrics = pd.DataFrame(rows)
    endpoint_metrics = pd.DataFrame(endpoint_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(args.output_dir / "action_probe_transitions.parquet", index=False)
    endpoint_metrics.to_parquet(
        args.output_dir / "action_probe_endpoint_transitions.parquet", index=False
    )

    comparison_keys = ["probe_id", "skill_id", "pre_checkpoint_id", "post_checkpoint_id"]
    wide = metrics.pivot(index=comparison_keys, columns="skill_condition", values=["action_flip", "action_js_divergence"])
    if ("action_js_divergence", "full_bank") in wide and ("action_js_divergence", "minus_skill") in wide:
        wide[("interaction", "js_full_minus_minus_skill")] = (
            wide[("action_js_divergence", "full_bank")]
            - wide[("action_js_divergence", "minus_skill")]
        )
    wide.reset_index().to_parquet(args.output_dir / "action_probe_interactions.parquet", index=False)
    atomic_write_json(args.output_dir / "action_probe_summary.json", {
        "probe_transition_rows": len(metrics),
        "flip_rate_by_condition": metrics.groupby("skill_condition")["action_flip"].mean().to_dict(),
        "mean_js_by_condition": metrics.groupby("skill_condition")["action_js_divergence"].mean().to_dict(),
        "endpoint_transition_rows": len(endpoint_metrics),
        "endpoint_flip_rate_by_condition": endpoint_metrics.groupby("skill_condition")["action_flip"].mean().to_dict(),
        "endpoint_mean_js_by_condition": endpoint_metrics.groupby("skill_condition")["action_js_divergence"].mean().to_dict(),
        "adjacent_condition_interactions": condition_interaction_summary(metrics),
        "endpoint_condition_interactions": condition_interaction_summary(endpoint_metrics),
    })


if __name__ == "__main__":
    main()
