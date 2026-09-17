#!/usr/bin/env python3
"""Minimal cross-seed forecast from first-action policy-Skill interaction shifts."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from phase1.archive import atomic_write_json

IDENTITY = [
    "skill_id",
    "anchor_id",
    "state_id",
    "source_eval_seed",
    "environment_seed",
    "game_id",
    "context_id",
    "trigger_step",
    "trigger_phase",
]
SEEDS = (101, 202, 303)
WINDOWS = ((0, 10), (10, 20), (20, 30))
FEATURE_SETS = {
    "prevalence": [],
    "old_margin": ["old_margin"],
    "old_plus_generic": ["old_margin", "generic_top1_shift"],
    "old_plus_interaction": [
        "old_margin",
        "generic_top1_shift",
        "top1_interaction_shift",
    ],
}


def build_transitions(paired: pd.DataFrame) -> pd.DataFrame:
    base = paired[paired["checkpoint_id"] == "b0"].set_index(IDENTITY)
    rows: list[pd.DataFrame] = []
    for seed in SEEDS:
        for pre_update, post_update in WINDOWS:
            pre = (
                base
                if pre_update == 0
                else paired[
                    (paired["rl_seed"] == seed)
                    & (paired["global_update"] == pre_update)
                ].set_index(IDENTITY)
            )
            post = paired[
                (paired["rl_seed"] == seed)
                & (paired["global_update"] == post_update)
            ].set_index(IDENTITY)
            block = pre.join(post, lsuffix="_pre", rsuffix="_post", how="inner")
            if len(block) != 200:
                raise ValueError(
                    f"Expected 200 fixed anchors for seed={seed}, "
                    f"window={pre_update}->{post_update}; got {len(block)}"
                )
            block = block.reset_index()
            block["rl_seed"] = seed
            block["pre_update"] = pre_update
            block["post_update"] = post_update
            rows.append(block)
    transitions = pd.concat(rows, ignore_index=True)
    transitions["old_margin"] = transitions[
        "semantic_success_utility_pre"
    ].astype(float)
    transitions["delta_semantic_success_utility"] = (
        transitions["semantic_success_utility_post"]
        - transitions["semantic_success_utility_pre"]
    ).astype(float)
    transitions["absolute_delta_semantic_success_utility"] = transitions[
        "delta_semantic_success_utility"
    ].abs()
    transitions["utility_changed"] = transitions[
        "delta_semantic_success_utility"
    ].ne(0)
    transitions["negative_delta"] = transitions[
        "delta_semantic_success_utility"
    ].lt(0)
    transitions["point_harmful_flip"] = (
        transitions["semantic_success_utility_pre"].gt(0)
        & transitions["semantic_success_utility_post"].lt(0)
    )
    transitions["pre_original_placebo_flip"] = transitions[
        "first_action_original_pre"
    ].ne(transitions["first_action_placebo_pre"])
    transitions["post_original_placebo_flip"] = transitions[
        "first_action_original_post"
    ].ne(transitions["first_action_placebo_post"])
    transitions["original_top1_shift"] = transitions[
        "first_action_original_pre"
    ].ne(transitions["first_action_original_post"])
    transitions["generic_top1_shift"] = transitions[
        "first_action_placebo_pre"
    ].ne(transitions["first_action_placebo_post"])
    transitions["top1_interaction_shift"] = transitions[
        "original_top1_shift"
    ].ne(transitions["generic_top1_shift"])
    transitions["action_pair_changed"] = transitions[
        "original_top1_shift"
    ] | transitions["generic_top1_shift"]
    transitions["original_placebo_flip_onset"] = (
        transitions["post_original_placebo_flip"]
        & ~transitions["pre_original_placebo_flip"]
    )
    transitions["original_placebo_flip_offset"] = (
        transitions["pre_original_placebo_flip"]
        & ~transitions["post_original_placebo_flip"]
    )
    feature_columns = [
        "generic_top1_shift",
        "original_top1_shift",
        "top1_interaction_shift",
        "action_pair_changed",
        "pre_original_placebo_flip",
        "post_original_placebo_flip",
        "original_placebo_flip_onset",
        "original_placebo_flip_offset",
    ]
    transitions[feature_columns] = transitions[feature_columns].astype(int)
    return transitions


def game_equal_weights(block: pd.DataFrame) -> np.ndarray:
    groups = ["rl_seed", "pre_update", "post_update", "game_id"]
    counts = block.groupby(groups, dropna=False)["anchor_id"].transform("count")
    weights = 1.0 / counts.to_numpy(float)
    return weights * (len(weights) / weights.sum())


def weighted_rate(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def association(block: pd.DataFrame, label: str) -> dict[str, float | int]:
    indicator = block["top1_interaction_shift"].to_numpy(bool)
    outcome = block[label].to_numpy(float)
    weights = game_equal_weights(block)
    on = weighted_rate(outcome[indicator], weights[indicator])
    off = weighted_rate(outcome[~indicator], weights[~indicator])
    return {
        "rows": len(block),
        "distinct_games": int(block["game_id"].nunique()),
        "interaction_positive_rows": int(indicator.sum()),
        "rate_when_interaction_shift": on,
        "rate_without_interaction_shift": off,
        "rate_difference": on - off,
        "risk_ratio": on / off if off > 0 else math.inf,
    }


def game_contributions(block: pd.DataFrame, label: str) -> np.ndarray:
    """Return per-game sufficient statistics under game-window equal weighting."""
    indicator = block["top1_interaction_shift"].to_numpy(bool)
    outcome = block[label].to_numpy(float)
    groups = ["rl_seed", "pre_update", "post_update", "game_id"]
    counts = block.groupby(groups, dropna=False)["anchor_id"].transform("count")
    weights = 1.0 / counts.to_numpy(float)
    contributions = pd.DataFrame({
        "game_id": block["game_id"].to_numpy(),
        "on_num": weights * indicator * outcome,
        "on_den": weights * indicator,
        "off_num": weights * (~indicator) * outcome,
        "off_den": weights * (~indicator),
    })
    return contributions.groupby("game_id", sort=False)[
        ["on_num", "on_den", "off_num", "off_den"]
    ].sum().to_numpy(float)


def effects_from_totals(totals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    on = np.divide(
        totals[..., 0],
        totals[..., 1],
        out=np.full(totals.shape[:-1], np.nan),
        where=totals[..., 1] > 0,
    )
    off = np.divide(
        totals[..., 2],
        totals[..., 3],
        out=np.full(totals.shape[:-1], np.nan),
        where=totals[..., 3] > 0,
    )
    ratios = np.divide(
        on,
        off,
        out=np.full_like(on, np.inf),
        where=off > 0,
    )
    return on - off, ratios


def bootstrap_association(
    block: pd.DataFrame,
    label: str,
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    contributions = game_contributions(block, label)
    sample_indices = rng.integers(
        0, len(contributions), size=(resamples, len(contributions))
    )
    totals = contributions[sample_indices].sum(axis=1)
    differences, ratios = effects_from_totals(totals)
    differences = differences[np.isfinite(differences)]
    finite_ratios = ratios[np.isfinite(ratios)]
    return {
        "rate_difference_lcb": float(np.quantile(differences, 0.025)),
        "rate_difference_ucb": float(np.quantile(differences, 0.975)),
        "risk_ratio_lcb": (
            float(np.quantile(finite_ratios, 0.025))
            if len(finite_ratios)
            else math.nan
        ),
        "risk_ratio_ucb": (
            float(np.quantile(finite_ratios, 0.975))
            if len(finite_ratios)
            else math.nan
        ),
    }


def hierarchical_bootstrap_association(
    data: pd.DataFrame,
    label: str,
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    seeds = np.array(SEEDS)
    contributions = {
        seed: game_contributions(data[data["rl_seed"] == seed], label)
        for seed in seeds
    }
    sampled_seeds = rng.choice(
        seeds, size=(resamples, len(seeds)), replace=True
    )
    totals = np.zeros((resamples, 4), dtype=float)
    for draw in range(len(seeds)):
        for seed in seeds:
            mask = sampled_seeds[:, draw] == seed
            count = int(mask.sum())
            if count == 0:
                continue
            values = contributions[int(seed)]
            indices = rng.integers(
                0, len(values), size=(count, len(values))
            )
            totals[mask] += values[indices].sum(axis=1)
    differences, ratios = effects_from_totals(totals)
    differences = differences[np.isfinite(differences)]
    finite_ratios = ratios[np.isfinite(ratios)]
    return {
        "rate_difference_lcb": float(np.quantile(differences, 0.025)),
        "rate_difference_ucb": float(np.quantile(differences, 0.975)),
        "risk_ratio_lcb": float(np.quantile(finite_ratios, 0.025)),
        "risk_ratio_ucb": float(np.quantile(finite_ratios, 0.975)),
    }


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic(
    data: pd.DataFrame,
    features: list[str],
    label: str,
    l2: float = 1.0,
) -> dict[str, np.ndarray | float]:
    y = data[label].to_numpy(float)
    weights = game_equal_weights(data)
    if not features:
        prevalence = np.clip(weighted_rate(y, weights), 1e-6, 1 - 1e-6)
        return {
            "features": np.array([], dtype=object),
            "mean": np.array([], dtype=float),
            "scale": np.array([], dtype=float),
            "coef": np.array([], dtype=float),
            "intercept": float(math.log(prevalence / (1 - prevalence))),
        }
    x = data[features].to_numpy(float)
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficients = parameters[1:]
        logits = intercept + z @ coefficients
        probabilities = sigmoid(logits)
        loss = np.sum(
            weights
            * (
                np.logaddexp(0.0, logits)
                - y * logits
            )
        ) / weights.sum()
        loss += 0.5 * l2 * float(coefficients @ coefficients) / len(data)
        residual = weights * (probabilities - y) / weights.sum()
        gradient = np.concatenate(
            ([residual.sum()], z.T @ residual + l2 * coefficients / len(data))
        )
        return float(loss), gradient

    initial_prevalence = np.clip(weighted_rate(y, weights), 1e-6, 1 - 1e-6)
    initial = np.zeros(len(features) + 1, dtype=float)
    initial[0] = math.log(initial_prevalence / (1 - initial_prevalence))
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Logistic fit failed: {result.message}")
    return {
        "features": np.array(features, dtype=object),
        "mean": mean,
        "scale": scale,
        "coef": result.x[1:],
        "intercept": float(result.x[0]),
    }


def predict_logistic(model: dict[str, np.ndarray | float], data: pd.DataFrame) -> np.ndarray:
    features = list(model["features"])
    if not features:
        return np.full(len(data), sigmoid(np.array([model["intercept"]]))[0])
    x = data[features].to_numpy(float)
    z = (x - model["mean"]) / model["scale"]
    return sigmoid(float(model["intercept"]) + z @ model["coef"])


def weighted_average_precision(
    y: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> float:
    positive_weight = float(np.sum(weights * y))
    if positive_weight == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    y = y[order]
    scores = scores[order]
    weights = weights[order]
    true_positive = 0.0
    predicted_weight = 0.0
    average_precision = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        group_weight = float(weights[start:end].sum())
        group_positive = float(np.sum(weights[start:end] * y[start:end]))
        true_positive += group_positive
        predicted_weight += group_weight
        precision = true_positive / predicted_weight
        average_precision += precision * group_positive / positive_weight
        start = end
    return average_precision


def prediction_metrics(
    block: pd.DataFrame, label: str, probabilities: np.ndarray
) -> dict[str, float]:
    y = block[label].to_numpy(float)
    weights = game_equal_weights(block)
    probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
    return {
        "prevalence": weighted_rate(y, weights),
        "auprc": weighted_average_precision(y, probabilities, weights),
        "brier": weighted_rate((probabilities - y) ** 2, weights),
        "log_loss": weighted_rate(
            -(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)),
            weights,
        ),
    }


def loso_metrics(data: pd.DataFrame, labels: Iterable[str]) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    coefficient_rows = []
    for label in labels:
        for test_seed in SEEDS:
            train = data[data["rl_seed"] != test_seed]
            test = data[data["rl_seed"] == test_seed]
            for model_name, features in FEATURE_SETS.items():
                model = fit_logistic(train, features, label)
                probabilities = predict_logistic(model, test)
                rows.append({
                    "label": label,
                    "test_seed": test_seed,
                    "train_seeds": [seed for seed in SEEDS if seed != test_seed],
                    "model": model_name,
                    **prediction_metrics(test, label, probabilities),
                })
                coefficient_rows.append({
                    "label": label,
                    "test_seed": test_seed,
                    "model": model_name,
                    "intercept": float(model["intercept"]),
                    "standardized_coefficients": {
                        str(feature): float(value)
                        for feature, value in zip(model["features"], model["coef"])
                    },
                })
    return pd.DataFrame(rows), coefficient_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-effects", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    args = parser.parse_args()

    paired = pd.read_parquet(args.paired_effects)
    transitions = build_transitions(paired)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transitions.to_parquet(args.output_dir / "transition_dataset.parquet", index=False)

    rng = np.random.default_rng(args.bootstrap_seed)
    association_rows = []
    for label in ("utility_changed", "negative_delta"):
        for seed in SEEDS:
            block = transitions[transitions["rl_seed"] == seed]
            association_rows.append({
                "label": label,
                "rl_seed": str(seed),
                **association(block, label),
                **bootstrap_association(
                    block, label, args.bootstrap_resamples, rng
                ),
            })
        association_rows.append({
            "label": label,
            "rl_seed": "pooled",
            **association(transitions, label),
            **hierarchical_bootstrap_association(
                transitions, label, args.bootstrap_resamples, rng
            ),
        })
    associations = pd.DataFrame(association_rows)
    associations.to_parquet(args.output_dir / "association_metrics.parquet", index=False)

    stratified_rows = []
    for (pre_update, post_update), block in transitions.groupby(
        ["pre_update", "post_update"]
    ):
        stratified_rows.append({
            "stratum": "update_window",
            "stratum_value": f"{pre_update}->{post_update}",
            **association(block, "utility_changed"),
        })
    for skill_id, block in transitions.groupby("skill_id"):
        stratified_rows.append({
            "stratum": "skill",
            "stratum_value": skill_id,
            **association(block, "utility_changed"),
        })
    for (seed, skill_id), block in transitions.groupby(["rl_seed", "skill_id"]):
        stratified_rows.append({
            "stratum": "seed_skill",
            "stratum_value": f"{seed}:{skill_id}",
            **association(block, "utility_changed"),
        })
    stratified = pd.DataFrame(stratified_rows)
    stratified.to_parquet(
        args.output_dir / "stratified_association_metrics.parquet", index=False
    )

    loso, coefficients = loso_metrics(
        transitions, ("utility_changed", "negative_delta")
    )
    loso.to_parquet(args.output_dir / "loso_metrics.parquet", index=False)

    post_rows = paired[paired["global_update"] > 0]
    full_suffix_steps = int(
        sum(
            post_rows[column].sum()
            for column in (
                "suffix_trajectory_length_original",
                "suffix_trajectory_length_placebo",
                "suffix_trajectory_length_null",
            )
        )
    )
    one_step_decisions = len(post_rows) * 2
    atomic_write_json(args.output_dir / "summary.json", {
        "schema_version": "phase1.minimal_interaction_forecast_results.v1",
        "transition_rows": len(transitions),
        "seed_count": len(SEEDS),
        "windows_per_seed": len(WINDOWS),
        "distinct_games": int(transitions["game_id"].nunique()),
        "distinct_skills": int(transitions["skill_id"].nunique()),
        "utility_changed_count": int(transitions["utility_changed"].sum()),
        "negative_delta_count": int(transitions["negative_delta"].sum()),
        "point_harmful_flip_count": int(transitions["point_harmful_flip"].sum()),
        "association_metrics": association_rows,
        "stratified_association_metrics": stratified_rows,
        "loso_metrics": loso.to_dict(orient="records"),
        "loso_coefficients": coefficients,
        "decision_cost": {
            "post_checkpoint_original_placebo_one_step_decisions": one_step_decisions,
            "archived_full_gold_suffix_decisions": full_suffix_steps,
            "one_step_fraction_of_full_suffix_decisions": one_step_decisions / full_suffix_steps,
        },
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
    })
    print(json.dumps({
        "transition_rows": len(transitions),
        "utility_changed_count": int(transitions["utility_changed"].sum()),
        "negative_delta_count": int(transitions["negative_delta"].sum()),
        "point_harmful_flip_count": int(transitions["point_harmful_flip"].sum()),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
