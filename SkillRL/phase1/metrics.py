from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

PAIR_FIELDS = (
    "checkpoint_id",
    "update_type",
    "rl_seed",
    "eval_seed",
    "environment_seed",
    "game_id",
    "context_id",
    "skill_id",
)


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    n_clusters: int


def clustered_bootstrap_interval(
    values: pd.DataFrame,
    *,
    value_column: str,
    cluster_column: str = "game_id",
    resamples: int = 2000,
    seed: int = 0,
) -> Interval:
    if values.empty:
        return Interval(np.nan, np.nan, np.nan, 0)
    clusters = values.groupby(cluster_column, dropna=False)[value_column].mean()
    data = clusters.to_numpy(dtype=float)
    estimate = float(np.mean(data))
    if len(data) == 1:
        return Interval(estimate, estimate, estimate, 1)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(resamples, len(data)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return Interval(estimate, float(lower), float(upper), len(data))


def matched_skill_differences(records: pd.DataFrame) -> pd.DataFrame:
    required = set(PAIR_FIELDS) | {"skill_condition", "success"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Missing required evaluator columns: {sorted(missing)}")
    duplicated = records.duplicated([*PAIR_FIELDS, "skill_condition"], keep=False)
    if duplicated.any():
        raise ValueError("Duplicate rollout unique keys detected")
    # In the step-routed protocol a target Skill is attributable only when it
    # was actually selected in the FULL_BANK trajectory.  Preserve the v1
    # behavior for legacy rows, which have no router version.
    if "skill_router_version" in records and "selected_skill_ids" in records:
        full_rows = records[records["skill_condition"] == "full_bank"]
        eligible_keys = set()
        for _, row in full_rows.iterrows():
            routed = pd.notna(row.get("skill_router_version"))
            selected = row.get("selected_skill_ids", [])
            selected = selected if isinstance(selected, (list, tuple, set, np.ndarray)) else []
            if not routed or row["skill_id"] in selected:
                eligible_keys.add(tuple(row[field] for field in PAIR_FIELDS))
        records = records[
            records.apply(
                lambda row: tuple(row[field] for field in PAIR_FIELDS) in eligible_keys,
                axis=1,
            )
        ]
    wide = records.pivot(index=list(PAIR_FIELDS), columns="skill_condition", values="success")
    if "full_bank" not in wide or "minus_skill" not in wide:
        return pd.DataFrame(columns=[*PAIR_FIELDS, "margin"])
    complete = wide[["full_bank", "minus_skill"]].dropna().copy()
    complete["margin"] = complete["full_bank"].astype(float) - complete["minus_skill"].astype(float)
    return complete.reset_index()


def classify_transition(
    pre_lower: float,
    pre_upper: float,
    post_lower: float,
    post_upper: float,
) -> str:
    values = (pre_lower, pre_upper, post_lower, post_upper)
    if any(pd.isna(value) for value in values):
        return "ambiguous"
    if pre_lower > 0 and post_upper < 0:
        return "harmful_sign_flip"
    if pre_upper < 0 and post_lower > 0:
        return "beneficial_sign_flip"
    if pre_lower > 0 and post_lower > 0:
        return "stable_positive"
    if pre_upper < 0 and post_upper < 0:
        return "stable_negative"
    return "ambiguous"


def checkpoint_order(value) -> tuple[int, str]:
    text = str(value)
    match = re.search(r"(?:checkpoint|global[_-]?step)[_-]?(\d+)", text, re.IGNORECASE)
    # Model identifiers often contain size/version numbers (for example,
    # Qwen2.5-1.5B-Instruct).  They represent the base checkpoint and must not
    # be ordered by the last incidental digit in the model name.
    return (int(match.group(1)) if match else 0, text)


def validate_matched_configuration(
    records: pd.DataFrame,
    shared_fields: Sequence[str] = (
        "environment_seed", "eval_seed", "temperature", "top_p", "max_steps",
        "action_projection_version", "prompt_template_version", "skill_bank_hash",
    ),
) -> list[str]:
    reasons: list[str] = []
    key_fields = list(PAIR_FIELDS)
    for _, block in records.groupby(key_fields, dropna=False):
        conditions = set(block["skill_condition"])
        if not {"full_bank", "minus_skill"}.issubset(conditions):
            reasons.append("incomplete_rollout")
            continue
        for field in shared_fields:
            if field in block and block[field].nunique(dropna=False) != 1:
                reasons.append("configuration_mismatch")
                break
    return reasons


def summarize_margins(
    records: pd.DataFrame,
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = matched_skill_differences(records)
    group_fields = ["checkpoint_id", "update_type", "rl_seed", "skill_id", "context_id"]
    metric_rows = []
    interval_rows = []
    for group_key, block in paired.groupby(group_fields, dropna=False):
        interval = clustered_bootstrap_interval(
            block,
            value_column="margin",
            resamples=resamples,
            seed=seed,
        )
        row = dict(zip(group_fields, group_key))
        row.update({
            "margin": interval.estimate,
            "margin_lcb": interval.lower,
            "margin_ucb": interval.upper,
            "n_game_clusters": interval.n_clusters,
        })
        metric_rows.append(row)
        interval_rows.append({**row, "metric": "margin"})
    return pd.DataFrame(metric_rows), pd.DataFrame(interval_rows)


def transition_metrics(margins: pd.DataFrame) -> pd.DataFrame:
    if margins.empty:
        return pd.DataFrame()
    rows = []
    group_fields = ["update_type", "rl_seed", "skill_id", "context_id"]
    for group_key, block in margins.groupby(group_fields, dropna=False):
        ordered = block.assign(
            _checkpoint_order=block["checkpoint_id"].map(checkpoint_order)
        ).sort_values("_checkpoint_order")
        for (_, pre), (_, post) in zip(ordered.iloc[:-1].iterrows(), ordered.iloc[1:].iterrows()):
            rows.append({
                **dict(zip(group_fields, group_key)),
                "pre_checkpoint_id": pre["checkpoint_id"],
                "post_checkpoint_id": post["checkpoint_id"],
                "pre_margin": pre["margin"],
                "post_margin": post["margin"],
                "pre_margin_lcb": pre["margin_lcb"],
                "pre_margin_ucb": pre["margin_ucb"],
                "post_margin_lcb": post["margin_lcb"],
                "post_margin_ucb": post["margin_ucb"],
                "delta_margin": post["margin"] - pre["margin"],
                "label": classify_transition(
                    pre["margin_lcb"], pre["margin_ucb"],
                    post["margin_lcb"], post["margin_ucb"],
                ),
            })
    return pd.DataFrame(rows)


def endpoint_transition_metrics(margins: pd.DataFrame) -> pd.DataFrame:
    """Compare the first and last checkpoint while preserving the strict labels."""
    if margins.empty:
        return pd.DataFrame()
    rows = []
    group_fields = ["update_type", "rl_seed", "skill_id", "context_id"]
    for group_key, block in margins.groupby(group_fields, dropna=False):
        ordered = block.assign(
            _checkpoint_order=block["checkpoint_id"].map(checkpoint_order)
        ).sort_values("_checkpoint_order")
        if len(ordered) < 2:
            continue
        pre = ordered.iloc[0]
        post = ordered.iloc[-1]
        rows.append({
            **dict(zip(group_fields, group_key)),
            "pre_checkpoint_id": pre["checkpoint_id"],
            "post_checkpoint_id": post["checkpoint_id"],
            "pre_margin": pre["margin"],
            "post_margin": post["margin"],
            "pre_margin_lcb": pre["margin_lcb"],
            "pre_margin_ucb": pre["margin_ucb"],
            "post_margin_lcb": post["margin_lcb"],
            "post_margin_ucb": post["margin_ucb"],
            "delta_margin": post["margin"] - pre["margin"],
            "label": classify_transition(
                pre["margin_lcb"], pre["margin_ucb"],
                post["margin_lcb"], post["margin_ucb"],
            ),
        })
    return pd.DataFrame(rows)


def stability_metrics(transitions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if transitions.empty:
        return pd.DataFrame(), {"events": 0}
    event_fields = ["update_type", "skill_id", "context_id", "pre_checkpoint_id", "post_checkpoint_id"]
    rows = []
    for key, block in transitions.groupby(event_fields, dropna=False):
        signs = np.sign(block["delta_margin"].to_numpy(dtype=float))
        nonzero = signs[signs != 0]
        agreement = float(max(np.mean(nonzero > 0), np.mean(nonzero < 0))) if len(nonzero) else np.nan
        rows.append({
            **dict(zip(event_fields, key)),
            "n_rl_seeds": int(block["rl_seed"].nunique()),
            "sign_agreement": agreement,
            "at_least_two_of_three_same_direction": bool(len(nonzero) >= 2 and agreement >= 2 / 3),
        })
    event_rows = pd.DataFrame(rows)
    correlations = []
    indexed = transitions.pivot_table(index=event_fields, columns="rl_seed", values="delta_margin", aggfunc="mean")
    seed_columns = list(indexed.columns)
    for left_index, left in enumerate(seed_columns):
        for right in seed_columns[left_index + 1:]:
            correlations.append(indexed[left].corr(indexed[right], method="spearman"))
    summary = {
        "events": len(event_rows),
        "two_of_three_direction_rate": float(event_rows["at_least_two_of_three_same_direction"].mean()),
        "mean_pairwise_seed_spearman": float(np.nanmean(correlations)) if correlations else np.nan,
    }
    return event_rows, summary


def game_level_transition_deltas(records: pd.DataFrame) -> pd.DataFrame:
    paired = matched_skill_differences(records)
    fields = ["checkpoint_id", "update_type", "rl_seed", "skill_id", "context_id", "game_id"]
    game_margins = paired.groupby(fields, dropna=False)["margin"].mean().reset_index()
    rows = []
    group_fields = ["update_type", "rl_seed", "skill_id", "context_id", "game_id"]
    for key, block in game_margins.groupby(group_fields, dropna=False):
        ordered = block.assign(
            _checkpoint_order=block["checkpoint_id"].map(checkpoint_order)
        ).sort_values("_checkpoint_order")
        for (_, pre), (_, post) in zip(ordered.iloc[:-1].iterrows(), ordered.iloc[1:].iterrows()):
            rows.append({
                **dict(zip(group_fields, key)),
                "pre_checkpoint_id": pre["checkpoint_id"],
                "post_checkpoint_id": post["checkpoint_id"],
                "delta_margin": float(post["margin"] - pre["margin"]),
            })
    return pd.DataFrame(rows)


def control_comparisons(records: pd.DataFrame, *, permutation_resamples: int = 10000, seed: int = 0) -> pd.DataFrame:
    deltas = game_level_transition_deltas(records)
    if deltas.empty or "real_rl" not in set(deltas["update_type"]):
        return pd.DataFrame()
    key_fields = [
        "rl_seed", "skill_id", "context_id", "game_id",
        "pre_checkpoint_id", "post_checkpoint_id",
    ]
    real = deltas[deltas["update_type"] == "real_rl"].set_index(key_fields)["delta_margin"]
    rows = []
    rng = np.random.default_rng(seed)
    for control_type in ("zero_update", "shuffled_reward", "random_parameter"):
        control = deltas[deltas["update_type"] == control_type].set_index(key_fields)["delta_margin"]
        paired = pd.concat({"real": real, "control": control}, axis=1).dropna()
        if paired.empty:
            continue
        difference = paired["real"].abs() - paired["control"].abs()
        observed = float(difference.mean())
        signs = rng.choice(np.array([-1.0, 1.0]), size=(permutation_resamples, len(difference)))
        permuted = (signs * difference.to_numpy()).mean(axis=1)
        p_value = float((np.count_nonzero(permuted >= observed) + 1) / (permutation_resamples + 1))
        control_median = float(paired["control"].abs().median())
        rows.append({
            "control_type": control_type,
            "n_matched_game_transitions": len(paired),
            "real_median_abs_delta_margin": float(paired["real"].abs().median()),
            "control_median_abs_delta_margin": control_median,
            "median_abs_delta_ratio": (
                float(paired["real"].abs().median() / control_median)
                if control_median > 0 else np.inf
            ),
            "mean_paired_abs_delta_difference": observed,
            "cluster_permutation_p_value": p_value,
        })
    return pd.DataFrame(rows)


def abstention_rows(records: pd.DataFrame, margins: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_fields = ["checkpoint_id", "update_type", "rl_seed", "skill_id", "context_id"]
    for _, margin in margins.iterrows():
        identifiers = {field: margin[field] for field in group_fields}
        if int(margin["n_game_clusters"]) < 8:
            rows.append({**identifiers, "abstain_reason": "low_support"})
        block = records
        for field in group_fields:
            block = block[block[field] == margin[field]]
        full = block[block["skill_condition"] == "full_bank"]
        if "skill_router_version" in full and full["skill_router_version"].notna().any():
            target_skill_id = margin["skill_id"]
            triggered = full["selected_skill_ids"].map(
                lambda values, skill_id=target_skill_id: skill_id
                in (values if isinstance(values, (list, tuple, set, np.ndarray)) else [])
            ).any()
            if not triggered:
                rows.append({**identifiers, "abstain_reason": "skill_not_selected"})
        elif "retrieved_skill_ids" in full:
            target_skill_id = margin["skill_id"]
            triggered = full["retrieved_skill_ids"].map(
                lambda values, skill_id=target_skill_id: skill_id
                in (values if isinstance(values, (list, tuple, set, np.ndarray)) else [])
            ).any()
            if not triggered:
                rows.append({**identifiers, "abstain_reason": "skill_not_triggered"})

    transition_fields = ["update_type", "rl_seed", "skill_id", "context_id", "pre_checkpoint_id", "post_checkpoint_id"]
    for _, transition in transitions.iterrows():
        identifiers = {field: transition[field] for field in transition_fields}
        if transition["pre_margin_lcb"] <= 0 <= transition["pre_margin_ucb"]:
            rows.append({**identifiers, "abstain_reason": "pre_margin_ambiguous"})
        if transition["post_margin_lcb"] <= 0 <= transition["post_margin_ucb"]:
            rows.append({**identifiers, "abstain_reason": "post_margin_ambiguous"})

    rows.extend({"abstain_reason": reason} for reason in validate_matched_configuration(records))
    return pd.DataFrame(rows)
