"""Training-free within-window ranking with fixed signs and gold-blind ties."""
import math

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def selection_weights(scores, k):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or not 1 <= k <= len(scores):
        raise ValueError("Finite scores and a valid within-pool budget are required")
    weights = np.zeros(len(scores))
    remaining = k
    for score in sorted(set(scores), reverse=True):
        indices = np.flatnonzero(scores == score)
        take = min(remaining, len(indices))
        weights[indices] = take/len(indices)
        remaining -= take
        if remaining == 0: break
    return weights


def budget_metrics(scores, delta, k, threshold=0., target="decline"):
    delta = np.asarray(delta, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if delta.shape != scores.shape or not np.isfinite(delta).all(): raise ValueError("Matched finite gold required")
    if target not in ("decline", "any_change"): raise ValueError("Unknown ranking target")
    quantity = -delta if target == "decline" else abs(delta)
    gain = np.maximum(quantity, 0.)
    labels = quantity > threshold+1e-12
    weights = selection_weights(scores, k)
    hits = float(weights @ labels)
    total_gain = float(gain.sum())
    variable = len(scores) > 1 and len(set(scores)) > 1 and len(set(quantity)) > 1
    return {"k": k, "n_candidates": len(scores), "events": int(labels.sum()), "threshold": threshold,
            "target": target, "expected_hits": hits, "precision_at_k": hits/k,
            "recall_at_k": hits/labels.sum() if labels.any() else np.nan,
            "captured_change_mass": float(weights @ gain)/total_gain if total_gain > 1e-12 else np.nan,
            "spearman": float(spearmanr(scores, quantity).statistic) if variable else np.nan,
            "kendall": float(kendalltau(scores, quantity).statistic) if variable else np.nan}


def score_snapshot(features, settings):
    result = []
    for _, row in features.iterrows():
        scores = {}
        raw = {}
        for name, orientation in settings["scores"].items():
            if orientation not in (-1, 1): raise ValueError("Prespecify every score orientation")
            value = row.get(name, np.nan)
            raw[name] = float(value) if pd.notna(value) and np.isfinite(value) else None
            scores[name] = orientation*raw[name] if raw[name] is not None else None
        for name in ("C_upd", "C_upd_centered", "advantage", "gate_coverage", "direction_coverage", "nonzero_advantage_decisions", "training_games"):
            value = row.get(name, np.nan)
            raw[name] = float(value) if pd.notna(value) and np.isfinite(value) else None
        scores["random_expected"] = 0.
        result.append({"skill_id": row.skill_id, "context_id": row.context_id, "phase": row.phase,
                       "supported": bool(row.supported), "raw_features": raw, "scores": scores})
    return result


def evaluate_snapshot(snapshot, labels, settings):
    """Labels are joined only AFTER the previously frozen score snapshot."""
    scores = pd.DataFrame([{**{k: row[k] for k in ("skill_id", "context_id", "phase", "supported")},
                            **row["scores"]} for row in snapshot])
    keys = ["skill_id", "context_id", "phase"]
    table = scores.merge(labels[keys+["delta_utility"]], on=keys, validate="one_to_one", how="left")
    names = list(settings["scores"])+["random_expected"]
    results, support = [], []
    for (context, phase), group in table.groupby(["context_id", "phase"]):
        q = group[group.supported].dropna(subset=names+["delta_utility"])
        support.append({"context_id": context, "phase": phase, "candidate_units": len(group),
                        "shared_supported_units": len(q), "excluded_units": len(group)-len(q),
                        "shared_skill_ids": q.skill_id.tolist()})
        if q.empty: continue
        budgets = {min(int(k), len(q)) for k in settings["budgets_k"]}
        budgets.update(max(1, math.ceil(f*len(q))) for f in settings["budgets_fraction"])
        for target in ("decline", "any_change"):
            for threshold in settings["event_thresholds"]:
                for name in names:
                    for k in sorted(budgets):
                        results.append({"context_id": context, "phase": phase, "score": name,
                            **budget_metrics(q[name], q.delta_utility, k, threshold, target)})
    return pd.DataFrame(results), support, table
