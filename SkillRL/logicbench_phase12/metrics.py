"""Skill-level ranking metrics; utility labels are read only in this module."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

METHODS = ("D_signed_gate", "D_original", "D_signed", "D_real",
           "D_factor", "D_orientation", "M_delta_centered", "M_delta_raw")


def _finite_or_none(value):
    return float(value) if math.isfinite(float(value)) else None


def measure_method(scores, delta_m, *, threshold=0.):
    scores = np.asarray(scores, dtype=float)
    delta = np.asarray(delta_m, dtype=float)
    if (scores.ndim != 1 or scores.shape != delta.shape or len(scores) == 0
            or not np.isfinite(scores).all() or not np.isfinite(delta).all()):
        raise ValueError("Finite skill-level score/utility vectors are required")
    labels = delta < -threshold
    inc = delta > threshold
    out = {"n_skills": len(scores), "n_declines": int(labels.sum()),
           "n_increases": int(inc.sum()), "threshold": threshold}
    if 0 < labels.sum() < len(labels):
        out["average_precision"] = float(average_precision_score(labels, scores))
        out["auroc_decline_vs_rest"] = float(roc_auc_score(labels, scores))
    else:
        out["average_precision"] = None
        out["auroc_decline_vs_rest"] = None
    keep = labels | inc
    out["auroc_decline_vs_increase"] = (
        float(roc_auc_score(labels[keep], scores[keep]))
        if labels[keep].sum() and inc[keep].sum() else None)
    severity = -delta
    out["spearman_decline_severity"] = (
        _finite_or_none(spearmanr(scores, severity).statistic) if len(scores) >= 3 else None)
    out["pearson_decline_severity"] = (
        _finite_or_none(pearsonr(scores, severity).statistic) if len(scores) >= 3 else None)
    out["kendall_decline_severity"] = (
        _finite_or_none(kendalltau(scores, severity).statistic) if len(scores) >= 3 else None)
    magnitude = np.abs(delta)
    large_change = magnitude > threshold
    out["n_large_absolute_changes"] = int(large_change.sum())
    out["absolute_change_spearman"] = (
        _finite_or_none(spearmanr(scores, magnitude).statistic) if len(scores) >= 3 else None)
    out["absolute_change_pearson"] = (
        _finite_or_none(pearsonr(scores, magnitude).statistic) if len(scores) >= 3 else None)
    out["absolute_change_kendall"] = (
        _finite_or_none(kendalltau(scores, magnitude).statistic) if len(scores) >= 3 else None)
    if 0 < large_change.sum() < len(large_change):
        out["absolute_change_average_precision"] = float(
            average_precision_score(large_change, scores))
        out["absolute_change_auroc"] = float(roc_auc_score(large_change, scores))
    else:
        out["absolute_change_average_precision"] = None
        out["absolute_change_auroc"] = None
    order = np.argsort(-scores, kind="stable")
    for k in (1, 3, 5):
        top = order[:min(k, len(scores))]
        out[f"precision_at_{k}"] = float(labels[top].mean())
        out[f"recall_at_{k}"] = float(labels[top].sum() / labels.sum()) if labels.sum() else None
        out[f"absolute_change_precision_at_{k}"] = float(large_change[top].mean())
        out[f"absolute_change_recall_at_{k}"] = (
            float(large_change[top].sum() / large_change.sum())
            if large_change.sum() else None)
    return out


def run(readout: Path, utility: Path, output: Path):
    scores = pd.read_csv(readout)
    target = pd.DataFrame(json.loads(utility.read_text()))
    if scores.skill_id.duplicated().any() or target.skill_id.duplicated().any():
        raise ValueError("Duplicate skill IDs in readout or independent utility")
    merged = scores.merge(target, on="skill_id", how="left", validate="one_to_one")
    if merged.empty:
        raise ValueError("No invoked skills were scored")
    missing = merged[merged.delta_m.isna()].skill_id.tolist()
    included = merged[merged.delta_m.notna()].copy()
    report = {"schema_version": "skillscope.logicbench_phase12_metrics.v1",
              "invoked_skills": len(merged), "evaluated_invoked_skills": len(included),
              "no_independent_eval_anchor": missing,
              "readout": str(readout), "independent_utility": str(utility),
              "thresholds": {}}
    for threshold in (0., 0.05):
        report["thresholds"][str(threshold)] = {
            method: measure_method(included[method], included.delta_m,
                                   threshold=threshold)
            for method in METHODS}
    output.mkdir(parents=True, exist_ok=False)
    merged.to_csv(output / "skill_scores_with_utility.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--utility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.readout, args.utility, args.output)


if __name__ == "__main__":
    main()
