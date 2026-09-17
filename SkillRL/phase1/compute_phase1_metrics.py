#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phase1.archive import atomic_write_json
from phase1.metrics import (
    abstention_rows,
    control_comparisons,
    endpoint_transition_metrics,
    stability_metrics,
    summarize_margins,
    transition_metrics,
)


def load_records(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_json(path, lines=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    records = pd.concat(
        [load_records(path) for path in args.input],
        ignore_index=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    margins, intervals = summarize_margins(
        records,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    transitions = transition_metrics(margins)
    endpoint_transitions = endpoint_transition_metrics(margins)
    stability, stability_summary = stability_metrics(transitions)
    controls = control_comparisons(records, seed=args.bootstrap_seed)
    abstentions = abstention_rows(records, margins, transitions)

    margins.to_parquet(args.output_dir / "skill_context_metrics.parquet", index=False)
    transitions.to_parquet(args.output_dir / "checkpoint_transition_metrics.parquet", index=False)
    endpoint_transitions.to_parquet(
        args.output_dir / "checkpoint_endpoint_transition_metrics.parquet", index=False
    )
    intervals.to_parquet(args.output_dir / "bootstrap_intervals.parquet", index=False)
    stability.to_parquet(args.output_dir / "stability_metrics.parquet", index=False)
    controls.to_parquet(args.output_dir / "control_comparisons.parquet", index=False)
    counts = abstentions.get("abstain_reason", pd.Series(dtype="object")).value_counts().to_dict()
    atomic_write_json(args.output_dir / "abstention_report.json", {
        "abstention_count": len(abstentions),
        "by_reason": counts,
        "records": abstentions.to_dict(orient="records"),
    })
    label_counts = transitions.get("label", pd.Series(dtype="object")).value_counts().to_dict()
    strict_flips = int(label_counts.get("harmful_sign_flip", 0))
    non_ambiguous = max(1, len(transitions) - int(label_counts.get("ambiguous", 0)))
    atomic_write_json(args.output_dir / "phase1_summary.json", {
        "rollout_records": len(records),
        "skill_context_checkpoint_rows": len(margins),
        "checkpoint_transitions": len(transitions),
        "checkpoint_endpoint_transitions": len(endpoint_transitions),
        "label_counts": label_counts,
        "strict_harmful_flips": strict_flips,
        "strict_harmful_flip_rate_non_ambiguous": strict_flips / non_ambiguous,
        "bootstrap_resamples": args.bootstrap_resamples,
        "stability": stability_summary,
        "control_comparisons": controls.to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()
