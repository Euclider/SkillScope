#!/usr/bin/env python3
"""Audit and summarize archived v2 step-routed trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from phase1.archive import atomic_write_json, utc_now


def summarize_trajectories(trajectories: list[dict]) -> dict:
    selection_counts: Counter[str] = Counter()
    context_selection_counts: dict[str, Counter[str]] = {}
    router_versions: set[str] = set()
    unique_counts = []
    lengths = []
    returns = []
    routing_switches = []
    violations: list[dict] = []

    for trajectory in trajectories:
        trajectory_id = trajectory.get("trajectory_id")
        context = str(trajectory.get("context_id") or "unknown")
        context_counter = context_selection_counts.setdefault(context, Counter())
        selected_sequence = []
        for step in trajectory.get("steps", []):
            selected = step.get("selected_skill_id")
            injected = list(step.get("injected_skill_ids", []))
            candidates = list(step.get("candidate_skill_ids", []))
            version = step.get("skill_router_version")
            if version:
                router_versions.add(str(version))
            if version and (not selected or injected != [selected]):
                violations.append({
                    "trajectory_id": trajectory_id,
                    "step_index": step.get("step_index"),
                    "type": "selected_injected_mismatch",
                    "selected_skill_id": selected,
                    "injected_skill_ids": injected,
                })
            if selected and selected not in candidates:
                violations.append({
                    "trajectory_id": trajectory_id,
                    "step_index": step.get("step_index"),
                    "type": "selected_not_in_candidates",
                    "selected_skill_id": selected,
                })
            if selected:
                selected_sequence.append(str(selected))
                selection_counts[str(selected)] += 1
                context_counter[str(selected)] += 1

        unique_counts.append(len(set(selected_sequence)))
        lengths.append(len(trajectory.get("steps", [])))
        returns.append(float(trajectory.get("episode_return") or 0.0))
        routing_switches.append(sum(
            left != right
            for left, right in zip(selected_sequence, selected_sequence[1:])
        ))

    count = len(trajectories)
    return {
        "schema_version": "phase1.step_routing_summary.v1",
        "created_at": utc_now(),
        "trajectory_count": count,
        "successful_trajectory_count": sum(value > 0 for value in returns),
        "success_rate": sum(value > 0 for value in returns) / count if count else None,
        "mean_trajectory_length": sum(lengths) / count if count else None,
        "mean_distinct_selected_skills": sum(unique_counts) / count if count else None,
        "max_distinct_selected_skills": max(unique_counts, default=0),
        "multi_skill_trajectory_count": sum(value >= 2 for value in unique_counts),
        "mean_routing_switches": sum(routing_switches) / count if count else None,
        "skill_selection_counts": dict(sorted(selection_counts.items())),
        "context_skill_selection_counts": {
            context: dict(sorted(counter.items()))
            for context, counter in sorted(context_selection_counts.items())
        },
        "skill_router_versions": sorted(router_versions),
        "routing_invariant_violation_count": len(violations),
        "routing_invariant_violations": violations,
        "pilot_gate": {
            "selected_skill_matches_injected_skill_each_step": not violations,
            "router_version_constant": len(router_versions) == 1,
            "at_least_two_distinct_selected_skills_in_some_multistep_trajectories": any(
                value >= 2 for value in unique_counts
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-index", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trajectories = []
    for line in args.trajectory_index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("run_id") != args.run_id:
            continue
        path = Path(row["trajectory_path"])
        trajectories.append(json.loads(path.read_text(encoding="utf-8")))
    if not trajectories:
        raise ValueError(f"No trajectories found for run_id={args.run_id!r}")
    payload = {"run_id": args.run_id, **summarize_trajectories(trajectories)}
    atomic_write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
