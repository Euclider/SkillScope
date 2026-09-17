#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from phase1.archive import append_jsonl_idempotent, atomic_write_json


def phase(step: int) -> str:
    if step == 0:
        return "initial"
    if step <= 4:
        return "early"
    if step <= 14:
        return "middle"
    return "late"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--combined-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        rows.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    append_jsonl_idempotent(
        args.combined_index,
        rows,
        unique_fields=(
            "checkpoint_id", "update_type", "rl_seed", "eval_seed",
            "game_id", "skill_id", "skill_condition",
        ),
    )

    trajectories = [json.loads(Path(row["trajectory_path"]).read_text(encoding="utf-8")) for row in rows]
    total_steps = sum(len(item["steps"]) for item in trajectories)
    parsed_steps = sum(
        bool(step.get("is_action_valid"))
        for trajectory in trajectories for step in trajectory["steps"]
    )
    admissible_steps = sum(
        bool(step.get("is_action_admissible"))
        for trajectory in trajectories for step in trajectory["steps"]
    )
    skill_trajectories: dict[str, set[str]] = defaultdict(set)
    skill_games: dict[str, set[str]] = defaultdict(set)
    first_steps: dict[str, list[int]] = defaultdict(list)
    phase_counts: dict[str, Counter[str]] = defaultdict(Counter)
    selection_counts: Counter[str] = Counter()
    for trajectory in trajectories:
        first_by_skill = {}
        for step in trajectory["steps"]:
            skill = step.get("selected_skill_id")
            if not skill:
                continue
            selection_counts[skill] += 1
            first_by_skill.setdefault(skill, int(step["step_index"]))
        for skill, first_step in first_by_skill.items():
            skill_trajectories[skill].add(trajectory["trajectory_id"])
            skill_games[skill].add(trajectory["game_id"])
            first_steps[skill].append(first_step)
            phase_counts[skill][phase(first_step)] += 1

    episode_count = len(trajectories)
    success_count = sum(bool(item["success"]) for item in trajectories)
    parse_rate = parsed_steps / total_steps if total_steps else 0.0
    admissible_rate = admissible_steps / total_steps if total_steps else 0.0
    non_initial_supported = any(any(step > 0 for step in values) for values in first_steps.values())
    payload = {
        "schema_version": "phase1.qwen35_b0_clean_summary.v1",
        "episode_count": episode_count,
        "distinct_game_count": len({item["game_id"] for item in trajectories}),
        "eval_seeds": sorted({item["eval_seed"] for item in trajectories}),
        "success_count": success_count,
        "success_rate": success_count / episode_count if episode_count else 0.0,
        "total_action_steps": total_steps,
        "action_parse_rate": parse_rate,
        "action_admissible_rate": admissible_rate,
        "mean_trajectory_length": total_steps / episode_count if episode_count else 0.0,
        "mean_distinct_selected_skills": (
            sum(item["unique_skill_count"] for item in trajectories) / episode_count
            if episode_count else 0.0
        ),
        "skill_selection_counts": dict(sorted(selection_counts.items())),
        "skill_support": {
            skill: {
                "trajectory_count": len(skill_trajectories[skill]),
                "distinct_game_count": len(skill_games[skill]),
                "first_step_min": min(steps),
                "first_step_max": max(steps),
                "first_step_mean": sum(steps) / len(steps),
                "first_step_phase_counts": dict(sorted(phase_counts[skill].items())),
            }
            for skill, steps in sorted(first_steps.items())
        },
        "base_gate": {
            "parse_rate_at_least_0_98": parse_rate >= 0.98,
            "admissible_rate_at_least_0_98": admissible_rate >= 0.98,
            "episode_success_nonzero": success_count > 0,
            "multiple_naturally_selected_skills": len(first_steps) >= 2,
            "has_non_initial_first_invocation": non_initial_supported,
        },
    }
    payload["base_gate"]["passed"] = all(payload["base_gate"].values())
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
