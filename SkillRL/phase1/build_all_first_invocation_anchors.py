#!/usr/bin/env python3
"""Build and audit first-natural-invocation anchors for every eligible Skill."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from phase1.archive import atomic_write_json
from phase1.first_invocation import build_anchor

PHASE_BINS = {
    "initial": lambda step: step == 0,
    "early": lambda step: 1 <= step <= 4,
    "middle": lambda step: 5 <= step <= 14,
    "late": lambda step: step >= 15,
}


def load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def eligible_skills(bank_path: Path, context: str) -> list[dict[str, str]]:
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    skills = list(bank["general_skills"]) + list(bank["task_specific_skills"][context])
    return [
        {
            "skill_id": item["skill_id"],
            "title": item.get("title", ""),
            "kind": "general" if item["skill_id"].startswith("gen_") else "task_specific",
        }
        for item in skills
    ]


def phase_name(step: int) -> str:
    return next(name for name, predicate in PHASE_BINS.items() if predicate(step))


def round_robin_games(anchors: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Deterministically cap support without letting repeated seeds dominate."""
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        by_game[anchor["game_id"]].append(anchor)
    for values in by_game.values():
        values.sort(key=lambda row: (row["source_eval_seed"], row["trigger_step"], row["anchor_id"]))
    games = sorted(by_game)
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < min(limit, len(anchors)):
        appended = False
        for game in games:
            values = by_game[game]
            if round_index < len(values):
                selected.append(values[round_index])
                appended = True
                if len(selected) == limit:
                    break
        if not appended:
            break
        round_index += 1
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-index", nargs="+", type=Path, required=True)
    parser.add_argument("--skill-bank", type=Path, required=True)
    parser.add_argument("--context-id", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--condition", default="full_bank")
    parser.add_argument("--minimum-occurrences", type=int, default=30)
    parser.add_argument("--minimum-games", type=int, default=10)
    parser.add_argument("--maximum-selected", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).parents[1]
    indices = load_jsonl(args.trajectory_index)
    eligible = eligible_skills(args.skill_bank, args.context_id)
    by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_trajectories = 0
    seen_sources: set[str] = set()
    for index in indices:
        if index.get("checkpoint_id") != args.checkpoint_id:
            continue
        if index.get("skill_condition") != args.condition:
            continue
        if index.get("context_id") != args.context_id:
            continue
        trajectory_path = Path(index["trajectory_path"])
        if not trajectory_path.is_absolute():
            trajectory_path = repo_root / trajectory_path
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        if trajectory["trajectory_id"] not in seen_sources:
            source_trajectories += 1
            seen_sources.add(trajectory["trajectory_id"])
        naturally_selected = {
            step.get("selected_skill_id")
            for step in trajectory.get("steps", [])
            if step.get("selected_skill_id")
        }
        for skill_id in naturally_selected:
            anchor = build_anchor(
                trajectory=trajectory,
                trajectory_path=str(trajectory_path),
                skill_id=skill_id,
                source_index=index,
            )
            if anchor is not None:
                anchor["trigger_phase"] = phase_name(int(anchor["trigger_step"]))
                by_skill[skill_id].append(anchor)

    coverage = []
    for metadata in eligible:
        skill_id = metadata["skill_id"]
        all_anchors = sorted(
            by_skill.get(skill_id, []),
            key=lambda row: (row["game_id"], row["source_eval_seed"], row["trigger_step"], row["anchor_id"]),
        )
        distinct_games = len({row["game_id"] for row in all_anchors})
        supported = len(all_anchors) >= args.minimum_occurrences and distinct_games >= args.minimum_games
        selected = round_robin_games(all_anchors, args.maximum_selected) if supported else []
        write_jsonl(args.output_dir / "all" / f"{skill_id}.jsonl", all_anchors)
        if supported:
            write_jsonl(args.output_dir / "selected" / f"{skill_id}.jsonl", selected)
        phase_counts = {
            phase: sum(row["trigger_phase"] == phase for row in all_anchors)
            for phase in PHASE_BINS
        }
        selected_phase_counts = {
            phase: sum(row["trigger_phase"] == phase for row in selected)
            for phase in PHASE_BINS
        }
        coverage.append({
            **metadata,
            "natural_occurrences": len(all_anchors),
            "distinct_games": distinct_games,
            "distinct_states": len({row["state_id"] for row in all_anchors}),
            "trigger_step_min": min((row["trigger_step"] for row in all_anchors), default=None),
            "trigger_step_max": max((row["trigger_step"] for row in all_anchors), default=None),
            "trigger_step_mean": (
                sum(row["trigger_step"] for row in all_anchors) / len(all_anchors)
                if all_anchors else None
            ),
            "phase_counts": phase_counts,
            "supported": supported,
            "unsupported_reason": None if supported else (
                f"requires >= {args.minimum_occurrences} occurrences and >= {args.minimum_games} games"
            ),
            "selected_occurrences": len(selected),
            "selected_distinct_games": len({row["game_id"] for row in selected}),
            "selected_phase_counts": selected_phase_counts,
        })

    payload = {
        "schema_version": "phase1.all_first_invocation_coverage.v1",
        "checkpoint_id": args.checkpoint_id,
        "condition": args.condition,
        "context_id": args.context_id,
        "source_trajectory_count": source_trajectories,
        "minimum_occurrences": args.minimum_occurrences,
        "minimum_distinct_games": args.minimum_games,
        "maximum_selected_per_skill": args.maximum_selected,
        "eligible_skill_count": len(eligible),
        "supported_skill_count": sum(row["supported"] for row in coverage),
        "unsupported_skill_count": sum(not row["supported"] for row in coverage),
        "coverage": coverage,
    }
    atomic_write_json(args.output_dir / "coverage.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
