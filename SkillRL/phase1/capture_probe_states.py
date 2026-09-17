#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase1.archive import append_jsonl_idempotent, stable_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--max-states-per-game", type=int, default=4)
    parser.add_argument("--max-total-states", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--split")
    parser.add_argument("--context-id")
    parser.add_argument("--global-step", type=int)
    args = parser.parse_args()

    if args.max_states_per_game <= 0:
        raise ValueError("--max-states-per-game must be positive")
    if args.max_total_states is not None and args.max_total_states <= 0:
        raise ValueError("--max-total-states must be positive")

    counts: dict[str, int] = {}
    rows = []
    for line in args.trajectory_index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        index = json.loads(line)
        if args.run_id is not None and index.get("run_id") != args.run_id:
            continue
        if args.split is not None and index.get("split") != args.split:
            continue
        if args.context_id is not None and index.get("context_id") != args.context_id:
            continue
        if args.global_step is not None and index.get("global_step") != args.global_step:
            continue
        trajectory = json.loads(Path(index["trajectory_path"]).read_text(encoding="utf-8"))
        game_id = str(trajectory.get("game_id"))
        for position, step in enumerate(trajectory["steps"]):
            if step.get("selected_skill_id") != args.skill_id and args.skill_id not in step.get("injected_skill_ids", []):
                continue
            if counts.get(game_id, 0) >= args.max_states_per_game:
                break
            prompt = step.get("prompt_text") or ""
            if not any(
                header in prompt
                for header in (
                    "## Retrieved Relevant Experience",
                    "## Selected Skill For This Step",
                )
            ):
                continue
            probe_id = stable_hash({
                "game_id": game_id,
                "trajectory_id": trajectory["trajectory_id"],
                "step_index": step["step_index"],
            })[:24]
            rows.append({
                "probe_id": probe_id,
                "source_trajectory_id": trajectory["trajectory_id"],
                "game_id": game_id,
                "context_id": trajectory.get("context_id"),
                "step_index": step["step_index"],
                "task_description": next(
                    (item.get("task_description") for item in trajectory["steps"] if item.get("task_description")),
                    None,
                ) or "",
                "prompt_text": prompt,
                "observation": step.get("observation"),
                "admissible_actions": step.get("admissible_actions", []),
                "source_projected_action": step.get("projected_action"),
                "candidate_skill_ids": step.get("candidate_skill_ids", []),
                "selected_skill_id": step.get("selected_skill_id"),
                "skill_router_version": step.get("skill_router_version"),
                "skill_router_state_flags": step.get("skill_router_state_flags", []),
                "history": [
                    {
                        "observation": prior.get("observation", ""),
                        "action": prior.get("projected_action", ""),
                    }
                    for prior in trajectory["steps"][:position]
                ],
            })
            counts[game_id] = counts.get(game_id, 0) + 1
            if args.max_total_states is not None and len(rows) >= args.max_total_states:
                append_jsonl_idempotent(args.output, rows, unique_fields=("probe_id",))
                return
    append_jsonl_idempotent(args.output, rows, unique_fields=("probe_id",))


if __name__ == "__main__":
    main()
