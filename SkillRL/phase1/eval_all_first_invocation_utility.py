#!/usr/bin/env python3
"""Evaluate all supported Skills with one model load per checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_system.memory import FrozenStepSkillRouter, SkillsOnlyMemory
from phase1.archive import append_jsonl_idempotent, atomic_write_json, repo_commit, sha256_file, stable_hash
from phase1.eval_first_invocation_utility import (
    ACTION_PROJECTION_VERSION,
    PROMPT_TEMPLATE_VERSION,
    TransformersPolicy,
    run_branch,
)
from phase1.first_invocation import PayloadArm

PROTOCOL_VERSION = "phase1.all-first-invocation-payload.v2"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--anchors-dir", type=Path, required=True)
    parser.add_argument("--placebo-dir", type=Path, required=True)
    parser.add_argument("--skill-bank", type=Path, required=True)
    parser.add_argument("--skills", nargs="*")
    parser.add_argument("--max-anchors-per-skill", type=int)
    parser.add_argument("--arms", nargs="+", choices=[arm.value for arm in PayloadArm], default=[arm.value for arm in PayloadArm])
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--router-general-top-k", type=int, default=12)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rl-seed", type=int, required=True)
    parser.add_argument("--global-update", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    anchor_paths = sorted(args.anchors_dir.glob("*.jsonl"))
    if args.skills:
        requested = set(args.skills)
        anchor_paths = [path for path in anchor_paths if path.stem in requested]
        missing = requested - {path.stem for path in anchor_paths}
        if missing:
            raise ValueError(f"missing selected anchor files: {sorted(missing)}")
    if not anchor_paths:
        raise ValueError("no supported Skill anchor files found")

    skill_inputs: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
    for anchor_path in anchor_paths:
        skill_id = anchor_path.stem
        placebo_path = args.placebo_dir / f"{skill_id}.json"
        placebo = json.loads(placebo_path.read_text(encoding="utf-8"))
        if placebo["actual_token_count"] != placebo["target_token_count"]:
            raise ValueError(f"placebo token mismatch for {skill_id}")
        anchors = load_jsonl(anchor_path)
        if args.max_anchors_per_skill is not None:
            if args.max_anchors_per_skill <= 0:
                raise ValueError("--max-anchors-per-skill must be positive")
            anchors = anchors[: args.max_anchors_per_skill]
        if any(anchor["skill_id"] != skill_id for anchor in anchors):
            raise ValueError(f"anchor Skill mismatch in {anchor_path}")
        skill_inputs.append((skill_id, anchors, placebo))

    policy = TransformersPolicy(args.checkpoint)
    memory = SkillsOnlyMemory(str(args.skill_bank), retrieval_mode="template", task_specific_top_k=None)
    router = FrozenStepSkillRouter(include_common_mistakes=False)
    skill_bank_hash = sha256_file(args.skill_bank)
    repository_commit = repo_commit(Path(__file__).parents[1])
    config_hash = stable_hash(vars(args))
    completed: set[tuple[str, str, str, str]] = set()
    if args.output.exists():
        for row in load_jsonl(args.output):
            completed.add((row["checkpoint_id"], row["skill_id"], row["anchor_id"], row["payload_arm"]))

    total_branches = sum(len(anchors) for _, anchors, _ in skill_inputs) * len(args.arms)
    branch_index = 0
    for skill_id, anchors, placebo in skill_inputs:
        for anchor in anchors:
            for arm_value in args.arms:
                branch_index += 1
                arm = PayloadArm(arm_value)
                key = (args.checkpoint_id, skill_id, anchor["anchor_id"], arm.value)
                if key in completed:
                    continue
                result = run_branch(
                    policy=policy,
                    anchor=anchor,
                    memory=memory,
                    router=router,
                    arm=arm,
                    target_skill_id=skill_id,
                    placebo_text=placebo["text"],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    history_length=args.history_length,
                    router_general_top_k=args.router_general_top_k,
                )
                trajectory_id = stable_hash({
                    "run_id": args.run_id,
                    "checkpoint_id": args.checkpoint_id,
                    "skill_id": skill_id,
                    "anchor_id": anchor["anchor_id"],
                    "payload_arm": arm.value,
                })[:24]
                trajectory_path = args.output.parent / "trajectories" / args.checkpoint_id / skill_id / f"{trajectory_id}.json"
                atomic_write_json(trajectory_path, {
                    "schema_version": "phase1.all_first_invocation_trajectory.v1",
                    "protocol_version": PROTOCOL_VERSION,
                    "run_id": args.run_id,
                    "trajectory_id": trajectory_id,
                    "checkpoint_id": args.checkpoint_id,
                    "model_path": args.checkpoint,
                    "rl_seed": args.rl_seed,
                    "global_update": args.global_update,
                    "skill_id": skill_id,
                    "payload_arm": arm.value,
                    "placebo_hash": placebo["text_hash"],
                    "anchor": anchor,
                    **result,
                })
                row = {
                    "schema_version": "phase1.all_first_invocation_index.v1",
                    "protocol_version": PROTOCOL_VERSION,
                    "run_id": args.run_id,
                    "repo_commit": repository_commit,
                    "config_hash": config_hash,
                    "skill_bank_hash": skill_bank_hash,
                    "checkpoint_id": args.checkpoint_id,
                    "model_path": args.checkpoint,
                    "rl_seed": args.rl_seed,
                    "global_update": args.global_update,
                    "anchor_id": anchor["anchor_id"],
                    "state_id": anchor["state_id"],
                    "source_trajectory_id": anchor["source_trajectory_id"],
                    "source_eval_seed": anchor["source_eval_seed"],
                    "environment_seed": anchor["environment_seed"],
                    "game_id": anchor["game_id"],
                    "context_id": anchor["context_id"],
                    "skill_id": skill_id,
                    "trigger_step": anchor["trigger_step"],
                    "trigger_phase": anchor.get("trigger_phase"),
                    "payload_arm": arm.value,
                    "placebo_hash": placebo["text_hash"],
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "action_projection_version": ACTION_PROJECTION_VERSION,
                    "trajectory_path": str(trajectory_path.resolve()),
                    **{name: value for name, value in result.items() if name != "steps"},
                }
                append_jsonl_idempotent(
                    args.output,
                    [row],
                    unique_fields=("checkpoint_id", "skill_id", "anchor_id", "payload_arm"),
                )
                completed.add(key)
                print(
                    f"[{args.checkpoint_id}] {branch_index}/{total_branches} "
                    f"skill={skill_id} anchor={anchor['anchor_id']} arm={arm.value} "
                    f"return={result['suffix_return']} success={result['success']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
