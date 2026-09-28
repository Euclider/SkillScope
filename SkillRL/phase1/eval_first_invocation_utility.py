#!/usr/bin/env python3
"""Resumable first-invocation Skill payload counterfactual evaluator."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from agent_system.environments.prompts.alfworld import (
    ALFWORLD_TEMPLATE_NO_HIS_WITH_MEMORY,
    ALFWORLD_TEMPLATE_WITH_MEMORY,
    use_action_only_instruction,
)
from agent_system.memory import FrozenStepSkillRouter, SkillsOnlyMemory
from phase1.archive import (
    append_jsonl_idempotent,
    atomic_write_json,
    repo_commit,
    sha256_file,
    stable_hash,
)
from phase1.eval_skill_margin import (
    ACTION_PROJECTION_VERSION,
    SingleGameEnvironment,
    TransformersPolicy,
    extract_task,
    format_actions,
    format_history,
    resolve_game_file,
)
from phase1.first_invocation import PayloadArm, intervention_payload

PROTOCOL_VERSION = "phase1.first-invocation-payload.v1"
PROMPT_TEMPLATE_VERSION = "skillrl-alfworld-first-invocation-payload-v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def replay_prefix(
    environment: SingleGameEnvironment,
    anchor: dict[str, Any],
) -> tuple[str, dict[str, Any], str, list[dict[str, str]], float]:
    observation, info = environment.reset()
    task = extract_task(observation)
    history: list[dict[str, str]] = []
    prefix_return = 0.0
    prefix_history = anchor.get("prefix_history", [])
    for index, action in enumerate(anchor.get("prefix_actions", [])):
        expected = prefix_history[index]
        if observation != expected["observation"]:
            raise ValueError(
                f"Prefix observation mismatch for {anchor['anchor_id']} at {index}"
            )
        next_observation, next_info, done = environment.step(action)
        prefix_return += 10.0 * float(next_info.get("won", False))
        history.append({"observation": observation, "action": action})
        observation, info = next_observation, next_info
        if done:
            raise ValueError(
                f"Prefix terminated before trigger for {anchor['anchor_id']}"
            )
    if observation != anchor["trigger_observation"]:
        raise ValueError(f"Trigger observation mismatch for {anchor['anchor_id']}")
    if list(info["admissible_commands"]) != list(
        anchor["trigger_admissible_actions"]
    ):
        raise ValueError(f"Trigger action-set mismatch for {anchor['anchor_id']}")
    if anchor.get("task_description") and task != anchor["task_description"]:
        raise ValueError(f"Task mismatch for {anchor['anchor_id']}")
    return observation, info, task, history, prefix_return


def run_branch(
    *,
    policy: TransformersPolicy,
    anchor: dict[str, Any],
    memory: SkillsOnlyMemory,
    router: FrozenStepSkillRouter,
    arm: PayloadArm,
    target_skill_id: str,
    placebo_text: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    history_length: int,
    router_general_top_k: int,
) -> dict[str, Any]:
    max_steps = int(anchor["max_steps"])
    trigger_step = int(anchor["trigger_step"])
    environment = SingleGameEnvironment(
        resolve_game_file(anchor["game_id"]),
        int(anchor["environment_seed"]),
        max_steps,
    )
    steps: list[dict[str, Any]] = []
    suffix_return = 0.0
    invalid_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    try:
        observation, info, task, history, prefix_return = replay_prefix(
            environment, anchor
        )
        candidate_retrieval = memory.retrieve(task, top_k=router_general_top_k)
        for step_index in range(trigger_step, max_steps):
            actions = list(info["admissible_commands"])
            routed = router.route(
                candidate_retrieval,
                task_description=task,
                current_observation=observation,
                admissible_actions=actions,
                history=history,
                step_index=step_index,
            )
            if step_index == trigger_step and (
                routed.get("selected_skill_id") != target_skill_id
            ):
                raise ValueError(
                    f"Frozen Router did not reproduce {target_skill_id} at anchor "
                    f"{anchor['anchor_id']}: {routed.get('selected_skill_id')}"
                )
            payload_text, payload_injected_ids = intervention_payload(
                memory=memory,
                routed=routed,
                arm=arm,
                target_skill_id=target_skill_id,
                placebo_text=placebo_text,
            )
            if step_index == 0:
                prompt = ALFWORLD_TEMPLATE_NO_HIS_WITH_MEMORY.format(
                    task_description=task,
                    current_observation=observation,
                    retrieved_memories=payload_text,
                    admissible_actions=format_actions(actions),
                )
            else:
                prompt = ALFWORLD_TEMPLATE_WITH_MEMORY.format(
                    task_description=task,
                    retrieved_memories=payload_text,
                    step_count=len(history),
                    history_length=min(history_length, len(history)),
                    action_history=format_history(history, history_length),
                    current_step=len(history) + 1,
                    current_observation=observation,
                    admissible_actions=format_actions(actions),
                )
            prompt = use_action_only_instruction(prompt)
            eval_seed = int(anchor["source_eval_seed"])
            raw, n_prompt, n_completion = policy.generate(
                prompt,
                eval_seed + step_index,
                temperature,
                top_p,
                max_new_tokens,
            )
            projected, valid = alfworld_projection([raw], [actions])
            action = projected[0]
            invalid_count += int(not valid[0])
            next_observation, next_info, done = environment.step(action)
            reward = 10.0 * float(next_info.get("won", False))
            suffix_return += reward
            prompt_tokens += n_prompt
            completion_tokens += n_completion
            steps.append({
                "step_index": step_index,
                "payload_arm": arm.value,
                "prompt_text": prompt,
                "payload_text": payload_text,
                "payload_text_hash": stable_hash(payload_text),
                "observation": observation,
                "admissible_actions": actions,
                "raw_model_output": raw,
                "projected_action": action,
                "is_action_valid": bool(valid[0]),
                "reward": reward,
                "next_observation": next_observation,
                "retrieved_skill_ids": routed.get("retrieved_skill_ids", []),
                "candidate_skill_ids": routed.get("candidate_skill_ids", []),
                "selected_skill_id": routed.get("selected_skill_id"),
                "router_injected_skill_ids": routed.get("injected_skill_ids", []),
                "payload_injected_skill_ids": payload_injected_ids,
                "skill_router_version": routed.get("skill_router_version"),
                "skill_router_scores": routed.get("skill_router_scores", {}),
                "skill_router_score_details": routed.get(
                    "skill_router_score_details", {}
                ),
                "skill_router_state_flags": routed.get(
                    "skill_router_state_flags", []
                ),
                "prompt_tokens": n_prompt,
                "completion_tokens": n_completion,
            })
            if "skill_router_api" in routed:
                steps[-1]["skill_router_api"] = dict(routed["skill_router_api"])
            history.append({"observation": observation, "action": action})
            observation, info = next_observation, next_info
            if done:
                break
        selected_ids = [
            step["selected_skill_id"] for step in steps
            if step.get("selected_skill_id")
        ]
        target_selected_count = sum(
            selected == target_skill_id for selected in selected_ids
        )
        target_payload_count = sum(
            target_skill_id in step["payload_injected_skill_ids"] for step in steps
        )
        return {
            "task_description": task,
            "prefix_replay_verified": True,
            "prefix_return": prefix_return,
            "suffix_return": suffix_return,
            "episode_return": prefix_return + suffix_return,
            "success": bool(info.get("won", False)),
            "invalid_action_count": invalid_count,
            "suffix_trajectory_length": len(steps),
            "total_trajectory_length": trigger_step + len(steps),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "first_action": steps[0]["projected_action"] if steps else None,
            "action_sequence": [step["projected_action"] for step in steps],
            "selected_skill_ids": sorted(set(selected_ids)),
            "skill_selection_counts": dict(sorted(Counter(selected_ids).items())),
            "target_selected_count": target_selected_count,
            "target_payload_injection_count": target_payload_count,
            "steps": steps,
        }
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--max-anchors", type=int)
    parser.add_argument("--skill-bank", type=Path, required=True)
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--placebo", type=Path, required=True)
    parser.add_argument(
        "--arms", nargs="+", choices=[arm.value for arm in PayloadArm],
        default=[arm.value for arm in PayloadArm],
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--router-general-top-k", type=int, default=12)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rl-seed", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).parents[1]
    anchors = load_jsonl(args.anchors)
    if args.max_anchors is not None:
        if args.max_anchors <= 0:
            raise ValueError("--max-anchors must be positive")
        anchors = anchors[:args.max_anchors]
    placebo = json.loads(args.placebo.read_text(encoding="utf-8"))
    if placebo["actual_token_count"] != placebo["target_token_count"]:
        raise ValueError("Placebo payload is not token-length matched")
    policy = TransformersPolicy(args.checkpoint)
    memory = SkillsOnlyMemory(
        str(args.skill_bank), retrieval_mode="template", task_specific_top_k=None
    )
    router = FrozenStepSkillRouter(include_common_mistakes=False)
    skill_bank_hash = sha256_file(args.skill_bank)
    config_hash = stable_hash(vars(args))
    completed: set[tuple[str, str, str]] = set()
    if args.output.exists():
        for row in load_jsonl(args.output):
            completed.add((
                row["checkpoint_id"], row["anchor_id"], row["payload_arm"]
            ))
    for anchor_index, anchor in enumerate(anchors, start=1):
        for arm_value in args.arms:
            arm = PayloadArm(arm_value)
            key = (args.checkpoint_id, anchor["anchor_id"], arm.value)
            if key in completed:
                continue
            result = run_branch(
                policy=policy,
                anchor=anchor,
                memory=memory,
                router=router,
                arm=arm,
                target_skill_id=args.skill_id,
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
                "anchor_id": anchor["anchor_id"],
                "payload_arm": arm.value,
            })[:24]
            trajectory_path = (
                args.output.parent / "trajectories" / args.checkpoint_id
                / f"{trajectory_id}.json"
            )
            atomic_write_json(trajectory_path, {
                "schema_version": "phase1.first_invocation_trajectory.v1",
                "protocol_version": PROTOCOL_VERSION,
                "run_id": args.run_id,
                "trajectory_id": trajectory_id,
                "checkpoint_id": args.checkpoint_id,
                "model_path": args.checkpoint,
                "payload_arm": arm.value,
                "placebo_hash": placebo["text_hash"],
                "anchor": anchor,
                **result,
            })
            row = {
                "schema_version": "phase1.first_invocation_index.v1",
                "protocol_version": PROTOCOL_VERSION,
                "run_id": args.run_id,
                "repo_commit": repo_commit(repo_root),
                "config_hash": config_hash,
                "skill_bank_hash": skill_bank_hash,
                "checkpoint_id": args.checkpoint_id,
                "model_path": args.checkpoint,
                "rl_seed": args.rl_seed,
                "anchor_id": anchor["anchor_id"],
                "state_id": anchor["state_id"],
                "source_trajectory_id": anchor["source_trajectory_id"],
                "source_eval_seed": anchor["source_eval_seed"],
                "environment_seed": anchor["environment_seed"],
                "game_id": anchor["game_id"],
                "context_id": anchor["context_id"],
                "skill_id": args.skill_id,
                "trigger_step": anchor["trigger_step"],
                "payload_arm": arm.value,
                "placebo_hash": placebo["text_hash"],
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "action_projection_version": ACTION_PROJECTION_VERSION,
                "trajectory_path": str(trajectory_path),
                **{key: value for key, value in result.items() if key != "steps"},
            }
            append_jsonl_idempotent(
                args.output,
                [row],
                unique_fields=("checkpoint_id", "anchor_id", "payload_arm"),
            )
            completed.add(key)
            print(
                f"[{args.checkpoint_id}] anchor {anchor_index}/{len(anchors)} "
                f"arm={arm.value} return={result['suffix_return']} "
                f"success={result['success']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
