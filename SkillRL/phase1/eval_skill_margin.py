#!/usr/bin/env python3
"""Standalone, resumable ALFWorld FULL/MINUS/NO-SKILL evaluator."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from agent_system.environments.prompts.alfworld import (
    ALFWORLD_TEMPLATE_NO_HIS,
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
from phase1.conditions import SkillCondition, apply_skill_condition

PROMPT_TEMPLATE_VERSION = "skillrl-8e66726-alfworld-v1"
STEP_ROUTER_PROMPT_TEMPLATE_VERSION = "skillrl-alfworld-step-skill-action-only-v3"
ACTION_PROJECTION_VERSION = "skillrl-8e66726-alfworld-projection-v1"


def load_game_ids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        value = json.loads(text)
        if isinstance(value, dict):
            value = value.get("game_ids", [])
        return [str(item) for item in value]
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def unbatch_info(info: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in info.items():
        if isinstance(value, (list, tuple)) and len(value) == 1:
            result[key] = value[0]
        else:
            result[key] = value
    return result


def extract_task(observation: str) -> str:
    marker = "Your task is to: "
    if marker not in observation:
        raise ValueError("Task description not found in initial ALFWorld observation")
    return observation.split(marker, 1)[1].strip()


def format_actions(actions: list[str]) -> str:
    return "\n ".join(f"'{action}'" for action in actions if action != "help")


def format_history(history: list[dict[str, str]], history_length: int) -> str:
    recent = history[-history_length:]
    start = len(history) - len(recent)
    return "\n".join(
        f"[Observation {start + index + 1}: '{item['observation']}', Action {start + index + 1}: '{item['action']}']"
        for index, item in enumerate(recent)
    )


class TransformersPolicy:
    def __init__(self, checkpoint: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        ).to(self.device).eval()

    def generate(self, prompt: str, seed: int, temperature: float, top_p: float, max_new_tokens: int) -> tuple[str, int, int]:
        torch = self.torch
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.device)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            kwargs.update({"temperature": temperature, "top_p": top_p})
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        response = output[0, inputs.input_ids.shape[1]:]
        return (
            self.tokenizer.decode(response, skip_special_tokens=True),
            int(inputs.input_ids.numel()),
            int(response.numel()),
        )


class SingleGameEnvironment:
    def __init__(self, game_file: Path, seed: int, max_steps: int):
        config_path = Path(__file__).parents[1] / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = copy.deepcopy(config)
        for key in ("data_path", "eval_id_data_path", "eval_ood_data_path"):
            config["dataset"][key] = str(game_file.parent)
        config["rl"]["training"]["max_nb_steps_per_episode"] = max_steps
        base = get_environment(config["env"]["type"])(config, train_eval="eval_in_distribution")
        if str(game_file) not in base.game_files:
            raise ValueError(f"Game is absent, unsupported, or unsolvable: {game_file}")
        base.game_files = [str(game_file)]
        base.num_games = 1
        self.env = base.init_env(batch_size=1)
        self.env.seed(seed)

    def reset(self) -> tuple[str, dict]:
        observations, infos = self.env.reset()
        return observations[0], unbatch_info(infos)

    def step(self, action: str) -> tuple[str, dict, bool]:
        observations, _, dones, infos = self.env.step([action])
        return observations[0], unbatch_info(infos), bool(dones[0])

    def close(self) -> None:
        self.env.close()


def resolve_game_file(game_id: str) -> Path:
    path = Path(os.path.expandvars(game_id)).expanduser()
    if not path.is_absolute():
        data_root = os.environ.get("ALFWORLD_DATA")
        if not data_root:
            raise OSError("ALFWORLD_DATA must be set for relative game IDs")
        path = Path(data_root) / path
    if path.is_dir():
        path = path / "game.tw-pddl"
    return path.resolve()


def run_episode(
    *,
    policy: TransformersPolicy,
    game_file: Path,
    memory: SkillsOnlyMemory,
    condition: SkillCondition,
    skill_id: str,
    environment_seed: int,
    eval_seed: int,
    temperature: float,
    top_p: float,
    max_steps: int,
    max_new_tokens: int,
    history_length: int,
    step_skill_router: FrozenStepSkillRouter | None = None,
    router_general_top_k: int = 12,
) -> dict:
    environment = SingleGameEnvironment(game_file, environment_seed, max_steps)
    history: list[dict[str, str]] = []
    steps = []
    invalid_count = 0
    inadmissible_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    episode_return = 0.0
    try:
        observation, info = environment.reset()
        task = extract_task(observation)
        with apply_skill_condition(memory, condition, skill_id):
            candidate_retrieval = memory.retrieve(
                task,
                top_k=router_general_top_k if step_skill_router else 1,
            )
        for step_index in range(max_steps):
            actions = list(info["admissible_commands"])
            retrieval = candidate_retrieval
            if step_skill_router is not None:
                retrieval = step_skill_router.route(
                    candidate_retrieval,
                    task_description=task,
                    current_observation=observation,
                    admissible_actions=actions,
                    history=history,
                    step_index=step_index,
                )
            if step_index == 0 and step_skill_router is not None:
                prompt = ALFWORLD_TEMPLATE_NO_HIS_WITH_MEMORY.format(
                    task_description=task,
                    current_observation=observation,
                    retrieved_memories=memory.format_for_prompt(retrieval),
                    admissible_actions=format_actions(actions),
                )
                injected_ids = retrieval["injected_skill_ids"]
            elif step_index == 0:
                prompt = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=observation,
                    admissible_actions=format_actions(actions),
                )
                injected_ids = []
            else:
                prompt = ALFWORLD_TEMPLATE_WITH_MEMORY.format(
                    task_description=task,
                    retrieved_memories=memory.format_for_prompt(retrieval),
                    step_count=len(history),
                    history_length=min(history_length, len(history)),
                    action_history=format_history(history, history_length),
                    current_step=len(history) + 1,
                    current_observation=observation,
                    admissible_actions=format_actions(actions),
                )
                injected_ids = retrieval["injected_skill_ids"]
            prompt = use_action_only_instruction(prompt)
            raw, n_prompt, n_completion = policy.generate(
                prompt, eval_seed + step_index, temperature, top_p, max_new_tokens
            )
            projected, valid = alfworld_projection([raw], [actions])
            action = projected[0]
            invalid_count += int(not valid[0])
            is_admissible = action in actions
            inadmissible_count += int(not is_admissible)
            next_observation, next_info, done = environment.step(action)
            reward = 10.0 * float(next_info.get("won", False))
            episode_return += reward
            prompt_tokens += n_prompt
            completion_tokens += n_completion
            steps.append({
                "step_index": step_index,
                "prompt_text": prompt,
                "observation": observation,
                "admissible_actions": actions,
                "raw_model_output": raw,
                "projected_action": action,
                "is_action_valid": bool(valid[0]),
                "is_action_admissible": is_admissible,
                "reward": reward,
                "next_observation": next_observation,
                "retrieved_skill_ids": retrieval["retrieved_skill_ids"],
                "candidate_skill_ids": retrieval.get("candidate_skill_ids", []),
                "injected_skill_ids": injected_ids,
                "disabled_skill_ids": retrieval["disabled_skill_ids"],
                "selected_skill_id": retrieval.get("selected_skill_id"),
                "skill_router_version": retrieval.get("skill_router_version"),
                "skill_router_scores": retrieval.get("skill_router_scores", {}),
                "skill_router_score_details": retrieval.get(
                    "skill_router_score_details", {}
                ),
                "skill_router_state_flags": retrieval.get(
                    "skill_router_state_flags", []
                ),
                "skill_router_selection_reason": retrieval.get(
                    "skill_router_selection_reason"
                ),
                "prompt_tokens": n_prompt,
                "completion_tokens": n_completion,
            })
            if "skill_router_api" in retrieval:
                steps[-1]["skill_router_api"] = dict(retrieval["skill_router_api"])
            history.append({"observation": observation, "action": action})
            observation, info = next_observation, next_info
            if done:
                break
        selected_ids = [
            step["selected_skill_id"] for step in steps
            if step.get("selected_skill_id")
        ]
        retrieved_ids = sorted({
            item for step in steps for item in step["retrieved_skill_ids"]
        })
        candidate_ids = sorted({
            item for step in steps for item in step.get("candidate_skill_ids", [])
        })
        return {
            "task_description": task,
            "context_id": candidate_retrieval["task_type"],
            "retrieved_skill_ids": retrieved_ids,
            "candidate_skill_ids": candidate_ids,
            "injected_skill_ids": sorted({item for step in steps for item in step["injected_skill_ids"]}),
            "unique_skill_count": len({item for step in steps for item in step["injected_skill_ids"]}),
            "selected_skill_ids": sorted(set(selected_ids)),
            "skill_selection_counts": dict(sorted(Counter(selected_ids).items())),
            "skill_router_version": (
                step_skill_router.version if step_skill_router is not None else None
            ),
            "success": bool(info.get("won", False)),
            "episode_return": episode_return,
            "invalid_action_count": invalid_count,
            "inadmissible_action_count": inadmissible_count,
            "trajectory_length": len(steps),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "action_sequence": [step["projected_action"] for step in steps],
            "steps": steps,
        }
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--skill-bank", type=Path, required=True)
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--context-id", required=True)
    parser.add_argument("--game-ids-file", type=Path, required=True)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--game-shard-index", type=int, default=0)
    parser.add_argument("--game-shard-count", type=int, default=1)
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--conditions", nargs="+", choices=[item.value for item in SkillCondition], default=[item.value for item in SkillCondition])
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--step-routing", action="store_true")
    parser.add_argument("--router-general-top-k", type=int, default=12)
    parser.add_argument("--router-include-common-mistakes", action="store_true")
    parser.add_argument("--environment-seed", type=int, default=1000)
    parser.add_argument("--rl-seed", type=int, default=0)
    parser.add_argument("--update-id", default="0")
    parser.add_argument("--update-type", choices=("real_rl", "zero_update", "shuffled_reward", "random_parameter"), default="real_rl")
    parser.add_argument("--split", choices=("development", "valid_seen", "valid_unseen"), default="valid_seen")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).parents[1]
    policy = TransformersPolicy(args.checkpoint)
    memory = SkillsOnlyMemory(
        str(args.skill_bank),
        retrieval_mode="template",
        task_specific_top_k=None if args.step_routing else 1,
    )
    step_skill_router = (
        FrozenStepSkillRouter(args.router_include_common_mistakes)
        if args.step_routing else None
    )
    checkpoint_id = Path(args.checkpoint).name
    skill_bank_hash = sha256_file(args.skill_bank)
    config_hash = stable_hash(vars(args))
    unique_fields = (
        "checkpoint_id", "update_type", "rl_seed", "eval_seed", "game_id",
        "skill_id", "skill_condition",
    )
    completed = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing = json.loads(line)
                completed.add(tuple(existing.get(field) for field in unique_fields))
    rows = []
    game_ids = load_game_ids(args.game_ids_file)
    if args.game_shard_count <= 0:
        raise ValueError("--game-shard-count must be positive")
    if not 0 <= args.game_shard_index < args.game_shard_count:
        raise ValueError("--game-shard-index must be in [0, game-shard-count)")
    game_ids = game_ids[args.game_shard_index::args.game_shard_count]
    if args.max_games is not None:
        if args.max_games <= 0:
            raise ValueError("--max-games must be positive")
        game_ids = game_ids[:args.max_games]
    for game_index, game_id in enumerate(game_ids):
        game_file = resolve_game_file(game_id)
        for eval_seed in args.eval_seeds:
            environment_seed = args.environment_seed + game_index
            for condition_value in args.conditions:
                condition = SkillCondition(condition_value)
                unique_key = (
                    checkpoint_id, args.update_type, args.rl_seed, eval_seed,
                    str(game_file), args.skill_id, condition.value,
                )
                if unique_key in completed:
                    continue
                result = run_episode(
                    policy=policy,
                    game_file=game_file,
                    memory=memory,
                    condition=condition,
                    skill_id=args.skill_id,
                    environment_seed=environment_seed,
                    eval_seed=eval_seed,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_steps=args.max_steps,
                    max_new_tokens=args.max_new_tokens,
                    history_length=args.history_length,
                    step_skill_router=step_skill_router,
                    router_general_top_k=args.router_general_top_k,
                )
                if result["context_id"] != args.context_id:
                    raise ValueError(f"Expected context {args.context_id}, got {result['context_id']} for {game_file}")
                trajectory_id = stable_hash({
                    "run_id": args.run_id, "checkpoint_id": checkpoint_id,
                    "game_id": str(game_file), "eval_seed": eval_seed,
                    "skill_id": args.skill_id, "condition": condition.value,
                })[:24]
                trajectory_path = args.output.parent / "trajectories" / args.run_id / f"{trajectory_id}.json"
                atomic_write_json(trajectory_path, {
                    "schema_version": (
                        "phase1.trajectory.v2" if args.step_routing
                        else "phase1.trajectory.v1"
                    ),
                    "run_id": args.run_id,
                    "trajectory_id": trajectory_id,
                    "checkpoint_id": checkpoint_id,
                    "game_id": str(game_file),
                    "eval_seed": eval_seed,
                    "environment_seed": environment_seed,
                    "skill_id": args.skill_id,
                    "skill_condition": condition.value,
                    **result,
                })
                rows.append({
                    "run_id": args.run_id,
                    "repo_commit": repo_commit(repo_root),
                    "config_hash": config_hash,
                    "skill_bank_hash": skill_bank_hash,
                    "model_id": args.checkpoint,
                    "checkpoint_id": checkpoint_id,
                    "update_id": args.update_id,
                    "update_type": args.update_type,
                    "rl_seed": args.rl_seed,
                    "eval_seed": eval_seed,
                    "environment_seed": environment_seed,
                    "game_id": str(game_file),
                    "split": args.split,
                    "context_id": result["context_id"],
                    "skill_id": args.skill_id,
                    "skill_condition": condition.value,
                    "retrieved_skill_ids": result["retrieved_skill_ids"],
                    "candidate_skill_ids": result["candidate_skill_ids"],
                    "injected_skill_ids": result["injected_skill_ids"],
                    "unique_skill_count": result["unique_skill_count"],
                    "selected_skill_ids": result["selected_skill_ids"],
                    "skill_selection_counts": result["skill_selection_counts"],
                    "skill_router_version": result["skill_router_version"],
                    "disabled_skill_ids": result["steps"][0]["disabled_skill_ids"],
                    "success": result["success"],
                    "episode_return": result["episode_return"],
                    "invalid_action_count": result["invalid_action_count"],
                    "inadmissible_action_count": result["inadmissible_action_count"],
                    "trajectory_length": result["trajectory_length"],
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "action_sequence": result["action_sequence"],
                    "trajectory_path": str(trajectory_path),
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_steps": args.max_steps,
                    "action_projection_version": ACTION_PROJECTION_VERSION,
                    "prompt_template_version": (
                        STEP_ROUTER_PROMPT_TEMPLATE_VERSION
                        if args.step_routing else PROMPT_TEMPLATE_VERSION
                    ),
                })
                append_jsonl_idempotent(
                    args.output,
                    [rows[-1]],
                    unique_fields=unique_fields,
                )
                completed.add(unique_key)


if __name__ == "__main__":
    main()
