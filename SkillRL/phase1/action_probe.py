#!/usr/bin/env python3
"""Fixed-state policy/skill action probe run before expensive matched rollouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from agent_system.memory import FrozenStepSkillRouter, SkillsOnlyMemory
from phase1.archive import append_jsonl_idempotent, stable_hash
from phase1.conditions import SkillCondition, apply_skill_condition

MEMORY_HEADERS = (
    "## Retrieved Relevant Experience\n\n",
    "## Selected Skill For This Step\n\n",
)
MEMORY_ENDS = ("\n\n## Current Progress", "\n\nYour admissible actions")


def replace_memory_section(prompt: str, memory_text: str) -> str:
    header = next((item for item in MEMORY_HEADERS if item in prompt), None)
    if header is None:
        raise ValueError("Probe prompt has no replaceable retrieved-memory section")
    start = prompt.find(header)
    content_start = start + len(header)
    end = min(
        (position for marker in MEMORY_ENDS
         if (position := prompt.find(marker, content_start)) >= 0),
        default=-1,
    )
    if end <= content_start:
        raise ValueError("Probe prompt has no replaceable retrieved-memory section")
    return prompt[:content_start] + memory_text + prompt[end:]


def conditioned_probe_prompt(
    state: dict,
    memory: SkillsOnlyMemory,
    condition: SkillCondition,
    skill_id: str | None,
    router: FrozenStepSkillRouter | None = None,
) -> tuple[str, dict]:
    with apply_skill_condition(memory, condition, skill_id):
        retrieved = memory.retrieve(
            state["task_description"], top_k=12 if router else 1
        )
        if router is not None:
            retrieved = router.route(
                retrieved,
                task_description=state["task_description"],
                current_observation=state["observation"],
                admissible_actions=state["admissible_actions"],
                history=state.get("history", []),
                step_index=int(state["step_index"]),
            )
        memory_text = memory.format_for_prompt(retrieved)
    return replace_memory_section(state["prompt_text"], memory_text), retrieved


def score_action_candidates(model, tokenizer, prompt: str, actions: list[str], device) -> dict[str, float]:
    """Normalized sequence scores under a fixed, reasoning-free action prefix."""
    import torch

    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    prefix_ids = tokenizer(chat_prompt + "<think></think><action>", return_tensors="pt").input_ids.to(device)
    scores = []
    with torch.inference_mode():
        for action in actions:
            target_ids = tokenizer(action + "</action>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            full_ids = torch.cat([prefix_ids, target_ids], dim=1)
            logits = model(full_ids).logits[:, :-1]
            log_probs = logits.log_softmax(dim=-1)
            start = prefix_ids.shape[1] - 1
            token_scores = log_probs[:, start:start + target_ids.shape[1]].gather(
                -1, target_ids.unsqueeze(-1)
            ).squeeze(-1)
            scores.append(float(token_scores.sum().item()))
    maximum = max(scores)
    normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in scores))
    return {action: value - normalizer for action, value in zip(actions, scores)}


def generate_action(model, tokenizer, prompt: str, device, max_new_tokens: int) -> tuple[str, str, bool, int, int]:
    import torch

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_ids = output[0, inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(response_ids, skip_special_tokens=True)
    # The upstream projection only validates/extracts the tagged action.  We
    # separately compare it with the archived admissible-action set below.
    projected, valid = alfworld_projection([raw], [[]])
    return raw, projected[0], bool(valid[0]), int(inputs.input_ids.numel()), int(response_ids.numel())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--skill-bank", type=Path, required=True)
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=[item.value for item in SkillCondition], default=[item.value for item in SkillCondition])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--step-routing", action="store_true")
    parser.add_argument("--router-include-common-mistakes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    states = [json.loads(line) for line in args.states.read_text(encoding="utf-8").splitlines() if line.strip()]
    memory = SkillsOnlyMemory(
        str(args.skill_bank),
        retrieval_mode="template",
        task_specific_top_k=None if args.step_routing else 1,
    )
    router = (
        FrozenStepSkillRouter(args.router_include_common_mistakes)
        if args.step_routing else None
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for checkpoint in args.checkpoints:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        ).to(device).eval()
        rows = []
        for state in states:
            for condition_value in args.conditions:
                condition = SkillCondition(condition_value)
                prompt, retrieval = conditioned_probe_prompt(
                    state, memory, condition, args.skill_id, router
                )
                raw, action, valid, prompt_tokens, completion_tokens = generate_action(
                    model, tokenizer, prompt, device, args.max_new_tokens
                )
                candidate_scores = score_action_candidates(
                    model, tokenizer, prompt, state["admissible_actions"], device
                )
                rows.append({
                    "probe_id": state["probe_id"],
                    "checkpoint": checkpoint,
                    "checkpoint_id": Path(checkpoint).name,
                    "skill_id": args.skill_id,
                    "skill_condition": condition.value,
                    "prompt_hash": stable_hash(prompt),
                    "retrieved_skill_ids": retrieval["retrieved_skill_ids"],
                    "candidate_skill_ids": retrieval.get("candidate_skill_ids", []),
                    "injected_skill_ids": retrieval["injected_skill_ids"],
                    "disabled_skill_ids": retrieval["disabled_skill_ids"],
                    "selected_skill_id": retrieval.get("selected_skill_id"),
                    "skill_router_version": retrieval.get("skill_router_version"),
                    "skill_router_state_flags": retrieval.get(
                        "skill_router_state_flags", []
                    ),
                    "greedy_raw_output": raw,
                    "greedy_projected_action": action,
                    "is_action_valid": valid,
                    "is_action_admissible": action in state["admissible_actions"],
                    "candidate_action_log_probs": candidate_scores,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "probe_protocol_version": (
                        "fixed-state-step-routed-action-v2"
                        if args.step_routing
                        else "fixed-state-constrained-action-v1"
                    ),
                })
        append_jsonl_idempotent(
            args.output,
            rows,
            unique_fields=("probe_id", "checkpoint_id", "skill_id", "skill_condition"),
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
