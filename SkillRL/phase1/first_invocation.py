"""Utilities for first-invocation anchored Skill payload interventions."""

from __future__ import annotations

import copy
from enum import Enum
from typing import Any, Mapping

from phase1.archive import stable_hash


class PayloadArm(str, Enum):
    ORIGINAL = "original"
    PLACEBO = "placebo"
    NULL = "null"


_PLACEBO_STREAM = (
    "Typography concerns the visual arrangement of letters in printed documents. "
    "Editors compare serif shapes, line spacing, page margins, ink density, and paper texture. "
    "A quiet archive catalogs alphabets, punctuation marks, type specimens, and binding patterns. "
    "The notes describe visual rhythm, balanced columns, historical fonts, and decorative initials. "
) * 16


def selected_payload_text(memory, routed: Mapping[str, Any]) -> str:
    """Format the single routed Skill without changing routing metadata."""
    payload = {
        "general_skills": list(routed.get("general_skills", [])),
        "task_specific_skills": list(routed.get("task_specific_skills", [])),
        "mistakes_to_avoid": list(routed.get("mistakes_to_avoid", [])),
        "task_type": routed.get("task_type", "unknown"),
        "retrieval_mode": routed.get("retrieval_mode", "template"),
    }
    if not any(payload[key] for key in (
        "general_skills", "task_specific_skills", "mistakes_to_avoid"
    )):
        return ""
    return memory.format_for_prompt(payload)


def make_length_matched_placebo(tokenizer, original_text: str) -> dict[str, Any]:
    """Create unrelated formatted text with exactly the original token count."""
    target_tokens = len(tokenizer(original_text, add_special_tokens=False).input_ids)
    first_line = original_text.splitlines()[0] if original_text.splitlines() else "### Skills"
    if not first_line.startswith("### "):
        raise ValueError("Original Skill payload does not begin with a section header")
    prefix = f"{first_line}\n- **Neutral Typography Note**: "
    suffix = "\n  _Apply when: reading an unrelated note about printed symbols._"
    stream_ids = tokenizer(
        _PLACEBO_STREAM, add_special_tokens=False
    ).input_ids
    matches: list[tuple[int, str]] = []
    for length in range(len(stream_ids) + 1):
        body = tokenizer.decode(stream_ids[:length], skip_special_tokens=True).strip()
        text = f"{prefix}{body}{suffix}"
        count = len(tokenizer(text, add_special_tokens=False).input_ids)
        if count == target_tokens:
            matches.append((length, text))
            break
        if count > target_tokens + 2:
            break
    if not matches:
        raise ValueError(
            f"Could not construct a {target_tokens}-token placebo payload"
        )
    _, text = matches[0]
    return {
        "schema_version": "phase1.skill_payload_placebo.v1",
        "text": text,
        "target_token_count": target_tokens,
        "actual_token_count": len(
            tokenizer(text, add_special_tokens=False).input_ids
        ),
        "text_hash": stable_hash(text),
        "template_header": first_line,
        "construction": "original section header and bullet template with fixed unrelated typography stream, matched to the Qwen tokenizer length",
    }


def intervention_payload(
    *,
    memory,
    routed: Mapping[str, Any],
    arm: PayloadArm,
    target_skill_id: str,
    placebo_text: str,
) -> tuple[str, list[str]]:
    """Return prompt payload after routing on the untouched descriptor bank.

    Non-target Skills are always rendered unchanged.  When the target is
    selected, routing identity remains fixed while only its prompt payload is
    changed.
    """
    selected_id = routed.get("selected_skill_id")
    if selected_id != target_skill_id:
        return selected_payload_text(memory, routed), list(
            routed.get("injected_skill_ids", [])
        )
    arm = PayloadArm(arm)
    if arm is PayloadArm.ORIGINAL:
        return selected_payload_text(memory, routed), [target_skill_id]
    if arm is PayloadArm.PLACEBO:
        return placebo_text, [target_skill_id]
    return "", []


def first_skill_step(trajectory: Mapping[str, Any], skill_id: str) -> int | None:
    for step in trajectory.get("steps", []):
        if (
            step.get("selected_skill_id") == skill_id
            or skill_id in step.get("injected_skill_ids", [])
        ):
            return int(step["step_index"])
    return None


def build_anchor(
    *,
    trajectory: Mapping[str, Any],
    trajectory_path: str,
    skill_id: str,
    source_index: Mapping[str, Any],
) -> dict[str, Any] | None:
    trigger_step = first_skill_step(trajectory, skill_id)
    if trigger_step is None:
        return None
    steps = list(trajectory["steps"])
    trigger = steps[trigger_step]
    prefix = steps[:trigger_step]
    anchor_identity = {
        "source_trajectory_id": trajectory["trajectory_id"],
        "skill_id": skill_id,
        "trigger_step": trigger_step,
    }
    state_identity = {
        "game_id": trajectory["game_id"],
        "trigger_step": trigger_step,
        "task_description": trajectory.get("task_description", ""),
        "observation": trigger.get("observation"),
        "admissible_actions": trigger.get("admissible_actions", []),
        "history": [
            {
                "observation": step.get("observation", ""),
                "action": step.get("projected_action", ""),
            }
            for step in prefix
        ],
    }
    return {
        "schema_version": "phase1.first_invocation_anchor.v1",
        "anchor_id": stable_hash(anchor_identity)[:24],
        "state_id": stable_hash(state_identity)[:24],
        "skill_id": skill_id,
        "context_id": trajectory.get("context_id") or source_index.get("context_id"),
        "split": source_index.get("split"),
        "game_id": trajectory["game_id"],
        "environment_seed": trajectory.get(
            "environment_seed", source_index.get("environment_seed")
        ),
        "source_eval_seed": trajectory.get("eval_seed", source_index.get("eval_seed")),
        "source_checkpoint_id": trajectory.get(
            "checkpoint_id", source_index.get("checkpoint_id")
        ),
        "source_trajectory_id": trajectory["trajectory_id"],
        "source_trajectory_path": trajectory_path,
        "task_description": trajectory.get("task_description", ""),
        "trigger_step": trigger_step,
        "max_steps": int(source_index.get("max_steps", 30)),
        "remaining_steps": int(source_index.get("max_steps", 30)) - trigger_step,
        "prefix_actions": [step.get("projected_action", "") for step in prefix],
        "prefix_history": state_identity["history"],
        "prefix_rewards": [step.get("reward", 0.0) for step in prefix],
        "trigger_observation": trigger.get("observation"),
        "trigger_admissible_actions": trigger.get("admissible_actions", []),
        "trigger_prompt_text": trigger.get("prompt_text"),
        "trigger_router_state_flags": trigger.get("skill_router_state_flags", []),
        "trigger_router_scores": trigger.get("skill_router_scores", {}),
        "trigger_selected_skill_id": trigger.get("selected_skill_id"),
        "state_hash": stable_hash(state_identity),
    }


def payload_routing_copy(routed: Mapping[str, Any]) -> dict[str, Any]:
    """Return a defensive copy for lossless trajectory archiving."""
    return copy.deepcopy(dict(routed))
