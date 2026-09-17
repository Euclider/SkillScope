"""Frozen, observable-state router for step-level ALFWorld Skill selection."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

ROUTER_VERSION = "alfworld-observable-phase-router-v1"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "any", "at", "be", "before", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "the", "then", "to",
    "with", "you", "your",
}

_PHASE_MARKERS: dict[str, tuple[str, ...]] = {
    "initial": (
        "at the very start", "at the start", "initial", "begin", "right after parsing",
        "after reading the goal", "receiving", "as soon as the goal",
    ),
    "visible_target": (
        "visible", "visual confirmation", "first sight", "becomes visible", "spotting",
        "discovered", "first observation", "comes into view",
    ),
    "holding_target": (
        "holding", "in hand", "carrying", "after acquiring", "after picking",
        "after picking up", "upon holding", "once the target object is in hand",
    ),
    "process_available": (
        "clean", "cleaning", "sink", "heat", "heating", "microwave", "cool",
        "cooling", "fridge", "appliance",
    ),
    "processed_recently": (
        "after cleaning", "after heating", "after cooling", "right after the cleaning",
        "right after the heating", "right after the cooling", "post-heat", "post-cooling",
        "action completes", "action succeeds", "before placing", "before final placement",
    ),
    "ready_to_place": (
        "place", "placement", "destination", "drop", "delivery", "target location",
        "target receptacle", "final location",
    ),
    "closed_container": (
        "closed container", "unopened", "open and inspect", "container sweep",
    ),
    "search": (
        "search", "locating", "unfound", "unexplored", "remains unknown", "hunt",
        "hasn't been found", "has not been found", "surface", "container",
    ),
    "loop": (
        "loop", "repeated", "revisit", "already-searched", "no new", "stuck",
    ),
    "lamp_available": (
        "lamp", "desklamp", "switch lamp on", "toggle",
    ),
    "multiple_targets": (
        "two", "multiple", "count", "instances", "first and second",
    ),
}


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS}


def _item_id(item: Mapping[str, Any]) -> str | None:
    return item.get("skill_id") or item.get("mistake_id")


def _item_text(item: Mapping[str, Any]) -> str:
    return " ".join(
        str(item.get(field, ""))
        for field in ("title", "principle", "when_to_apply", "description", "how_to_avoid")
    ).lower()


class FrozenStepSkillRouter:
    """Select exactly one Skill from an episode candidate bundle.

    The router is deterministic and uses only information visible to the agent:
    task text, current observation, admissible actions, and action/observation
    history.  It has no trainable parameters and therefore stays fixed while the
    policy is updated.
    """

    version = ROUTER_VERSION

    def __init__(self, include_common_mistakes: bool = False):
        self.include_common_mistakes = include_common_mistakes

    @staticmethod
    def _state_flags(
        *,
        task_description: str,
        current_observation: str,
        admissible_actions: Sequence[str],
        history: Sequence[Mapping[str, str]],
        step_index: int,
    ) -> dict[str, bool]:
        task = task_description.lower()
        observation = current_observation.lower()
        actions = [action.lower().strip() for action in admissible_actions]
        recent_actions = [str(item.get("action", "")).lower().strip() for item in history[-3:]]

        take_actions = [action for action in actions if action.startswith(("take ", "pick "))]
        put_actions = [action for action in actions if action.startswith(("put ", "move "))]
        process_actions = [
            action for action in actions
            if action.startswith(("clean ", "heat ", "cool "))
        ]
        task_tokens = _tokens(task)
        visible_target = any(bool(task_tokens & _tokens(action)) for action in take_actions)
        holding_target = bool(put_actions or process_actions)
        processed_recently = (
            any(phrase in observation for phrase in ("you clean", "you heat", "you cool"))
            or bool(recent_actions and recent_actions[-1].startswith(("clean ", "heat ", "cool ")))
        )
        repeated_action = (
            len(recent_actions) >= 2
            and recent_actions[-1]
            and recent_actions[-1] == recent_actions[-2]
        )
        lamp_available = any(
            "desklamp" in action and action.startswith(("use ", "turn "))
            for action in actions
        )
        # Initial routing has its own strong phase signal (for plan/location
        # priors); do not prematurely label the untouched room as a failed
        # search state.
        search = step_index > 0 and not visible_target and not holding_target and not process_actions

        return {
            "initial": step_index == 0,
            "visible_target": visible_target,
            "holding_target": holding_target,
            "process_available": bool(process_actions),
            "processed_recently": processed_recently,
            "ready_to_place": bool(put_actions),
            "closed_container": any(action.startswith("open ") for action in actions),
            "search": search,
            "loop": observation.strip() == "nothing happens." or repeated_action,
            "lamp_available": lamp_available,
            "multiple_targets": " two " in f" {task} " or task.startswith("find two"),
        }

    def route(
        self,
        candidate_bundle: Mapping[str, Any],
        *,
        task_description: str,
        current_observation: str,
        admissible_actions: Sequence[str],
        history: Sequence[Mapping[str, str]],
        step_index: int,
    ) -> dict[str, Any]:
        general = list(candidate_bundle.get("general_skills", []))
        task_specific = list(candidate_bundle.get("task_specific_skills", []))
        mistakes = (
            list(candidate_bundle.get("mistakes_to_avoid", []))
            if self.include_common_mistakes else []
        )
        typed_candidates = (
            [("general", item) for item in general]
            + [("task_specific", item) for item in task_specific]
            + [("mistake", item) for item in mistakes]
        )
        candidate_ids = [item_id for _, item in typed_candidates if (item_id := _item_id(item))]
        flags = self._state_flags(
            task_description=task_description,
            current_observation=current_observation,
            admissible_actions=admissible_actions,
            history=history,
            step_index=step_index,
        )

        query_text = " ".join(
            [task_description, current_observation, *admissible_actions]
            + [str(item.get("observation", "")) for item in history[-2:]]
            + [str(item.get("action", "")) for item in history[-2:]]
        )
        query_tokens = _tokens(query_text)
        scores: dict[str, float] = {}
        score_details: dict[str, dict[str, float]] = {}

        for position, (kind, item) in enumerate(typed_candidates):
            item_id = _item_id(item)
            if not item_id:
                continue
            descriptor = _item_text(item)
            lexical = min(1.5, 0.08 * len(query_tokens & _tokens(descriptor)))
            phase_score = 0.0
            matched_phases = 0
            for phase, active in flags.items():
                if not active:
                    continue
                matches = sum(marker in descriptor for marker in _PHASE_MARKERS[phase])
                if matches:
                    matched_phases += 1
                    phase_score += min(3.0, 1.15 + 0.35 * (matches - 1))
            task_specificity = 0.35 if kind == "task_specific" else 0.0
            # Stable document-order tie break, too small to override semantics.
            tie_break = -position * 1e-6
            total = lexical + phase_score + task_specificity + tie_break
            scores[item_id] = total
            score_details[item_id] = {
                "lexical": lexical,
                "phase": phase_score,
                "task_specificity": task_specificity,
                "matched_phases": float(matched_phases),
                "total": total,
            }

        selected_id = max(scores, key=scores.get) if scores else None
        selected_kind = None
        selected_item = None
        for kind, item in typed_candidates:
            if _item_id(item) == selected_id:
                selected_kind = kind
                selected_item = item
                break

        result = dict(candidate_bundle)
        result["general_skills"] = [selected_item] if selected_kind == "general" else []
        result["task_specific_skills"] = [selected_item] if selected_kind == "task_specific" else []
        result["mistakes_to_avoid"] = [selected_item] if selected_kind == "mistake" else []
        # ``retrieved`` records the bank items matched before an evaluation
        # mask; ``candidate`` records the enabled items the router could
        # actually choose on this step.  Keeping both makes MINUS/NO-SKILL
        # interventions auditable.
        result["retrieved_skill_ids"] = list(
            candidate_bundle.get("retrieved_skill_ids", candidate_ids)
        )
        result["candidate_skill_ids"] = candidate_ids
        result["injected_skill_ids"] = [selected_id] if selected_id else []
        result["selected_skill_id"] = selected_id
        result["skill_router_version"] = self.version
        result["skill_router_scores"] = scores
        result["skill_router_score_details"] = score_details
        result["skill_router_state_flags"] = [name for name, active in flags.items() if active]
        result["skill_router_selection_reason"] = (
            "deterministic observable-state phase/description score"
            if selected_id else "no enabled Skill candidates"
        )
        return result
