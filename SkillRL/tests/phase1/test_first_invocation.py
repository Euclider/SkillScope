import pandas as pd

from phase1.compute_first_invocation_metrics import paired_effects
from phase1.first_invocation import (
    PayloadArm,
    build_anchor,
    intervention_payload,
)


class DummyMemory:
    @staticmethod
    def format_for_prompt(payload):
        items = payload["general_skills"] + payload["task_specific_skills"]
        return items[0]["title"] if items else "sentinel"


def routed_target():
    return {
        "general_skills": [],
        "task_specific_skills": [{"skill_id": "a", "title": "real"}],
        "mistakes_to_avoid": [],
        "task_type": "pick_and_place",
        "retrieval_mode": "template",
        "selected_skill_id": "a",
        "injected_skill_ids": ["a"],
    }


def test_payload_arms_do_not_change_routing_identity():
    memory = DummyMemory()
    routed = routed_target()
    original = intervention_payload(
        memory=memory, routed=routed, arm=PayloadArm.ORIGINAL,
        target_skill_id="a", placebo_text="irrelevant",
    )
    placebo = intervention_payload(
        memory=memory, routed=routed, arm=PayloadArm.PLACEBO,
        target_skill_id="a", placebo_text="irrelevant",
    )
    null = intervention_payload(
        memory=memory, routed=routed, arm=PayloadArm.NULL,
        target_skill_id="a", placebo_text="irrelevant",
    )
    assert original == ("real", ["a"])
    assert placebo == ("irrelevant", ["a"])
    assert null == ("", [])
    assert routed["selected_skill_id"] == "a"


def test_non_target_payload_is_unchanged_in_every_arm():
    memory = DummyMemory()
    routed = routed_target()
    routed["task_specific_skills"][0] = {"skill_id": "b", "title": "other"}
    routed["selected_skill_id"] = "b"
    routed["injected_skill_ids"] = ["b"]
    for arm in PayloadArm:
        assert intervention_payload(
            memory=memory, routed=routed, arm=arm,
            target_skill_id="a", placebo_text="irrelevant",
        ) == ("other", ["b"])


def test_anchor_uses_first_target_invocation_and_preserves_prefix():
    trajectory = {
        "trajectory_id": "t",
        "game_id": "game",
        "context_id": "pick_and_place",
        "task_description": "task",
        "steps": [
            {"step_index": 0, "observation": "o0", "admissible_actions": ["a0"],
             "projected_action": "a0", "reward": 0, "selected_skill_id": "b"},
            {"step_index": 1, "observation": "o1", "admissible_actions": ["a1"],
             "projected_action": "a1", "reward": 0, "selected_skill_id": "a",
             "skill_router_scores": {"a": 2}},
            {"step_index": 2, "observation": "o2", "admissible_actions": ["a2"],
             "projected_action": "a2", "reward": 0, "selected_skill_id": "a"},
        ],
    }
    anchor = build_anchor(
        trajectory=trajectory,
        trajectory_path="t.json",
        skill_id="a",
        source_index={"max_steps": 30, "eval_seed": 11, "environment_seed": 7},
    )
    assert anchor["trigger_step"] == 1
    assert anchor["prefix_actions"] == ["a0"]
    assert anchor["trigger_observation"] == "o1"
    assert anchor["remaining_steps"] == 29


def test_paired_effects_separates_semantic_total_and_prompt_effects():
    shared = {
        "checkpoint_id": "base", "anchor_id": "a", "state_id": "s",
        "source_eval_seed": 11, "environment_seed": 1, "game_id": "g",
        "context_id": "pick_and_place", "skill_id": "pic_001",
        "trigger_step": 0, "invalid_action_count": 0,
        "target_selected_count": 1, "target_payload_injection_count": 1,
        "suffix_trajectory_length": 30,
    }
    records = pd.DataFrame([
        {**shared, "payload_arm": "original", "suffix_return": 10,
         "success": True, "first_action": "a"},
        {**shared, "payload_arm": "placebo", "suffix_return": 0,
         "success": False, "first_action": "b"},
        {**shared, "payload_arm": "null", "suffix_return": 5,
         "success": False, "first_action": "a",
         "target_payload_injection_count": 0},
    ])
    result = paired_effects(records).iloc[0]
    assert result["semantic_utility"] == 10
    assert result["total_utility"] == 5
    assert result["prompt_nuisance"] == -5
    assert bool(result["original_placebo_first_action_flip"])
    assert not bool(result["original_null_first_action_flip"])
