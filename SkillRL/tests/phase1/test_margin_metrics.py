import pandas as pd
import pytest

from phase1.metrics import (
    checkpoint_order,
    endpoint_transition_metrics,
    matched_skill_differences,
)


def test_checkpoint_order_treats_numbered_model_id_as_base():
    assert checkpoint_order("Qwen2.5-1.5B-Instruct")[0] == 0
    assert checkpoint_order("checkpoint-1")[0] == 1
    assert checkpoint_order("global_step_5")[0] == 5


def test_endpoint_transition_uses_base_and_last_checkpoint():
    rows = pd.DataFrame([
        {
            "checkpoint_id": checkpoint,
            "update_type": "real_rl",
            "rl_seed": 101,
            "skill_id": "pic_001",
            "context_id": "pick_and_place",
            "margin": margin,
            "margin_lcb": margin - 0.1,
            "margin_ucb": margin + 0.1,
        }
        for checkpoint, margin in [
            ("checkpoint-1", 0.1),
            ("Qwen2.5-1.5B-Instruct", -0.1),
            ("checkpoint-5", 0.0),
        ]
    ])
    result = endpoint_transition_metrics(rows).iloc[0]
    assert result["pre_checkpoint_id"] == "Qwen2.5-1.5B-Instruct"
    assert result["post_checkpoint_id"] == "checkpoint-5"
    assert result["delta_margin"] == pytest.approx(0.1)


def test_margin_is_full_minus_target_skill():
    base = {
        "checkpoint_id": 0, "update_type": "real_rl", "rl_seed": 1,
        "eval_seed": 2, "environment_seed": 3, "game_id": "g1",
        "context_id": "clean", "skill_id": "cle_001",
    }
    rows = [
        {**base, "skill_condition": "full_bank", "success": 1},
        {**base, "skill_condition": "minus_skill", "success": 0},
    ]
    output = matched_skill_differences(pd.DataFrame(rows))
    assert output.iloc[0]["margin"] == 1.0


def test_step_routed_margin_requires_target_selection_in_full_arm():
    base = {
        "checkpoint_id": 0, "update_type": "real_rl", "rl_seed": 1,
        "eval_seed": 2, "environment_seed": 3, "game_id": "g1",
        "context_id": "clean", "skill_id": "cle_003",
        "skill_router_version": "router-v1",
    }
    rows = [
        {
            **base, "skill_condition": "full_bank", "success": 1,
            "selected_skill_ids": ["cle_006", "gen_002"],
        },
        {
            **base, "skill_condition": "minus_skill", "success": 0,
            "selected_skill_ids": ["cle_006", "gen_002"],
        },
    ]
    assert matched_skill_differences(pd.DataFrame(rows)).empty


def test_step_routed_margin_keeps_selected_target():
    base = {
        "checkpoint_id": 0, "update_type": "real_rl", "rl_seed": 1,
        "eval_seed": 2, "environment_seed": 3, "game_id": "g1",
        "context_id": "clean", "skill_id": "cle_003",
        "skill_router_version": "router-v1",
    }
    rows = [
        {
            **base, "skill_condition": "full_bank", "success": 1,
            "selected_skill_ids": ["cle_003"],
        },
        {
            **base, "skill_condition": "minus_skill", "success": 0,
            "selected_skill_ids": ["gen_005"],
        },
    ]
    output = matched_skill_differences(pd.DataFrame(rows))
    assert output.iloc[0]["margin"] == 1.0
