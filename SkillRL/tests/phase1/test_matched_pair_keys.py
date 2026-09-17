import pandas as pd

from phase1.metrics import validate_matched_configuration


def records():
    shared = {
        "checkpoint_id": 0, "update_type": "real_rl", "rl_seed": 1,
        "eval_seed": 2, "environment_seed": 3, "game_id": "g1",
        "context_id": "clean", "skill_id": "cle_001", "success": True,
        "temperature": 0.4, "top_p": 1.0, "max_steps": 30,
        "action_projection_version": "a1", "prompt_template_version": "p1",
        "skill_bank_hash": "hash",
    }
    return [{**shared, "skill_condition": condition} for condition in ("full_bank", "minus_skill")]


def test_matched_pair_requires_identical_configuration():
    rows = records()
    assert validate_matched_configuration(pd.DataFrame(rows)) == []
    rows[1]["temperature"] = 0.8
    assert validate_matched_configuration(pd.DataFrame(rows)) == ["configuration_mismatch"]

