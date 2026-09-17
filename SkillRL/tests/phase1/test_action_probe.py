import pandas as pd
import pytest

from phase1.action_probe import replace_memory_section
from phase1.compute_action_probe_metrics import validate_fixed_state_matches


def test_probe_replaces_only_memory_content():
    prompt = "prefix\n## Retrieved Relevant Experience\n\nold\n\n## Current Progress\nsuffix"
    assert replace_memory_section(prompt, "new") == "prefix\n## Retrieved Relevant Experience\n\nnew\n\n## Current Progress\nsuffix"


def test_probe_replaces_initial_step_selected_skill_content():
    prompt = "prefix\n## Selected Skill For This Step\n\nold\n\nYour admissible actions are: []"
    assert replace_memory_section(prompt, "new") == "prefix\n## Selected Skill For This Step\n\nnew\n\nYour admissible actions are: []"


def test_probe_rejects_initial_prompt_without_skill_section():
    with pytest.raises(ValueError):
        replace_memory_section("initial prompt", "new")


def test_probe_rejects_prompt_drift_between_checkpoints():
    rows = pd.DataFrame([
        {
            "probe_id": "p", "skill_id": "s", "skill_condition": "full_bank",
            "checkpoint_id": "0", "prompt_hash": "a",
            "candidate_action_log_probs": {"open door": 0.0},
        },
        {
            "probe_id": "p", "skill_id": "s", "skill_condition": "full_bank",
            "checkpoint_id": "5", "prompt_hash": "b",
            "candidate_action_log_probs": {"open door": 0.0},
        },
    ])
    with pytest.raises(ValueError, match="Prompt changed"):
        validate_fixed_state_matches(rows)
