"""Offline setting-contract tests; no policy, ALFWorld, training, or API calls."""

import json
import os
import re
import subprocess
import sys

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationToMissingValueError, MissingMandatoryValue

from scripts.inspect_skillrl_alignment import CODE_ROOT, PROFILE_PATH, inspect_alignment, load_config


@pytest.fixture
def profile():
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def config():
    return load_config()


def test_archived_config_contract(config, profile):
    result = inspect_alignment(config, profile)
    assert result["status"] == "CONFIGURATION_VALIDATED_NOT_RUN"
    assert result["errors"] == []
    assert set(result["required_runtime_values_unset"]) == set(profile["required_unset_fields"])
    assert result["pending_before_launch"]
    assert result["training_started"] is False
    assert result["gpu_forward_validated"] is False
    assert result["full_game_coverage_verified"] is False
    assert result["external_api_calls"] == result["writes"] == 0


def test_batch_units_and_schedule(config, profile):
    result = inspect_alignment(config, profile)["derived"]
    assert result["trajectories_per_iteration"] == 16 * 8 == 128
    assert result["nominal_training_trajectories"] == 19200
    assert result["local_minibatch_rows_full"] == 128 // 4 == 32
    assert result["gradient_accumulation_full_minibatch"] == 32
    assert result["opportunity_steps"] == list(range(5, 151, 5))
    assert len(result["opportunity_steps"]) == 30
    assert config.actor_rollout_ref.rollout.n == 1  # No accidental extra x8.
    assert profile["derived"]["public_script_gradient_accumulation_full_minibatch"] == 8
    assert "flattened" in profile["derived"]["ppo_minibatch_unit"]


INPUT_ALIASES = [
    ("seed", "env.seed"),
    ("model_path", "actor_rollout_ref.model.path"),
    ("train_files", "data.train_files"),
    ("val_files", "data.val_files"),
    ("run_id", "trainer.experiment_name"),
    ("output_dir", "trainer.default_local_dir"),
    ("archive_dir", "env.phase1_archive.output_dir"),
    ("router_cache", "env.skills_only_memory.step_routing.cache_path"),
    ("ray_temp_dir", "ray_init._temp_dir"),
]


@pytest.mark.parametrize("input_key,alias", INPUT_ALIASES)
def test_required_inputs_do_not_inherit_historical_defaults(config, input_key, alias):
    assert OmegaConf.is_missing(config.alignment_run, input_key)
    with pytest.raises((MissingMandatoryValue, InterpolationToMissingValueError)):
        OmegaConf.select(config, alias, throw_on_missing=True)


def test_inputs_resolve_only_after_explicit_assignment_in_memory(config, profile):
    for key, alias in INPUT_ALIASES:
        expected = 99173 if key == "seed" else f"unexecuted-test-{key}"
        OmegaConf.update(config, f"alignment_run.{key}", expected)
        assert OmegaConf.select(config, alias, throw_on_missing=True) == expected
    assert config.data.seed == config.env.seed == 99173
    assert config.env.phase1_archive.run_id == config.trainer.experiment_name
    result = inspect_alignment(config, profile)
    assert result["required_runtime_values_unset"] == []
    # Filling inputs is not an execution authorization for the archived template.
    assert len(result["errors"]) == len(INPUT_ALIASES)
    assert result["pending_before_launch"]


@pytest.mark.parametrize("key,value", [
    ("actor_rollout_ref.actor.optim.lr", 1e-5),
    ("data.train_batch_size", 8),
    ("env.rollout.n", 4),
    ("actor_rollout_ref.rollout.n", 8),
    ("actor_rollout_ref.actor.ppo_mini_batch_size", 32),
    ("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", 4),
    ("data.max_prompt_length", 6000),
    ("data.max_response_length", 1024),
    ("env.max_steps", 30),
    ("trainer.total_training_steps", 30),
    ("trainer.test_freq", 10),
    ("env.alfworld.eval_dataset", "eval_out_of_distribution"),
    ("env.alfworld.task_types", [1]),
    ("env.skills_only_memory.enable_dynamic_update", True),
    ("env.skills_only_memory.update_skills_from_train", True),
    ("env.skills_only_memory.step_routing.backend", "phase"),
    ("env.skills_only_memory.step_routing.max_api_calls", 1),
    ("trainer.resume_mode", "auto"),
])
def test_setting_drift_is_reported(config, profile, key, value):
    OmegaConf.update(config, key, value)
    result = inspect_alignment(config, profile)
    assert result["status"] == "CONFIGURATION_INVALID"
    assert any(key in error for error in result["errors"])


@pytest.mark.parametrize("key,value,error", [
    ("edit_operation", "add_only", "action space"),
    ("allowed_edit_operations", ["modify"], "action space"),
    ("approved_for_execution", True, "not execution approval"),
])
def test_phase3_design_boundary_is_enforced(config, profile, key, value, error):
    profile["phase3"][key] = value
    assert any(error in item for item in inspect_alignment(config, profile)["errors"])


def test_phase3_is_shared_editor_design_not_native_or_enabled(config, profile):
    phase3 = profile["phase3"]
    assert phase3["allowed_edit_operations"] == ["add", "delete", "modify"]
    assert phase3["same_editor_required"] is True
    assert phase3["editor_model"] is None  # Router choice does not choose the editor.
    assert phase3["approved_for_execution"] is False
    assert phase3["original_bank_writable"] is False
    assert phase3["status"] == "design_only_not_implemented_or_launched"
    assert phase3["pending_decisions"]
    assert config.env.skills_only_memory.enable_dynamic_update is False
    assert config.env.skills_only_memory.update_skills_from_train is False
    assert "controlled SkillRL-style" in profile["paper_code_differences"][-1]["resolution"]


def test_monitoring_is_not_exhaustive_or_unseen_selection(config, profile):
    evaluation = profile["evaluation"]
    assert config.data.val_batch_size == evaluation["training_monitor"]["episodes_per_check"] == 64
    assert evaluation["training_monitor"]["sampled"] is True
    assert evaluation["training_monitor"]["exhaustive"] is False
    assert evaluation["final_reports"]["seen"]["games"] == 140
    assert evaluation["final_reports"]["unseen"]["games"] == 134
    assert evaluation["no_unseen_based_selection_or_editing"] is True
    assert evaluation["aggregate_seen_unseen"] is False
    assert evaluation["unique_game_coverage_verified"] is False


def test_frozen_bank_and_router_identity_drift_is_reported(config, profile):
    profile["bank_router"]["bank_manifest_sha256"] = "0" * 64
    profile["bank_router"]["router_model"] = "incorrect-test-model"
    errors = inspect_alignment(config, profile)["errors"]
    assert "Frozen initial-bank manifest hash differs" in errors
    assert "Router model differs from the archived contract" in errors


def test_source_receipts_are_commit_pinned(profile):
    upstream = profile["upstream"]
    receipt = json.loads((CODE_ROOT / upstream["source_audit"]).read_text(encoding="utf-8"))
    assert receipt["commit"] == upstream["commit"] == "8e66726ed866a4e0a7f053586a41022798192e6c"
    assert len(receipt["files"]) == 9
    paths = [item["path"] for item in receipt["files"]]
    assert len(set(paths)) == len(paths)
    assert upstream["entrypoint"] in paths
    for item in receipt["files"]:
        assert f"/{upstream['commit']}/" in item["url"]
        assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        assert item["bytes"] > 0


@pytest.mark.parametrize("ready,exit_code", [(False, 0), (True, 2)])
def test_readonly_cli_explicitly_does_not_claim_readiness(ready, exit_code):
    command = [sys.executable, "-B", "-m", "scripts.inspect_skillrl_alignment"]
    if ready:
        command.append("--require-ready")
    result = subprocess.run(command, cwd=CODE_ROOT, text=True, capture_output=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=30)
    assert result.returncode == exit_code, result.stderr
    report = json.loads(result.stdout)
    assert report["errors"] == []
    assert report["required_runtime_values_unset"]
    assert report["pending_before_launch"]
    assert report["training_started"] is False
    assert report["external_api_calls"] == report["writes"] == 0
