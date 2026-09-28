"""Router opt-in does not change legacy manifests or launch any RL workload."""
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from agent_system.memory.router_cache import RouterBudgetExceeded
from agent_system.memory.skillrl_embedding_router import FrozenSentenceEncoder, SkillRLEmbeddingStepRouter
from skillnet_cohort.common import REPO, file_hash, read_json, require_authorization, write_new_bytes, write_new_json
from skillnet_cohort.runtime import (
    authorized_runtime, make_runtime, router_backend, router_registration, runtime_settings, verify_runtime_identity,
)
from skillnet_cohort.training import configuration

LEGACY = REPO / "docs/experiments/skillnet-phase12-preparation-v1/assets-v2/manifest.json"


@pytest.fixture(autouse=True)
def no_forward_or_network(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError("Only offline configuration/contracts are allowed in this test")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(FrozenSentenceEncoder, "encode", forbidden)


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "new-preparation"
    spec = read_json(LEGACY.parent / "spec.json")
    spec.update(router_registration("skillrl_embedding_state", model_path="/not-loaded", device="cpu"))
    manifest = read_json(LEGACY)
    for row in manifest["assets"]:
        target = root / row["path"]
        if row["path"] == "spec.json":
            write_new_json(target, spec)
        else:
            write_new_bytes(target, (LEGACY.parent / row["path"]).read_bytes())
        row["sha256"] = file_hash(target)
    write_new_json(root / "manifest.json", manifest)
    return root / "manifest.json", spec


def permit_for(prepared, tmp_path, **changes):
    preparation, _ = prepared
    permit = {"approved": True, "preparation_sha256": file_hash(preparation),
              "operations": ["training", "evaluation", "readout", "exports"],
              "router_max_api_calls": 0, "router_max_local_calls": 3,
              "router_cache_path": str(tmp_path / "router.sqlite3"), **changes}
    path = tmp_path / "permit.json"
    write_new_json(path, permit)
    return path, permit


def test_backend_opt_in_and_old_runtime_still_mini(prepared, tmp_path):
    _, spec = prepared
    before = file_hash(LEGACY.parent / "spec.json")
    old = runtime_settings(read_json(LEGACY.parent / "spec.json"), tmp_path / "old.sqlite3", 0)
    new = runtime_settings(spec, tmp_path / "new.sqlite3", max_local_calls=0)
    assert router_backend(old) == "external_llm"
    assert router_backend(new) == "skillrl_embedding_state"
    verify_runtime_identity(old)
    verify_runtime_identity(new)
    assert file_hash(LEGACY.parent / "spec.json") == before
    memory, router = make_runtime(new)
    assert isinstance(router, SkillRLEmbeddingStepRouter) and router._encoder is None
    with pytest.raises(RouterBudgetExceeded):
        router.route(memory.retrieve(""), task_description="unit", current_observation="unit",
                     admissible_actions=["look"], history=[], step_index=0)
    assert router._encoder is None
    with pytest.raises(ValueError, match="external API client"):
        make_runtime(new, client=object())


def test_training_and_readout_share_exact_protocol_and_keep_rl_recipe(prepared, tmp_path):
    prep, spec = prepared
    run = tmp_path / "new-run"
    cfg = configuration(prep, run, gpu_count=8, router_local_calls=3)
    routing = cfg.env.skills_only_memory.step_routing
    assert routing.backend == "skillrl_embedding_state" and routing.device == "cpu"
    assert routing.max_local_calls == 3 and "max_api_calls" not in routing
    assert cfg.data.train_batch_size * cfg.env.rollout.n == 128
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 128
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6
    assert cfg.algorithm.adv_estimator == "grpo" and cfg.trainer.total_training_steps == 150
    assert cfg.trainer.test_freq == cfg.trainer.save_freq == 5
    assert cfg.trainer.n_gpus_per_node == 8
    assert cfg.actor_rollout_ref.actor.fsdp_config.cpu_shard_init
    assert cfg.env.seed == cfg.data.seed == cfg.actor_rollout_ref.cohort_seed == 404
    assert not cfg.env.skills_only_memory.enable_dynamic_update
    assert not cfg.env.skills_only_memory.update_skills_from_train
    assert not run.exists()  # Merely composing configuration is read-only.
    from agent_system.memory.skillnet_runtime import create_embedding_skillnet37_runtime
    _, training_router = create_embedding_skillnet37_runtime(**{
        k: routing[k] for k in ("model_path", "device", "cache_path", "max_local_calls", "profile_path")})
    _, readout_router = make_runtime(runtime_settings(spec, routing.cache_path, max_local_calls=3))
    assert training_router.protocol_hash == readout_router.protocol_hash


def test_segment_uses_local_budget_and_does_not_change_five_update_schedule(prepared, tmp_path):
    from skillnet_cohort.segmented_training import block_config
    prep, _ = prepared
    cfg = block_config(prep, tmp_path / "run", {"router_max_local_calls": 9, "gpu_ids": list(range(8))}, 5)
    assert cfg.env.skills_only_memory.step_routing.max_local_calls == 9
    assert cfg.skillnet_cohort.segment_start == 5 and cfg.skillnet_cohort.segment_end == 10
    assert cfg.trainer.total_training_steps == 150 and cfg.env.seed == 420


def test_authorization_is_local_only_and_never_translates_to_paid_calls(prepared, tmp_path):
    prep, spec = prepared
    path, permit = permit_for(prepared, tmp_path)
    assert require_authorization(path, prep, "training") == permit
    settings = authorized_runtime(spec, permit["router_cache_path"], permit)
    assert settings["max_api_calls"] == 0 and settings["max_local_calls"] == 3
    with pytest.raises(ValueError, match="zero external API"):
        runtime_settings(spec, tmp_path / "invalid.db", 3)
    with pytest.raises(ValueError, match="cannot authorize a paid"):
        runtime_settings(read_json(LEGACY.parent / "spec.json"), tmp_path / "invalid.db", 0, max_local_calls=3)


@pytest.mark.parametrize("changes", [{"router_max_local_calls": 0}, {"router_max_local_calls": True},
                                     {"router_max_api_calls": 1}])
def test_invalid_local_permission_rejected_before_inference(prepared, tmp_path, changes):
    path, _ = permit_for(prepared, tmp_path, **changes)
    with pytest.raises(PermissionError):
        require_authorization(path, prepared[0], "evaluation")


@pytest.mark.parametrize("change", ["backend", "profile", "placement"])
def test_fail_closed_on_changed_runtime_identity(prepared, tmp_path, change):
    runtime = runtime_settings(prepared[1], tmp_path / "runtime.sqlite3")
    if change == "backend":
        runtime["router_backend"] = "unknown"
    elif change == "profile":
        runtime["router_profile_sha256"] = "0" * 64
    else:
        runtime["router_device"] = "auto"
    with pytest.raises(ValueError):
        make_runtime(runtime)
    assert not (tmp_path / "runtime.sqlite3").exists()


def test_phase2_window_registration_carries_the_same_backend(prepared, tmp_path):
    from test_preparation import fake_model, fake_result
    from skillnet_cohort.evaluate import execute_jobs, job_plan
    from skillnet_cohort.support import build_support, register_window
    from agent_system.memory.frozen_skill_bank import load_skillnet37
    prep, spec = prepared
    training = tmp_path / "training"
    identity = fake_model(training / "models/u0000")
    for name in ("batches", "old_logprobs", "new_logprobs", "optimizer_steps"):
        (training / name).mkdir()
    jobs = job_plan(prep, training / "models/u0000", 0, "valid_seen", "anchors")
    jobs["checkpoint_identity"] = identity
    execute_jobs(jobs, tmp_path / "source", lambda job: fake_result(job, load_skillnet37().skill_ids[0]))
    build_support(prep, tmp_path / "source", tmp_path / "support")
    auth, permit = permit_for(prepared, tmp_path)
    cfg = register_window(prep, tmp_path / "support/manifest.json", training, tmp_path / "window", 0, auth)
    assert cfg["runtime"]["router_backend"] == "skillrl_embedding_state"
    assert cfg["runtime"]["max_api_calls"] == 0 and cfg["runtime"]["max_local_calls"] == 3
    assert cfg["evaluation"]["arms"] == ["original", "placebo", "null"]
    assert cfg["runtime"]["router_profile_sha256"] == spec["router_profile_sha256"]
    assert cfg["runtime"]["cache_path"] == permit["router_cache_path"]
