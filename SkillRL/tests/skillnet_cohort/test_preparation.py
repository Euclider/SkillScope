"""Offline contracts only: no real environment, weights, Ray cluster or API."""
import random
from pathlib import Path

import numpy as np
import pytest

from agent_system.memory.frozen_skill_bank import load_skillnet37
from skillnet_cohort.assets import LocalTokenizer, model_inventory
from skillnet_cohort.common import (
    REPO, checked_relative, digest, exclusive_writer, file_hash, load_preparation,
    read_json, require_authorization, write_new_bytes, write_new_json,
)
from skillnet_cohort.evaluate import collect_results, execute_jobs, job_plan, verify_prediction
from skillnet_cohort.runtime import BoundedPolicy, admit_capture, disk_gate, seed_process
from skillnet_cohort.support import build_support, register_window
from skillnet_cohort.training import configuration, execute, plan

PREPARATION = REPO / "docs/experiments/skillnet-phase12-preparation-v1/assets-v2/manifest.json"


@pytest.fixture(autouse=True)
def forbid_real_work(monkeypatch):
    import socket
    from transformers import AutoModelForCausalLM
    import phase1.eval_skill_margin as evaluator

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline tests cannot load weights, connect, or construct a real environment")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(evaluator, "SingleGameEnvironment", forbidden)


def fake_result(job, skill=None):
    return {"success": False, "trajectory_id": job["job_id"], "game_id": job["game_id"],
            "context_id": "all_alfworld", "environment_seed": job["environment_seed"],
            "eval_seed": job["eval_seed"], "checkpoint_id": f"u{job['update']:04d}",
            "steps": [{"step_index": 0, "selected_skill_id": skill, "observation": "synthetic",
                       "admissible_actions": ["look"], "projected_action": "look"}]}


def fake_model(path):
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        write_new_json(path / name, {})
    write_new_bytes(path / "chat_template.jinja", b"unit fixture")
    write_new_bytes(path / "model.safetensors", b"unit fixture, never loaded")
    return model_inventory(path)


def test_assets_and_confirmed_schedule():
    assert load_preparation(PREPARATION)["experiments_started"] is False
    spec = read_json(PREPARATION.parent / "spec.json")
    inventory = read_json(PREPARATION.parent / "games.json")
    assert spec["seed"] == 404 and spec["seed_confirmed"]
    assert spec["readout"]["horizon"] == 5
    assert [(w["start"], w["end"]) for w in spec["readout"]["windows"]] == [(i, i+5) for i in range(0,150,5)]
    assert {k: v["count"] for k, v in inventory["splits"].items()} == {"train":3553,"valid_seen":140,"valid_unseen":134}
    streams = [spec["evaluation"][key] for key in ("performance_seeds", "anchor_source_seeds", "evidence_seeds", "gold_seeds")]
    assert sum(map(len, streams)) == len(set(sum(streams, [])))


def test_superseded_payload_only_controls_cannot_launch():
    rejected = PREPARATION.parent.parent / "assets/manifest.json"
    with pytest.raises(ValueError, match="prompt-boundary"):
        load_preparation(rejected)


def test_runtime_configuration_keeps_public_recipe_and_enables_capture(tmp_path):
    cfg = configuration(PREPARATION, tmp_path / "new")
    assert cfg.env.seed == cfg.data.seed == cfg.actor_rollout_ref.cohort_seed == 404
    assert cfg.data.train_batch_size * cfg.env.rollout.n == 128
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 128
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert cfg.actor_rollout_ref.rollout.micro_batch_size == 1
    assert cfg.actor_rollout_ref.rollout.n == 1
    assert cfg.phase2.enabled and cfg.algorithm.adv_estimator == "grpo"
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6
    assert cfg.trainer.total_training_steps == 150
    assert cfg.trainer.test_freq == cfg.trainer.save_freq == 5
    assert cfg.data.filter_overlong_prompts is False and cfg.data.truncation == "error"
    assert cfg.trainer.resume_mode == "disable"
    assert cfg.env.skills_only_memory.step_routing.max_api_calls == 0
    report = plan(PREPARATION, tmp_path / "new")
    assert not report["training_started"] and not report["legacy_elastic_launcher_used"]
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("name", ["PHASE2_ELASTIC_TRAINING", "PHASE2_CPU_ADAM"])
def test_legacy_execution_overrides_cannot_change_new_recipe(monkeypatch, name):
    from skillnet_cohort.training import reject_legacy_overrides
    monkeypatch.setenv(name, "1")
    with pytest.raises(ValueError, match=name):
        reject_legacy_overrides()


@pytest.mark.parametrize("module_name", ["phase2.measure", "phase2.evaluate"])
def test_new_readout_entrypoints_require_authorization_before_work(supported_window, tmp_path, monkeypatch, module_name):
    import importlib
    import sys
    training, _, support, _ = supported_window
    root = tmp_path / "window"
    register_window(PREPARATION, support / "manifest.json", training, root, 0)
    args = [module_name, "--root", str(root), "--update", "0"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(PermissionError):
        importlib.import_module(module_name).main()
    assert not (root / "evaluations").exists() and not (root / "signals").exists()


def test_placeholder_rows_have_correct_modality_and_cardinality():
    import pyarrow.parquet as pq
    for name, count in (("train",16), ("seen-monitor",64)):
        rows = pq.read_table(PREPARATION.parent / f"datasets/{name}.parquet").to_pylist()
        assert len(rows) == count
        assert all(row["data_source"] == "text" and row["prompt"] == [{"role":"user","content":""}] for row in rows)


def test_all_controls_exact_local_tokens():
    spec = read_json(PREPARATION.parent / "spec.json")
    tokenizer = LocalTokenizer(spec["model_path"])
    controls = read_json(PREPARATION.parent / "placebos.json")["controls"]
    bank = load_skillnet37()
    assert set(controls) == set(bank.skill_ids)
    for skill in bank.skills:
        control = controls[skill.skill_id]
        assert control["original_payload_sha256"] == skill.payload_sha256
        assert len(tokenizer.encode(skill.payload)) == len(tokenizer.encode(control["text"])) < 4096


def test_no_clobber_and_exclusive_writer(tmp_path):
    path = tmp_path / "asset.json"
    write_new_json(path, {"a": 1})
    write_new_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_new_json(path, {"a": 2})
    with exclusive_writer(tmp_path):
        with pytest.raises(RuntimeError, match="active writer"):
            with exclusive_writer(tmp_path):
                pass
    assert read_json(path) == {"a": 1}


@pytest.mark.parametrize("path", ["../escape", "/tmp/escape", "."])
def test_manifest_paths_cannot_escape(tmp_path, path):
    with pytest.raises(ValueError):
        checked_relative(tmp_path, path)


@pytest.mark.parametrize("operation", ["training", "evaluation", "exports"])
def test_authorization_missing_or_wrong_scope(tmp_path, operation):
    with pytest.raises(PermissionError):
        require_authorization(None, PREPARATION, operation)
    permit = tmp_path / "permit.json"
    write_new_json(permit, {"approved": True, "preparation_sha256": "another", "operations": [operation]})
    with pytest.raises(PermissionError):
        require_authorization(permit, PREPARATION, operation)


def test_export_authorization_does_not_require_a_paid_budget(tmp_path):
    permit = tmp_path / "permit.json"
    write_new_json(permit, {"approved": True, "preparation_sha256": file_hash(PREPARATION), "operations": ["exports", "evaluation"]})
    require_authorization(permit, PREPARATION, "exports")
    with pytest.raises(PermissionError, match="budget"):
        require_authorization(permit, PREPARATION, "evaluation")


def test_training_and_evaluation_fail_before_side_effects(tmp_path):
    from skillnet_cohort.evaluate import execute as evaluate
    from skillnet_cohort.checkpoints import export_base
    with pytest.raises(PermissionError):
        execute(PREPARATION, tmp_path / "run", None)
    with pytest.raises(PermissionError):
        evaluate(PREPARATION, "missing", 0, "valid_seen", "performance", tmp_path / "eval", None)
    with pytest.raises(PermissionError):
        export_base(PREPARATION, tmp_path / "export", None)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("split,expected", [("valid_seen",140), ("valid_unseen",134)])
def test_exhaustive_fake_eval_and_idempotent_resume(tmp_path, split, expected):
    jobs = job_plan(PREPARATION, tmp_path / "model", 0, split)
    calls = []

    def runner(job):
        calls.append(job["job_id"])
        return fake_result(job)

    result = execute_jobs(jobs, tmp_path / "eval", runner)
    assert result["episodes"] == result["unique_games"] == expected
    assert sum(row["episodes"] for row in result["per_task"].values()) == expected
    assert execute_jobs(jobs, tmp_path / "eval", runner) == result
    assert len(calls) == expected


def test_partial_eval_resumes_only_missing_games(tmp_path):
    jobs = job_plan(PREPARATION, tmp_path / "model", 0, "valid_seen")
    calls = []

    def interrupted(job):
        calls.append(job["job_id"])
        if len(calls) == 4:
            raise RuntimeError("unit interruption")
        return fake_result(job)

    with pytest.raises(RuntimeError, match="unit interruption"):
        execute_jobs(jobs, tmp_path, interrupted)
    rows, missing = collect_results(jobs, tmp_path)
    assert len(rows) == 3 and len(missing) == 137
    assert not (tmp_path / "completion.json").exists()
    resumed = []
    execute_jobs(jobs, tmp_path, lambda job: (resumed.append(job["job_id"]), fake_result(job))[1])
    assert len(resumed) == 137 and not set(calls[:3]) & set(resumed)


@pytest.mark.parametrize("problem", ["foreign", "corrupt", "identity", "duplicate_plan"])
def test_eval_rejects_invalid_evidence(tmp_path, problem):
    jobs = job_plan(PREPARATION, tmp_path / "model", 0, "valid_seen")
    job = jobs["jobs"][0]
    row = {"job": job, "status": "complete", "result": fake_result(job), "result_sha256": digest(fake_result(job))}
    name = job["job_id"]
    if problem == "foreign":
        name = "not-an-expected-job"
    elif problem == "corrupt":
        row["result"]["success"] = True
    elif problem == "identity":
        row["job"] = {**job, "eval_seed": 999}
    else:
        jobs["jobs"].append(job)
    write_new_json(tmp_path / "results" / f"{name}.json", row)
    with pytest.raises(ValueError):
        collect_results(jobs, tmp_path)


@pytest.fixture
def supported_window(tmp_path):
    training = tmp_path / "training"
    checkpoint = training / "models/u0000"
    identity = fake_model(checkpoint)
    for name in ("batches", "old_logprobs", "new_logprobs", "optimizer_steps"):
        (training / name).mkdir(parents=True)
    jobs = job_plan(PREPARATION, checkpoint, 0, "valid_seen", "anchors")
    jobs["checkpoint_identity"] = identity
    skill = load_skillnet37().skill_ids[0]
    source, support = tmp_path / "source", tmp_path / "support"
    execute_jobs(jobs, source, lambda job: fake_result(job, skill))
    manifest = build_support(PREPARATION, source, support)
    return training, source, support, manifest


def test_natural_support_accounts_for_all_37_without_forcing(supported_window):
    _, _, _, manifest = supported_window
    assert len(manifest["coverage"]) == 37
    supported = [row for row in manifest["coverage"] if row["supported"]]
    assert len(supported) == 1 and supported[0]["occurrences"] == 280
    assert supported[0]["games"] == 140 and supported[0]["selected"] == 50
    assert all(row["unsupported_reason"] == "insufficient_natural_support" for row in manifest["coverage"] if not row["supported"])
    assert manifest["anchor_sets"][0]["context_id"] == "all_alfworld"
    control = read_json(manifest["anchor_sets"][0]["placebo_path"])
    assert control["original_text"] == load_skillnet37().get(supported[0]["skill_id"]).payload


def test_window_adapter_freezes_assets_and_keeps_shared_math(supported_window, tmp_path):
    from phase2.protocol import evaluation_jobs, validate_extended
    training, _, support, _ = supported_window
    root = tmp_path / "window"
    cfg = register_window(PREPARATION, support / "manifest.json", training, root, 0)
    assert cfg["windows"] == [{"start": 0, "end": 5, "role": "test"}]
    assert cfg["post_updates"] == [1, 2, 3, 4, 5]
    assert cfg["evaluation"]["split"] == "valid_seen"
    assert cfg["prediction"]["enable_fitted_models"] is False
    assert len(evaluation_jobs(cfg, REPO)) == 50 * 6 * 3
    assert (root / "models").resolve() == (training / "models").resolve()
    validate_extended(cfg, REPO)
    with pytest.raises(FileExistsError):
        register_window(PREPARATION, support / "manifest.json", training, root, 0)


def test_prediction_lock_binds_cohort_protocol_window_features_and_endpoint(supported_window, tmp_path):
    training, _, support, _ = supported_window
    root = tmp_path / "window"
    cfg = register_window(PREPARATION, support / "manifest.json", training, root, 0)
    directory = root / "window_signals/u0000-u0005"
    write_new_bytes(directory / "skill_context_features.parquet", b"offline synthetic features")
    features_hash = file_hash(directory / "skill_context_features.parquet")
    write_new_json(directory / "committed.json", {"global_update":5,"start_update":0,"features_sha256":features_hash,"gold_read":False})
    write_new_json(directory / "prediction.json", {"global_update":5,"start_update":0,"features_sha256":features_hash,
                   "target_gold_read":False,"rl_path_id":cfg["rl_path_id"],"protocol_sha256":file_hash(root/"protocol.json")})
    prediction = directory / "prediction.json"
    assert verify_prediction(prediction, PREPARATION, 5, training / "models/u0005")["sha256"] == file_hash(prediction)
    with pytest.raises(ValueError, match="checkpoint"):
        verify_prediction(prediction, PREPARATION, 5, training / "models/u0010")
    with pytest.raises(PermissionError):
        verify_prediction(prediction, PREPARATION, 10)
    with pytest.raises(PermissionError):
        verify_prediction(None, PREPARATION, 5)


def test_process_seed_replays_cpu_streams_and_separates_ranks():
    import torch
    def sample():
        return random.random(), float(np.random.random()), float(torch.rand(()))
    seed_process(404)
    a = sample()
    seed_process(404)
    assert sample() == a
    assert seed_process(404, 1) == 100404
    assert sample() != a


def test_resource_gate_preserves_existing_files(tmp_path):
    write_new_bytes(tmp_path / "evidence", b"do not delete")
    with pytest.raises(OSError):
        disk_gate(tmp_path, 10, minimum_free_bytes=0, maximum_run_bytes=15)
    assert (tmp_path / "evidence").read_bytes() == b"do not delete"
    write_new_json(tmp_path / "resource_limits.json", {"vocab_size":248320,"checkpoint_reserve_bytes":0,
                   "minimum_free_bytes":0,"maximum_run_bytes":1_000_000})
    with pytest.raises(OSError):
        admit_capture(tmp_path, 1)
    assert not (tmp_path / "old_logprobs").exists()


def test_bound_policy_fails_before_generation():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "wrap " + messages[0]["content"]
        def encode(self, text, **kwargs):
            return text.split()
    class Policy:
        tokenizer = Tokenizer()
        def generate(self, prompt, *args, **kwargs):
            return "called"
    bounded = BoundedPolicy(Policy(), 3)
    assert bounded.generate("one two") == "called"
    with pytest.raises(ValueError, match="no truncation"):
        bounded.generate("one two three")


@pytest.mark.parametrize("with_history", [False, True])
def test_all_original_placebo_full_prompts_and_counterinputs_are_aligned(with_history):
    import torch
    from transformers import AutoTokenizer
    from agent_system.environments.prompts.alfworld import ALFWORLD_TEMPLATE_NO_HIS_WITH_MEMORY, ALFWORLD_TEMPLATE_WITH_MEMORY, use_action_only_instruction
    from phase2.measure import counter_input
    tokenizer = AutoTokenizer.from_pretrained(read_json(PREPARATION.parent / "spec.json")["model_path"], local_files_only=True)
    controls = read_json(PREPARATION.parent / "placebos.json")["controls"]
    template = ALFWORLD_TEMPLATE_WITH_MEMORY if with_history else ALFWORLD_TEMPLATE_NO_HIS_WITH_MEMORY
    for skill in load_skillnet37().skills:
        prompts = [use_action_only_instruction(template.format(
            task_description="put a mug on a table", current_observation="You see a mug and a table.",
            step_count=2, history_length=2, action_history="Looked at a mug; walked to table 1.", current_step=3,
            retrieved_memories=payload, admissible_actions="'look'")) for payload in (skill.payload, controls[skill.skill_id]["text"])]
        ids = [tokenizer.encode(tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False), add_special_tokens=False) for prompt in prompts]
        assert len(ids[0]) == len(ids[1]) < 4096
        original = ids[0]
        tensors = {"input_ids":torch.tensor([original + [10,11]]),
                   "attention_mask":torch.ones((1,len(original)+2), dtype=torch.long),
                   "responses":torch.tensor([[10,11]])}
        metadata = {"decision_id":"synthetic", "info":{"prompt_text":prompts[0], "phase2_payload_text":skill.payload}}
        for arm in ("placebo", "null"):
            changed, mask, _ = counter_input(tokenizer, metadata, tensors, 0, arm, controls[skill.skill_id]["text"])
            assert changed[-2:].tolist() == [10,11]
            assert mask[-2:].tolist() == [1,1]
            if arm == "placebo":
                assert changed[:-2].tolist() == ids[1]
