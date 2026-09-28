"""Offline embedding-adapter contracts; fake vectors, no downloaded weights/API."""
import copy
import json
import sqlite3
from dataclasses import replace

import numpy as np
import pytest
from omegaconf import OmegaConf

from agent_system.memory.external_skill_router import ExternalRouterError
from agent_system.memory.frozen_skill_bank import FrozenSkillBankMemory, load_skillnet37
from agent_system.memory.router_cache import RouterBudgetExceeded, RouterCacheError, canonical_json, digest
from agent_system.memory.skillnet_runtime import DEFAULT_EMBEDDING_ROUTER_PROFILE, create_skillnet37_runtime
from agent_system.memory.skillrl_embedding_router import (
    EmbeddingRouterConfig, EmbeddingRouterError, SkillRLEmbeddingStepRouter, isolated_encoder_rng, load_profile, skill_texts,
)


STATE = dict(task_description="put a clean mug in cabinet 1", current_observation="You see a dirty mug.",
             admissible_actions=["take mug 1 from countertop 1", "look"], history=[], step_index=0)


class FakeEncoder:
    def __init__(self):
        self.calls = []
        self.closed = False

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        if len(texts) == 37:
            vectors[np.arange(37), np.arange(37)] = 1
        else:
            visible = json.loads(texts[0])
            vectors[0, int("picked up" in visible["current_observation"])] = 1
        return vectors, {"input_tokens": 10 * len(texts), "texts": len(texts), "encode_latency_ms": 1.0}

    def close(self):
        self.closed = True


@pytest.fixture
def runtime(tmp_path):
    memory = FrozenSkillBankMemory(load_skillnet37())
    config, files = load_profile(DEFAULT_EMBEDDING_ROUTER_PROFILE)
    encoder = FakeEncoder()
    router = SkillRLEmbeddingStepRouter(memory, config, model_files=files, model_path="/not-loaded",
        device="cpu", cache_path=tmp_path / "embedding.sqlite3", max_local_calls=4, _encoder=encoder)
    return memory, router, encoder


def route(runtime, **changes):
    memory, router, _ = runtime
    return router.route(memory.retrieve(""), **{**STATE, **changes})


def test_official_text_mapping_and_pinned_profile(runtime):
    memory, router, _ = runtime
    assert router.config == EmbeddingRouterConfig()
    assert len(skill_texts(memory)) == 37
    assert skill_texts(memory) == [f"{s.name}. {s.description}" for s in memory.bank.skills]
    assert router.protocol["selection_count"] == 1
    assert router.protocol["upstream_commit"] == "8e66726ed866a4e0a7f053586a41022798192e6c"


def test_state_changes_rescore_all_37_with_single_original_payload(runtime):
    memory, router, encoder = runtime
    a = route(runtime)
    b = route(runtime, current_observation="You picked up mug 1.")
    assert a["selected_skill_id"] == memory.bank.skill_ids[0]
    assert b["selected_skill_id"] == memory.bank.skill_ids[1]
    assert memory.format_for_prompt(a) == memory.bank.skills[0].payload
    assert len(a["injected_skill_ids"]) == 1 and len(a["candidate_skill_ids"]) == 37
    assert set(a["skill_router_scores"]) == set(memory.bank.skill_ids)
    assert [len(texts) for texts in encoder.calls] == [37, 1, 1]
    assert a["skill_router_api"]["usage"]["total_tokens"] == 380
    assert b["skill_router_api"]["usage"]["total_tokens"] == 10
    assert a["skill_router_api"]["api_calls_this_step"] == 0
    assert a["skill_router_api"]["local_calls_this_step"] == 1
    assert not router._skill_embeddings.flags.writeable


def test_cache_replay_is_zero_local_work_and_no_policy_identity(runtime):
    memory, router, encoder = runtime
    a = route(runtime)
    b = route(runtime)
    assert len(encoder.calls) == 2
    assert b["skill_router_scores"] == a["skill_router_scores"]
    assert b["skill_router_api"]["cache_hit"]
    assert b["skill_router_api"]["latency_ms"] == 0
    for key in ("usage", "index_encoding", "query_encoding"):
        assert all(v == 0 for v in b["skill_router_api"][key].values())
    replay = SkillRLEmbeddingStepRouter(memory, router.config, model_files=router.protocol["model_files"],
        model_path="/no-model-required-for-cache", device="cpu", cache_path=router.cache.path, max_local_calls=0)
    assert replay.route(memory.retrieve(""), **STATE)["selected_skill_id"] == a["selected_skill_id"]
    assert replay._encoder is None
    assert replay.stats() == dict(local_attempts=1, max_local_calls=0, successful_decisions=1,
                                 failed_attempts=0, cache_hits=2, external_api_calls=0)


def test_only_two_visible_history_entries_are_used(runtime):
    history = [{"observation": f"visible {i}", "action": "look", "reward": 100,
                "policy_checkpoint": "hidden", "game_id": "hidden"} for i in range(3)]
    route(runtime, history=history)
    query = json.loads(runtime[2].calls[-1][0])
    assert query["history"] == [{"observation": f"visible {i}", "action": "look"} for i in (1, 2)]
    assert set(query) == set(STATE)


def test_no_budget_does_not_load_encode_or_call_api(runtime):
    runtime[1].cache.max_api_calls = 0
    with pytest.raises(RouterBudgetExceeded, match="Local embedding"):
        route(runtime)
    assert runtime[2].calls == []
    with pytest.raises(EmbeddingRouterError, match="never uses"):
        runtime[1]._get_client()


@pytest.mark.parametrize("change", ["filtered", "masked", "modified"])
def test_cannot_prefilter_mask_or_rewrite_candidates(runtime, change):
    memory, router, encoder = runtime
    bundle = copy.deepcopy(memory.retrieve(""))
    if change == "filtered":
        bundle["candidate_skill_ids"] = bundle["candidate_skill_ids"][:1]
    elif change == "masked":
        bundle["disabled_skill_ids"] = [memory.bank.skill_ids[0]]
    else:
        bundle["general_skills"] = []
    with pytest.raises(ExternalRouterError):
        router.route(bundle, **STATE)
    assert encoder.calls == []


@pytest.mark.parametrize("change", ["dimension", "nan", "unnormalized", "exception"])
def test_invalid_forward_fails_closed_and_is_charged_to_local_budget(runtime, monkeypatch, change):
    def invalid(texts):
        if change == "exception":
            raise RuntimeError("synthetic failure")
        vectors = np.zeros((len(texts), 512 if change == "dimension" else 1024), dtype=np.float32)
        vectors[:, 0] = np.nan if change == "nan" else 2
        return vectors, {"input_tokens": 10}
    monkeypatch.setattr(runtime[2], "encode", invalid)
    with pytest.raises(EmbeddingRouterError, match="no retry or fallback"):
        route(runtime)
    stats = runtime[1].stats()
    assert stats["local_attempts"] == stats["failed_attempts"] == 1
    assert stats["successful_decisions"] == stats["external_api_calls"] == 0


def test_exact_ties_use_canonical_order_without_coverage_rotation(runtime, monkeypatch):
    encoder = runtime[2]
    encode = encoder.encode
    def tied(texts):
        vectors, usage = encode(texts)
        if len(texts) == 1:
            vectors[:] = 0
            vectors[0, :2] = np.float32(1 / np.sqrt(2))
        return vectors, usage
    monkeypatch.setattr(encoder, "encode", tied)
    assert route(runtime)["selected_skill_id"] == runtime[0].bank.skill_ids[0]
    assert route(runtime, step_index=1)["selected_skill_id"] == runtime[0].bank.skill_ids[0]


def test_existing_mini_cache_cannot_be_reused(tmp_path):
    memory, api = create_skillnet37_runtime(cache_path=tmp_path / "mini.sqlite3", max_api_calls=0)
    config, files = load_profile(DEFAULT_EMBEDDING_ROUTER_PROFILE)
    with pytest.raises(RouterCacheError, match="protocol mismatch"):
        SkillRLEmbeddingStepRouter(memory, config, model_files=files, model_path="not-loaded", device="cpu",
            cache_path=api.cache.path, max_local_calls=0)


def test_checksum_valid_but_wrong_cached_winner_is_rejected(runtime):
    result = route(runtime)
    cache = runtime[1].cache
    key = result["skill_router_api"]["cache_key"]
    record = cache.lookup(key)
    record["selected_skill_id"] = runtime[0].bank.skill_ids[1]
    with sqlite3.connect(cache.path) as db:
        db.execute("UPDATE decisions SET record=?, hash=? WHERE key=?", (canonical_json(record), digest(record), key))
    with pytest.raises(EmbeddingRouterError, match="canonical cosine top-1"):
        route(runtime)


def test_byte_overflow_is_not_silently_truncated(runtime):
    runtime[1]._config = replace(runtime[1].config, max_input_utf8_bytes=1)
    with pytest.raises(EmbeddingRouterError, match="no truncation"):
        route(runtime)
    assert runtime[2].calls == []


@pytest.mark.parametrize("fail", [False, True])
def test_encoder_rng_isolation_restores_all_cpu_streams_even_on_error(fail):
    import random
    import torch
    py, npr, th = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try:
        with isolated_encoder_rng("cpu"):
            random.random()
            np.random.random()
            torch.rand(7)
            if fail:
                raise RuntimeError("synthetic load/encode failure")
    except RuntimeError:
        assert fail
    after_np = np.random.get_state()
    assert py == random.getstate() and torch.equal(th, torch.get_rng_state())
    assert np.array_equal(npr[1], after_np[1]) and npr[2:] == after_np[2:]


@pytest.mark.parametrize("terminal", [False, True])
def test_manager_routes_each_live_state_and_preserves_archive(runtime, tmp_path, monkeypatch, terminal):
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.memory import skillnet_runtime
    from phase1.archive import archive_rollout_batch
    from test_environment_integration import FakeEnvironment
    memory, router, encoder = runtime
    monkeypatch.setattr(skillnet_runtime, "create_embedding_skillnet37_runtime", lambda **kwargs: (memory, router))
    cfg = OmegaConf.create({"env": {"history_length": 2, "use_skills_only_memory": True,
        "alfworld": {"action_only_prompt": True}, "skills_only_memory": {"top_k": 1, "step_routing": {
            "enabled": True, "backend": "skillrl_embedding_state", "model_path": "/not-loaded", "device": "cpu",
            "cache_path": str(router.cache.path), "max_local_calls": 4}}}})
    manager = AlfWorldEnvironmentManager(FakeEnvironment(terminal), lambda acts, cmds: (acts, [True]), cfg)
    manager.reset({})
    observation, _, _, infos = manager.step(["take mug 1 from countertop 1"])
    assert len(encoder.calls) == (2 if terminal else 3)
    if not terminal:
        assert memory.bank.skills[1].payload in observation["text"][0]
    assert infos[0]["skill_router_api"]["api_calls_this_step"] == 0
    assert len(infos[0]["skill_router_scores"]) == 37
    archive_rollout_batch(output_dir=tmp_path, run_id="embed-unit", split="train", global_step=1,
        total_batch_list=[[{"active_masks": True, "rewards": 0, "attention_mask": [1], "responses": [1]}]],
        total_infos=[[infos[0]]], episode_rewards=[0], episode_lengths=[1], trajectory_ids=["synthetic"])
    saved = json.loads((tmp_path / "trajectories/embed-unit/synthetic.json").read_bytes())
    assert saved["steps"][0]["skill_router_api"]["backend"] == "skillrl_embedding_state"
    assert len(saved["steps"][0]["skill_router_scores"]) == 37
