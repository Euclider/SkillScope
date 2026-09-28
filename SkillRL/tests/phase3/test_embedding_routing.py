"""Portable fake-vector tests; no policy, benchmark, model weights or API."""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from agent_system.memory.router_cache import RouterBudgetExceeded
from agent_system.memory.skillnet_runtime import DEFAULT_EMBEDDING_ROUTER_PROFILE
from phase3.bank import Bank
from phase3.common import ProtocolError, write_new
from phase3.embedding_routing import EmbeddingRouterPool, local_totals, validate_settings
from phase3.prepare import ARMS, validate_runtime
from skillnet_cohort.common import file_hash


STATE = dict(task_description="put a clean mug in cabinet 1", current_observation="You hold a mug.",
             admissible_actions=["look"], history=[], step_index=0)


def settings(limit=10):
    return {"backend": "skillrl_embedding_state", "model_path": "/not-loaded",
            "device": "cpu", "profile_sha256": file_hash(DEFAULT_EMBEDDING_ROUTER_PROFILE), "max_local_calls": limit}


def batched_settings(limit=10, device="cuda:0"):
    execution = {"mode": "state_batch_fp32_v1", "intra_op_threads": 1}
    if device.startswith("cuda:"):
        execution["shared_gpu_physical_id"] = 0
        execution["transport"] = "subprocess_pipe_v1"
    return {**settings(limit), "device": device, "execution": execution}


class Encoder:
    def __init__(self):
        self.calls = []
    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        if len(texts) == 1 and texts[0].startswith('{'):
            vectors[0, 0] = 1
        else:
            vectors[np.arange(len(texts)), np.arange(len(texts))] = 1
        return vectors, {"input_tokens": 10*len(texts), "texts": len(texts), "encode_latency_ms": 1.}
    def close(self):
        pass


class BatchEncoder(Encoder):
    def encode(self, texts):
        vectors, accounting = super().encode(texts)
        if len(texts) > 1 and texts[0].startswith('{'):
            vectors[:] = 0
            vectors[:, 0] = 1
        return vectors, {**accounting, "input_token_counts": [10] * len(texts)}


def proposal(kind, bank):
    targets = [bank.skills[0]] if kind in ("MODIFY", "DELETE") else list(bank.skills[:2]) if kind == "MERGE" else []
    return {"operations": [{"op": kind, "targets": [{"skill_id": s.skill_id, "version_sha256": s.version_sha256} for s in targets],
        "skill": None if kind in ("DELETE", "NOOP") else {"name": "new", "description": "new reusable description", "body": "Guidance"},
        "rationale": "unit evidence", "evidence_ids": ["e"]}]}


def decision(pool, bank, **change):
    router = pool.for_bank(bank)
    return router.route(router.memory.retrieve(""), **{**STATE, **change})


@pytest.mark.parametrize("kind,size", [("ADD",38), ("MODIFY",37), ("DELETE",36), ("MERGE",36)])
def test_edit_reindexes_complete_active_library_and_separates_cache(tmp_path, kind, size):
    bank = Bank.initial("readout_d")
    encoder = Encoder()
    pool = EmbeddingRouterPool(settings(), tmp_path / "local.sqlite3", bank.branch_id, _encoder=encoder)
    first = decision(pool, bank)
    changed, _ = bank.apply(proposal(kind, bank), event_id="u5", evidence_ids={"e"})
    second = decision(pool, changed)
    assert len(second["candidate_skill_ids"]) == len(second["skill_router_scores"]) == size
    assert second["skill_router_api"]["protocol_hash"] != first["skill_router_api"]["protocol_hash"]
    assert second["skill_router_api"]["backend"] == "phase3_skillrl_embedding_state"
    assert second["skill_version_sha256"] == changed.active_versions[second["selected_skill_id"]]
    assert [len(c) for c in encoder.calls] == [37, 1, size, 1]
    assert decision(pool, bank)["skill_router_api"]["cache_hit"]
    report = local_totals(pool.ledger.path)
    assert report["local_attempts"] == report["bank_versions"] == 2
    assert report["cache_hits"] == 1 and report["external_api_calls"] == 0
    assert report["usage_known_subtotals"]["total_tokens"] == (37+size+2)*10


def test_total_local_budget_does_not_reset_when_bank_grows(tmp_path):
    bank = Bank.initial("readout_d")
    encoder = Encoder()
    pool = EmbeddingRouterPool(settings(1), tmp_path / "local.sqlite3", bank.branch_id, _encoder=encoder)
    decision(pool, bank)
    changed, _ = bank.apply(proposal("ADD", bank), event_id="u5", evidence_ids={"e"})
    with pytest.raises(RouterBudgetExceeded):
        decision(pool, changed)
    assert len(encoder.calls) == 2
    assert decision(pool, bank)["skill_router_api"]["cache_hit"]
    assert local_totals(pool.ledger.path)["local_attempts"] == 1


def test_different_process_instances_share_budget_and_replay(tmp_path):
    bank = Bank.initial("readout_p")
    a = EmbeddingRouterPool(settings(1), tmp_path / "local.sqlite3", bank.branch_id, _encoder=Encoder())
    b = EmbeddingRouterPool(settings(1), tmp_path / "local.sqlite3", bank.branch_id, _encoder=Encoder())
    with ThreadPoolExecutor(2) as executor:
        rows = list(executor.map(lambda p: decision(p, bank), [a, b]))
    assert sum(r["skill_router_api"]["local_calls_this_step"] for r in rows) == 1
    assert local_totals(a.ledger.path)["cache_hits"] == 1
    with pytest.raises(RouterBudgetExceeded):
        decision(b, bank, step_index=1)


def test_foreign_branch_or_changed_budget_rejected(tmp_path):
    pool = EmbeddingRouterPool(settings(1), tmp_path / "local.sqlite3", "readout_d", _encoder=Encoder())
    with pytest.raises(ProtocolError, match="Foreign branch"):
        pool.for_bank(Bank.initial("readout_c"))
    with pytest.raises(ProtocolError, match="profile"):
        EmbeddingRouterPool(settings(2), tmp_path / "local.sqlite3", "readout_d", _encoder=Encoder())


def test_failed_decision_is_counted_and_not_automatically_retried(tmp_path):
    class Broken(Encoder):
        def encode(self, texts):
            self.calls.append(texts)
            raise RuntimeError("unit encode failure")
    bank = Bank.initial("readout_c")
    encoder = Broken()
    pool = EmbeddingRouterPool(settings(), tmp_path / "local.sqlite3", bank.branch_id, _encoder=encoder)
    with pytest.raises(ValueError, match="no retry"):
        decision(pool, bank)
    with pytest.raises(ProtocolError, match="reconciliation"):
        decision(pool, bank)
    assert len(encoder.calls) == 1
    totals = local_totals(pool.ledger.path)
    assert totals["failed_attempts"] == totals["local_attempts"] == 1
    assert totals["incomplete_usage_attempts"] == 1


def test_mutating_bound_bank_object_is_rejected_before_encoding(tmp_path):
    bank = Bank.initial("readout_d")
    pool = EmbeddingRouterPool(settings(), tmp_path / "local.sqlite3", bank.branch_id, _encoder=Encoder())
    router = pool.for_bank(bank)
    bank.event = "illegal-mutation"
    with pytest.raises(ProtocolError, match="mutated"):
        router.route(router.memory.retrieve(""), **STATE)
    assert pool.encoder.calls == []


def test_initial_phase3_and_phase12_have_identical_encoding_and_ranking(tmp_path):
    from agent_system.memory.frozen_skill_bank import FrozenSkillBankMemory, load_skillnet37
    from agent_system.memory.skillrl_embedding_router import SkillRLEmbeddingStepRouter, load_profile
    frozen = FrozenSkillBankMemory(load_skillnet37())
    config, files = load_profile(DEFAULT_EMBEDDING_ROUTER_PROFILE)
    encoder = Encoder()
    phase12 = SkillRLEmbeddingStepRouter(frozen, config, model_files=files, model_path='/not-loaded',
        device='cpu', cache_path=tmp_path/'p12.sqlite3', max_local_calls=1, _encoder=encoder)
    a = phase12.route(frozen.retrieve(''), **STATE)
    bank = Bank.initial('readout_d')
    pool = EmbeddingRouterPool(settings(), tmp_path/'p3.sqlite3', bank.branch_id, _encoder=Encoder())
    b = decision(pool, bank)
    assert pool.encoder.calls == encoder.calls
    assert a['skill_router_scores'] == b['skill_router_scores']
    assert a['selected_skill_id'] == b['selected_skill_id']
    assert phase12.protocol['skill_texts'] == pool.for_bank(bank).protocol['skill_texts']


@pytest.mark.parametrize("branch", ARMS)
def test_all_six_branches_use_same_encoder_query_and_top_one(tmp_path, monkeypatch, branch):
    from test_runtime import runtime
    import phase3.training as training
    cfg = runtime()
    cfg["router"] = settings()
    validate_runtime(cfg)
    monkeypatch.setattr(training, "load", lambda _: ({}, cfg))
    prep = tmp_path / "assets/manifest.json"
    write_new(prep.parent / "split.json", {"evidence_seen": [{"game_id": "json_2.1.1/valid_seen/unit"}]})
    bank = Bank.initial(branch)
    path = bank.save(tmp_path / "banks")
    training_cfg = training.configuration(prep, tmp_path / "run", branch, path, bank.manifest_sha256, 0)
    routing = training_cfg.env.skills_only_memory.step_routing
    assert routing.backend == "phase3_skillrl_embedding_state"
    assert "max_api_calls" not in routing and routing.cache_path.endswith("router-local.sqlite3")
    pool = EmbeddingRouterPool(settings(), tmp_path / "local.sqlite3", branch, _encoder=Encoder())
    result = decision(pool, bank)
    assert len(result["injected_skill_ids"]) == 1 and result["skill_router_api"]["api_calls_this_step"] == 0
    router = pool.for_bank(bank)
    assert router.protocol["selection_count"] == 1
    assert router.config.model == "Qwen/Qwen3-Embedding-0.6B"
    assert set(json.loads(pool.encoder.calls[-1][0])) == set(STATE)


@pytest.mark.parametrize("change", ["api", "profile", "device", "limit"])
def test_new_runtime_rejects_mixed_or_unfrozen_router(change):
    cfg = settings()
    if change == "api":
        cfg["max_api_calls"] = 1
    elif change == "profile":
        cfg["profile_sha256"] = "0" * 64
    elif change == "device":
        cfg["device"] = "auto"
    else:
        cfg["max_local_calls"] = None
    with pytest.raises(ValueError):
        validate_settings(cfg)


def test_phase3_batched_router_deduplicates_and_preserves_branch_accounting(tmp_path):
    bank = Bank.initial("readout_d")
    encoder = BatchEncoder()
    pool = EmbeddingRouterPool(batched_settings(), tmp_path / "local.sqlite3", bank.branch_id,
                               _encoder=encoder)
    router = pool.for_bank(bank)
    request = {"candidate_bundle": router.memory.retrieve(""), **STATE}
    rows = router.route_many([request, request, {**request, "step_index": 1}])
    assert [r["selected_skill_id"] for r in rows] == [rows[0]["selected_skill_id"]] * 3
    assert [r["skill_router_api"]["local_calls_this_step"] for r in rows] == [1, 0, 1]
    assert [len(call) for call in encoder.calls] == [37, 2]
    assert rows[0]["skill_router_api"]["backend"] == "phase3_skillrl_embedding_state_batch"
    assert local_totals(pool.ledger.path)["local_attempts"] == 2
    assert local_totals(pool.ledger.path)["cache_hits"] == 1
    again = router.route_many([request, {**request, "step_index": 1}])
    assert all(r["skill_router_api"]["cache_hit"] for r in again)
    assert len(encoder.calls) == 2


def test_phase3_batched_router_reindexes_after_edit_and_has_separate_protocol(tmp_path):
    bank = Bank.initial("readout_d")
    pool = EmbeddingRouterPool(batched_settings(), tmp_path / "batched.sqlite3", bank.branch_id,
                               _encoder=BatchEncoder())
    first = decision(pool, bank)
    changed, _ = bank.apply(proposal("ADD", bank), event_id="u5", evidence_ids={"e"})
    second = decision(pool, changed)
    assert len(second["candidate_skill_ids"]) == 38
    assert first["skill_router_api"]["protocol_hash"] != second["skill_router_api"]["protocol_hash"]
    assert local_totals(pool.ledger.path)["bank_versions"] == 2
    old_pool = EmbeddingRouterPool(settings(), tmp_path / "old.sqlite3", bank.branch_id, _encoder=Encoder())
    assert old_pool.for_bank(bank).protocol_hash != pool.for_bank(bank).protocol_hash


def test_phase3_batched_budget_is_atomic_and_no_partial_encoder_call(tmp_path):
    bank = Bank.initial("readout_d")
    encoder = BatchEncoder()
    pool = EmbeddingRouterPool(batched_settings(limit=1), tmp_path / "local.sqlite3", bank.branch_id,
                               _encoder=encoder)
    router = pool.for_bank(bank)
    request = {"candidate_bundle": router.memory.retrieve(""), **STATE}
    with pytest.raises(RouterBudgetExceeded):
        router.route_many([request, {**request, "step_index": 1}])
    assert encoder.calls == []
    assert local_totals(pool.ledger.path)["local_attempts"] == 0


def test_phase3_batched_accepts_omegaconf_training_mapping(tmp_path):
    from omegaconf import OmegaConf
    bank = Bank.initial("readout_d")
    cfg = OmegaConf.create(batched_settings())
    validate_settings(cfg)
    pool = EmbeddingRouterPool(cfg, tmp_path / "local.sqlite3", bank.branch_id,
                               _encoder=BatchEncoder())
    assert pool.settings["execution"] == dict(cfg.execution)
    assert decision(pool, bank)["skill_router_api"]["backend"] == "phase3_skillrl_embedding_state_batch"


def test_gpu_sidecar_message_and_physical_gpu_selection(monkeypatch):
    import io
    from phase3.gpu_encoder_service import physical_gpu, receive, send
    stream = io.BytesIO()
    send(stream, {"operation": "encode", "texts": ["first", "second"]})
    stream.seek(0)
    assert receive(stream) == {"operation": "encode", "texts": ["first", "second"]}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert physical_gpu(0) == 0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert physical_gpu(0) == 3
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    assert physical_gpu(0) == 0


def test_training_and_validation_share_one_gpu_encoder_sidecar(tmp_path, monkeypatch):
    from phase3 import gpu_encoder_service
    created = []

    class FakeProxy:
        def __init__(self, **kwargs):
            created.append(kwargs)
        def close(self):
            pass

    monkeypatch.setattr(gpu_encoder_service, 'GPUEncoderProxy', FakeProxy)
    monkeypatch.setattr(gpu_encoder_service, '_SHARED_ENCODERS', {})
    config = batched_settings()
    ledger = tmp_path / 'branch.sqlite3'
    train = EmbeddingRouterPool(config, ledger, 'readout_d')
    validation = EmbeddingRouterPool(config, ledger, 'readout_d')
    assert train.encoder is validation.encoder
    assert len(created) == 1
    other = EmbeddingRouterPool(config, tmp_path / 'other.sqlite3', 'readout_d')
    assert other.encoder is not train.encoder
    assert len(created) == 2


def test_gpu_batch_requires_isolated_transport():
    cfg = batched_settings()
    del cfg["execution"]["transport"]
    with pytest.raises(ProtocolError, match="execution settings"):
        validate_settings(cfg)


def test_gpu_microbatch_has_new_protocol_and_keeps_outer_batch(tmp_path):
    from agent_system.memory.skillrl_embedding_batch_router import BatchedSentenceEncoder
    from agent_system.memory.skillrl_embedding_router import load_profile
    cfg = batched_settings()
    cfg["execution"].update(mode="state_batch_fp32_micro_v2", forward_microbatch_size=2)
    validate_settings(cfg)
    bank = Bank.initial("readout_d")
    pool = EmbeddingRouterPool(cfg, tmp_path / "micro.sqlite3", bank.branch_id,
                               _encoder=BatchEncoder())
    old = EmbeddingRouterPool(batched_settings(), tmp_path / "old.sqlite3", bank.branch_id,
                              _encoder=BatchEncoder())
    assert pool.for_bank(bank).protocol_hash != old.for_bank(bank).protocol_hash
    assert pool.for_bank(bank).protocol["version"].endswith("v3")
    assert decision(pool, bank)["skill_router_api"]["local_calls_this_step"] == 1
    profile, files = load_profile(DEFAULT_EMBEDDING_ROUTER_PROFILE)
    encoder = BatchedSentenceEncoder(profile, "/not-loaded", files, "cuda:0",
                                     intra_op_threads=1, forward_microbatch_size=2)
    assert profile.encode_batch_size == 8 and encoder._forward_batch_size() == 2


@pytest.mark.parametrize("size", [0, 9, True, "2"])
def test_gpu_microbatch_rejects_invalid_forward_size(size):
    cfg = batched_settings()
    cfg["execution"].update(mode="state_batch_fp32_micro_v2", forward_microbatch_size=size)
    with pytest.raises(ProtocolError, match="execution settings"):
        validate_settings(cfg)
