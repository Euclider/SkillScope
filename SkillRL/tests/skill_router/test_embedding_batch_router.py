import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from agent_system.memory.frozen_skill_bank import FrozenSkillBankMemory, load_skillnet37
from agent_system.memory.router_cache import RouterBudgetExceeded, RouterCacheError
from agent_system.memory.skillnet_runtime import DEFAULT_BATCH_EMBEDDING_ROUTER_PROFILE
from agent_system.memory.skillrl_embedding_batch_router import BatchedEmbeddingStepRouter, load_batch_profile
from agent_system.memory.skillrl_embedding_router import EmbeddingRouterError


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        if len(texts) == 37:
            vectors[np.arange(37), np.arange(37)] = 1
        else:
            for i, text in enumerate(texts):
                vectors[i, json.loads(text)["step_index"] % 37] = 1
        return vectors, dict(input_tokens=len(texts) * 10, input_token_counts=[10] * len(texts),
                             texts=len(texts), encode_latency_ms=float(len(texts)))


def runtime(tmp_path, budget=20):
    memory = FrozenSkillBankMemory(load_skillnet37())
    config, files, execution = load_batch_profile(DEFAULT_BATCH_EMBEDDING_ROUTER_PROFILE)
    encoder = Encoder()
    router = BatchedEmbeddingStepRouter(memory, config, model_files=files, model_path="/not-loaded",
        device="cpu", cache_path=tmp_path / 'cache.sqlite3', max_local_calls=budget,
        execution=execution, _encoder=encoder)
    return memory, router, encoder


def request(memory, step=0):
    return dict(candidate_bundle=memory.retrieve(""), task_description="put a mug on table",
        current_observation="A mug is visible.", admissible_actions=["look"], history=[], step_index=step)


def test_deduplicates_batch_preserves_order_and_exact_accounting(tmp_path):
    memory, router, encoder = runtime(tmp_path)
    rows = router.route_many([request(memory, s) for s in (1, 0, 1, 2, 0)])
    assert [r['selected_skill_id'] for r in rows] == [memory.bank.skill_ids[s] for s in (1, 0, 1, 2, 0)]
    assert [len(x) for x in encoder.calls] == [37, 3]
    assert [r['skill_router_api']['cache_hit'] for r in rows] == [False, False, True, False, True]
    assert sum(r['skill_router_api']['usage']['total_tokens'] for r in rows) == 400
    assert sum(r['skill_router_api']['local_calls_this_step'] for r in rows) == 3
    assert all(len(r['skill_router_scores']) == 37 for r in rows)
    assert router.stats()['successful_decisions'] == 3
    assert router.stats()['cache_hits'] == 2
    replay = router.route_many([request(memory, s) for s in (0, 1, 2)])
    assert len(encoder.calls) == 2
    assert all(r['skill_router_api']['usage']['total_tokens'] == 0 for r in replay)


def test_batch_budget_admission_is_atomic(tmp_path):
    memory, router, encoder = runtime(tmp_path, budget=1)
    with pytest.raises(RouterBudgetExceeded):
        router.route_many([request(memory, 0), request(memory, 1)])
    assert encoder.calls == [] and router.stats()['local_attempts'] == 0


def test_failed_batch_cannot_retry(tmp_path, monkeypatch):
    memory, router, encoder = runtime(tmp_path)
    def fail(texts):
        raise RuntimeError('injected failure')
    monkeypatch.setattr(encoder, 'encode', fail)
    with pytest.raises(EmbeddingRouterError, match='no retry'):
        router.route_many([request(memory, 0), request(memory, 1)])
    assert router.stats()['failed_attempts'] == 2
    with pytest.raises(RouterCacheError, match='no automatic retry'):
        router.route_many([request(memory, 0)])
    assert router.stats()['local_attempts'] == 2


def test_chunking_and_single_query_route_use_same_contract(tmp_path):
    memory, router, encoder = runtime(tmp_path)
    rows = router.route_many([request(memory, s) for s in range(17)])
    assert [len(x) for x in encoder.calls] == [37, 8, 8, 1]
    assert sum(r['skill_router_api']['usage']['total_tokens'] for r in rows) == 540
    state = request(memory, 0)
    bundle = state.pop('candidate_bundle')
    assert router.route(bundle, **state)['selected_skill_id'] == rows[0]['selected_skill_id']
    assert router.route_many([]) == []


def test_v1_cache_is_not_accepted_and_invalid_profile_rejected(tmp_path):
    from agent_system.memory.skillnet_runtime import create_embedding_skillnet37_runtime
    memory, v1 = create_embedding_skillnet37_runtime(model_path='/not-loaded', device='cpu',
        cache_path=tmp_path / 'cache.sqlite3', max_local_calls=0)
    with pytest.raises(RouterCacheError, match='protocol mismatch'):
        runtime(tmp_path)
    profile = json.loads(DEFAULT_BATCH_EMBEDDING_ROUTER_PROFILE.read_bytes())
    profile['execution']['intra_op_threads'] = 0
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(profile))
    with pytest.raises(EmbeddingRouterError):
        load_batch_profile(path)
