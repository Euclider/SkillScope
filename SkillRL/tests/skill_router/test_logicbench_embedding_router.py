"""Question-only routing against the frozen 19-skill bank."""

import json

import numpy as np
import pytest

from agent_system.memory.frozen_skill_bank import FrozenSkillBankMemory, load_skillnet37
from agent_system.memory.skillrl_embedding_router import EmbeddingRouterConfig, EmbeddingRouterError
from agent_system.memory.sra_logicbench_bank import load_sra_logicbench19


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        if len(texts) == 19:
            vectors[np.arange(19), np.arange(19)] = 1.0
        else:
            for index, value in enumerate(texts):
                question = json.loads(value)["question"].lower()
                vectors[index, 1 if "not q" in question else 0] = 1.0
        return vectors, {
            "input_tokens": len(texts) * 7,
            "input_token_counts": [7] * len(texts),
            "texts": len(texts),
            "encode_latency_ms": float(len(texts)),
        }


def make_router(tmp_path, memory=None):
    from agent_system.memory.logicbench_embedding_router import LogicBenchEmbeddingRouter

    memory = memory or FrozenSkillBankMemory(load_sra_logicbench19())
    encoder = Encoder()
    router = LogicBenchEmbeddingRouter(
        memory, EmbeddingRouterConfig(), model_files={}, model_path="/unused",
        device="cpu", cache_path=tmp_path / "logicbench-router.sqlite3",
        max_local_calls=3,
        execution={"device": "cpu", "intra_op_threads": 1,
                   "request_order": "first_occurrence_deduplicated",
                   "batch_accounting": "equal_share_latency_exact_per_query_tokens"},
        _encoder=encoder,
    )
    return memory, router, encoder


def test_question_only_top1_and_replay_share_one_frozen_choice(tmp_path):
    memory, router, encoder = make_router(tmp_path)
    question = "If P then Q. Not Q. Does not P follow?"
    routed = router.route_question(question)
    assert routed["selected_skill_id"] == "logicbench_001"
    assert routed["candidate_skill_ids"] == list(memory.bank.skill_ids)
    assert routed["injected_skill_ids"] == ["logicbench_001"]
    assert memory.format_for_prompt(routed) == memory.bank.get("logicbench_001").payload
    assert [len(batch) for batch in encoder.calls] == [19, 1]
    assert json.loads(encoder.calls[-1][0]) == {"question": question}
    assert router.protocol["query_formatter"] == "logicbench-question-only-json-v1"
    replay = router.route_question(question)
    assert replay["selected_skill_id"] == routed["selected_skill_id"]
    assert replay["skill_router_api"]["cache_hit"] is True
    assert [len(batch) for batch in encoder.calls] == [19, 1]


def test_gold_fields_and_mutated_candidates_are_not_accepted(tmp_path):
    memory, router, _ = make_router(tmp_path)
    with pytest.raises(TypeError):
        router.route_question("If P then Q", answer="yes")
    candidate = memory.retrieve("")
    candidate["candidate_skill_ids"] = candidate["candidate_skill_ids"][:-1]
    with pytest.raises(Exception, match="candidate"):
        router.route_many([{"candidate_bundle": candidate, "question": "If P then Q"}])


def test_skillnet_bank_cannot_enter_logicbench_router(tmp_path):
    memory = FrozenSkillBankMemory(load_skillnet37())
    with pytest.raises(EmbeddingRouterError, match="LogicBench"):
        make_router(tmp_path, memory)
