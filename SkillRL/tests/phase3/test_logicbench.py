"""LogicBench revision boundary: real bank transactions, offline model/API doubles."""
from types import SimpleNamespace

import numpy as np
import pytest

from agent_system.memory.sra_logicbench_bank import load_sra_logicbench19
from agent_system.memory.skillnet_runtime import DEFAULT_EMBEDDING_ROUTER_PROFILE
from phase1.logicbench_single_step import LogicBenchQuestion, GenerationOutput
from phase3.common import ProtocolError
from phase3.logicbench import (initial_bank, rank_skills, editor_input, propose,
                              evaluate_bank, revise_once, validate_split, SkillPromptBudgetError)
from phase3.logicbench_routing import LogicBenchRouterPool
from skillnet_cohort.common import file_hash


def evidence(bank):
    return [{"evidence_id": f"e{i}", "question_id": f"q{i}", "context_id": f"c{i}",
             "split": "train", "sampling_policy_update": 0,
             "selected_skill_id": bank.skill_ids[i % 2],
             "skill_version_sha256": bank.skills[i % 2].version_sha256,
             "question": "Does the conclusion follow?", "task_type": "BQA",
             "response": "yes", "success": i % 2 == 0,
             "forbidden_eval_label": 999} for i in range(6)]


def patch(bank, kind="MODIFY"):
    targets = bank.skills[:2] if kind == "MERGE" else bank.skills[:1] if kind in ("MODIFY", "DELETE") else []
    return {"operations": [{"op": kind, "targets": [
        {"skill_id": s.skill_id, "version_sha256": s.version_sha256} for s in targets],
        "skill": {"name": "Revised rule", "description": "Reusable logical inference",
                  "body": "Check the antecedent before applying implication."} if kind not in ("NOOP", "DELETE") else None,
        "rationale": "Observed training evidence", "evidence_ids": ["e0"]}]}


class Encoder:
    def encode(self, texts):
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0 if text.startswith('{') else i] = 1
        return vectors, {"input_tokens": 10 * len(texts), "texts": len(texts),
                         "encode_latency_ms": 1., "input_token_counts": [10] * len(texts)}

    def close(self):
        pass


def pool(tmp_path, bank):
    settings = {"backend": "skillrl_embedding_state", "model_path": "/not-loaded",
                "device": "cpu", "profile_sha256": file_hash(DEFAULT_EMBEDDING_ROUTER_PROFILE),
                "max_local_calls": 100,
                "execution": {"mode": "state_batch_fp32_v1", "intra_op_threads": 1}}
    return LogicBenchRouterPool(settings, tmp_path / "router.sqlite3", bank.branch_id, _encoder=Encoder())


@pytest.mark.parametrize("kind,size", [("ADD", 20), ("MODIFY", 19), ("DELETE", 18), ("MERGE", 18)])
def test_logic_edits_reindex_full_bank_without_alf_state(tmp_path, kind, size):
    bank = initial_bank("test")
    frozen = load_sra_logicbench19()
    assert [s.payload for s in bank.skills] == [s.payload for s in frozen.skills]
    provider = pool(tmp_path, bank)
    first = provider.for_bank(bank).route_question("A implies B; A. Does B follow?")
    changed, _ = bank.apply(patch(bank, kind), event_id="u5", evidence_ids={"e0"})
    second = provider.for_bank(changed).route_question("A implies B; A. Does B follow?")
    assert len(second["candidate_skill_ids"]) == size
    assert first["skill_router_api"]["protocol_hash"] != second["skill_router_api"]["protocol_hash"]
    assert second["skill_version_sha256"] == changed.get(second["selected_skill_id"]).version_sha256
    assert provider.for_bank(bank).route_question("A implies B; A. Does B follow?")["skill_router_api"]["cache_hit"]
    assert provider.for_bank(bank).protocol["query_formatter"] == "logicbench-question-only-json-v1"


def test_same_invoked_pool_preserves_zero_direction_and_low_support():
    bank = initial_bank("test")
    rows = evidence(bank)
    scores = [{"skill_id": bank.skill_ids[0], "D_signed_gate": -2., "M_delta_raw": 20.},
              {"skill_id": bank.skill_ids[1], "D_signed_gate": 0., "M_delta_raw": 2.}]
    assert rank_skills(bank, rows, scores, "D_signed_gate", k=2) == [bank.skill_ids[1], bank.skill_ids[0]]
    assert rank_skills(bank, rows, scores, "M_delta_raw", k=2) == list(bank.skill_ids[:2])
    assert rank_skills(bank, rows, scores, "failure_rate", k=2) == [bank.skill_ids[1], bank.skill_ids[0]]
    with pytest.raises(ProtocolError, match="coverage"):
        rank_skills(bank, rows, scores[:1], "D_signed_gate", k=2)


def test_editor_budget_blinding_and_success_evidence_are_shared():
    bank = initial_bank("test")
    payload = editor_input(bank, evidence(bank), bank.skill_ids[:2], evidence_per_skill=2, mutation_units=3)
    assert len(payload["evidence"]) == 4
    assert {r["selected_skill_id"] for r in payload["evidence"]} == set(bank.skill_ids[:2])
    assert any(r["success"] for r in payload["evidence"])
    assert "forbidden_eval_label" not in str(payload)
    assert "D_signed_gate" not in str(payload)
    assert "ALFWorld" not in str(payload)
    assert all("steps" not in r for r in payload["evidence"])
    bad = evidence(bank)
    bad[0]["split"] = "eval"
    with pytest.raises(ProtocolError):
        editor_input(bank, bad, bank.skill_ids[:2], evidence_per_skill=2, mutation_units=3)


class Editor:
    config = SimpleNamespace(stage="editor", max_input_tokens=100000)

    def __init__(self, value):
        self.value = value

    def request(self, *, validate, **kwargs):
        validate(self.value)
        return self.value, {"external_api_calls": 0, "offline_fixture": True}


def test_editor_cannot_target_outside_candidates_and_enforces_input_cap():
    bank = initial_bank("test")
    payload = editor_input(bank, evidence(bank), [bank.skill_ids[1]], evidence_per_skill=2, mutation_units=3)
    with pytest.raises(ProtocolError, match="unexposed"):
        propose(bank, api=Editor(patch(bank)), payload=payload, event_id="u5", token_counter=lambda _: 1)
    api = Editor(patch(bank, "NOOP"))
    with pytest.raises(ProtocolError, match="input"):
        propose(bank, api=api, payload=payload, event_id="u5", token_counter=lambda _: 100001)


def test_single_answer_evaluation_has_paired_rng_and_no_answer_in_prompt(tmp_path):
    bank = initial_bank("test")
    rows = [LogicBenchQuestion("qtest", "A implies B; A. Does B follow?", "BQA", "yes", "not_for_router")]
    seen = []

    def generate(prompt, seed):
        seen.append((prompt, seed))
        return GenerationOutput("yes", 2, False, 100)

    provider = pool(tmp_path, bank)
    before = evaluate_bank(bank, rows, provider, generate, seeds=(0, 1), rng_mode="question_seed_v1")
    changed, _ = bank.apply(patch(bank), event_id="u5", evidence_ids={"e0"})
    after = evaluate_bank(changed, rows, provider, generate, seeds=(0, 1), rng_mode="question_seed_v1")
    assert [r["rng_seed"] for r in before] == [r["rng_seed"] for r in after]
    assert all(r["success"] and r["format_valid"] for r in before + after)
    assert len(seen) == 4 and all("not_for_router" not in prompt for prompt, _ in seen)
    assert before[0]["bank_sha256"] != after[0]["bank_sha256"]
    with pytest.raises(ProtocolError):
        evaluate_bank(bank, rows, provider, generate, seeds=(0, 0), rng_mode="question_seed_v1")


@pytest.mark.parametrize("after_success,accepted", [(True, True), (False, False)])
def test_revision_gate_commits_only_nonregressing_candidate(tmp_path, after_success, accepted):
    bank = initial_bank("test")
    payload = editor_input(bank, evidence(bank), bank.skill_ids[:2], evidence_per_skill=2, mutation_units=3)

    def evaluate(current):
        return [{"game_id": "heldout", "eval_seed": 0, "success": True if current is bank else after_success,
                 "bank_sha256": current.manifest_sha256}]

    result = revise_once(bank, api=Editor(patch(bank)), payload=payload, evaluate=evaluate,
                         output=tmp_path, event_id="u5", tolerance_pp=0., gate_ids={"heldout"},
                         identity={"checkpoint": "fixed-u5"}, token_counter=lambda _: 1)
    assert result["accepted"] is accepted
    assert (result["bank"].manifest_sha256 != bank.manifest_sha256) is accepted
    assert (tmp_path / "complete.json").exists()


def test_context_leakage_and_invalid_gate_pair_are_rejected(tmp_path):
    with pytest.raises(ProtocolError, match="context"):
        validate_split([{"question_id": "q1", "context_id": "c"}],
                       [{"question_id": "q2", "context_id": "c"}])
    bank = initial_bank("test")
    payload = editor_input(bank, evidence(bank), bank.skill_ids[:2], evidence_per_skill=2, mutation_units=3)
    with pytest.raises(ProtocolError, match="gate"):
        revise_once(bank, api=Editor(patch(bank)), payload=payload,
                    evaluate=lambda b: [{"game_id": "wrong", "eval_seed": 0, "success": True,
                                         "bank_sha256": b.manifest_sha256}],
                    output=tmp_path, event_id="u5", tolerance_pp=0., gate_ids={"heldout"},
                    identity={"checkpoint": "fixed-u5"}, token_counter=lambda _: 1)


def test_interrupted_gate_resumes_persisted_proposal_without_editor_replay(tmp_path):
    bank = initial_bank("test")
    payload = editor_input(bank, evidence(bank), bank.skill_ids[:2], evidence_per_skill=2, mutation_units=3)
    kwargs = dict(payload=payload, output=tmp_path, event_id="u5", tolerance_pp=0.,
                  gate_ids={"heldout"}, identity={"checkpoint": "fixed-u5"}, token_counter=lambda _: 1)

    def interrupted(_):
        raise RuntimeError("interrupted gate")

    with pytest.raises(RuntimeError, match="interrupted gate"):
        revise_once(bank, api=Editor(patch(bank)), evaluate=interrupted, **kwargs)

    class NoReplay(Editor):
        def request(self, **kwargs):
            raise AssertionError("Persisted proposal should bypass editor API/cache accounting replay")

    result = revise_once(bank, api=NoReplay(None), evaluate=lambda b: [
        {"game_id": "heldout", "eval_seed": 0, "success": True, "bank_sha256": b.manifest_sha256}], **kwargs)
    assert result["accepted"]


def test_oversized_skill_is_rejected_without_generation(tmp_path):
    bank = initial_bank("test")
    payload = editor_input(bank, evidence(bank), bank.skill_ids[:2], evidence_per_skill=2, mutation_units=3)

    def budget(_):
        raise SkillPromptBudgetError("Edited prompt exceeds cap")

    def no_generation(_):
        raise AssertionError("Invalid candidate must not reach generation")

    result = revise_once(bank, api=Editor(patch(bank)), payload=payload, evaluate=no_generation,
        output=tmp_path, event_id="u5", tolerance_pp=0., gate_ids={"heldout"},
        identity={"checkpoint": "fixed-u5"}, token_counter=lambda _: 1, payload_validator=budget)
    assert not result["accepted"] and not result["noop"]
    assert result["bank"] is bank and result["budget_rejection"] == "Edited prompt exceeds cap"


def test_v13_uncapped_input_keeps_cost_estimate():
    bank = initial_bank('test')
    payload = editor_input(bank, evidence(bank), [bank.skill_ids[0]], evidence_per_skill=2, mutation_units=3)
    _, result = propose(bank,api=Editor(patch(bank,'NOOP')),payload=payload,event_id='u5',
        token_counter=lambda _:100001,enforce_input_cap=False)
    assert result['input_token_estimate'] == 100257
