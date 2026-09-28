import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from agent_system.memory.external_skill_router import (
    API_KEY_ENV, ExternalLLMSkillRouter, ExternalRouterError, RouterConfig,
    RouterResponseError, SYSTEM_PROMPT,
)
from agent_system.memory.frozen_skill_bank import FrozenSkillBankMemory, load_skillnet37
from agent_system.memory.router_cache import RouterBudgetExceeded, RouterCacheError
from agent_system.memory.skillnet_runtime import DEFAULT_ROUTER_PROFILE, create_skillnet37_runtime
from phase1.first_invocation import PayloadArm, intervention_payload
from scripts.test_skillnet_router import SYNTHETIC_STATE, run_smoke


def response(skill_id, **changes):
    result = NS(
        choices=[NS(finish_reason="stop", message=NS(content=json.dumps({"skill_id": skill_id}), refusal=None, tool_calls=None))],
        model="gpt-5.4-mini", id="chatcmpl-unit", _request_id="req-unit", system_fingerprint="fp-unit",
        usage=NS(prompt_tokens=100, completion_tokens=20, total_tokens=120,
                 prompt_tokens_details=NS(cached_tokens=0), completion_tokens_details=NS(reasoning_tokens=0)),
    )
    result.__dict__.update(changes)
    return result


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture(scope="module")
def bank():
    return load_skillnet37()


@pytest.fixture
def setup(bank, tmp_path):
    memory = FrozenSkillBankMemory(bank)
    client = FakeClient(response(bank.skill_ids[0]))
    router = ExternalLLMSkillRouter(memory, RouterConfig(), cache_path=tmp_path / "router.sqlite3", max_api_calls=3, client=client)
    return memory, router, client


def route(memory, router, **changes):
    state = copy.deepcopy(SYNTHETIC_STATE)
    state.update(changes)
    return router.route(memory.retrieve(state["task_description"]), **state)


def test_profile_and_factory_are_explicit_and_do_not_call_api(tmp_path):
    config = RouterConfig.from_profile(DEFAULT_ROUTER_PROFILE)
    assert config.model == "gpt-5.4-mini"
    assert config.sdk_version == "3.14.1"
    assert config.base_url == "https://api.zhizengzeng.com/v1"
    memory, router = create_skillnet37_runtime(cache_path=tmp_path / "cache.sqlite3", max_api_calls=0)
    assert len(memory) == 37
    assert router.cache.stats()["api_attempts"] == 0


def test_request_uses_full_canonical_catalog_and_only_visible_input(setup):
    memory, router, client = setup
    result = route(memory, router)
    request = client.calls[0]
    assert request["model"] == "gpt-5.4-mini"
    assert request["reasoning_effort"] == "none"
    assert request["temperature"] == 0
    assert request["max_completion_tokens"] == 128
    assert request["store"] is False
    assert request["n"] == 1
    assert request["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    data = json.loads(request["messages"][1]["content"])
    assert data["candidates"] == memory.bank.router_catalog()
    assert set(data["state"]) == {"task_description", "current_observation", "admissible_actions", "history", "step_index"}
    schema = request["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["skill_id"]["enum"] == list(memory.bank.skill_ids)
    assert schema["additionalProperties"] is False
    assert result["candidate_skill_ids"] == list(memory.bank.skill_ids)
    assert memory.format_for_prompt(result) == memory.bank.skills[0].payload
    assert result["skill_router_scores"] == {}  # Never invent LLM relevance scores.


def test_persistent_cache_replay_needs_neither_a_key_nor_another_call(setup, monkeypatch):
    memory, router, client = setup
    first = route(memory, router)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    replay = ExternalLLMSkillRouter(memory, RouterConfig(), cache_path=router.cache.path, max_api_calls=0)
    second = route(memory, replay)
    assert len(client.calls) == 1
    assert first["selected_skill_id"] == second["selected_skill_id"]
    assert second["skill_router_api"]["cache_hit"] is True
    assert second["skill_router_api"]["usage"]["total_tokens"] == 0
    assert second["skill_router_api"]["source_decision_usage"]["total_tokens"] == 120
    assert second["skill_router_api"]["latency_ms"] == 0
    assert second["skill_router_api"]["source_decision_latency_ms"] == first["skill_router_api"]["latency_ms"]
    assert replay.cache.stats()["api_attempts"] == 1


def test_history_whitelist_and_equivalent_training_eval_keys_share_cache(setup):
    memory, router, client = setup
    histories = copy.deepcopy(SYNTHETIC_STATE["history"])
    first = route(memory, router, history=histories)
    alternate = [{"text_obs": row["observation"], "action": row["action"], "gold": "hidden", "payload_arm": "null", "policy_logits": "hidden"} for row in histories]
    second = route(memory, router, history=[{"observation": "discarded", "action": "old"}] + alternate)
    assert first["selected_skill_id"] == second["selected_skill_id"]
    assert len(client.calls) == 1
    assert "hidden" not in json.dumps(client.calls)


def test_checkpoint_and_payload_arm_do_not_enter_the_router_cache_key(setup):
    memory, router, client = setup
    for arm in PayloadArm:
        candidates = memory.retrieve("goal")
        candidates.update(payload_arm=arm.value, checkpoint_id="checkpoint-" + arm.value, future_reward=10)
        result = router.route(candidates, **SYNTHETIC_STATE)
        expected = memory.bank.skills[0].payload if arm is PayloadArm.ORIGINAL else "neutral" if arm is PayloadArm.PLACEBO else ""
        assert intervention_payload(memory=memory, routed=result, arm=arm,
                                    target_skill_id=memory.bank.skill_ids[0], placebo_text="neutral")[0] == expected
    assert len(client.calls) == 1
    assert "checkpoint-" not in json.dumps(client.calls)


@pytest.mark.parametrize("mutation", ["candidate_ids", "description", "mask", "bank_hash", "preselected"])
def test_noncanonical_or_masked_candidates_fail_without_request(setup, mutation):
    memory, router, client = setup
    candidates = memory.retrieve("goal")
    if mutation == "candidate_ids":
        candidates["candidate_skill_ids"].pop()
    elif mutation == "description":
        candidates["general_skills"][0]["description"] = "changed"
    elif mutation == "mask":
        candidates["disabled_skill_ids"] = [memory.bank.skill_ids[0]]
    elif mutation == "bank_hash":
        candidates["bank_manifest_sha256"] = "bad"
    else:
        candidates = memory.selected_bundle(memory.bank.skill_ids[0])
    with pytest.raises(ExternalRouterError):
        router.route(candidates, **SYNTHETIC_STATE)
    assert client.calls == []


@pytest.mark.parametrize("content", [
    "not JSON", "[]", "null", '{}', '{"skill_id":null}', '{"skill_id":[]}',
    '{"skill_id":"missing"}', '{"skill_id":"missing","skill_id":"missing"}',
    '{"skill_id":"missing","rationale":"advice"}', '```json\n{}\n```',
])
def test_invalid_content_is_never_cached_or_injected(setup, content):
    memory, router, client = setup
    client.result.choices[0].message.content = content
    with pytest.raises(RouterResponseError):
        route(memory, router)
    assert len(client.calls) == 1
    assert router.cache.stats()["failed_attempts"] == 1
    assert router.cache.stats()["successful_decisions"] == 0


@pytest.mark.parametrize("problem", ["truncated", "refusal", "tool_call", "empty", "multiple", "model_drift"])
def test_response_protocol_violations_fail_closed(setup, problem):
    memory, router, client = setup
    if problem == "truncated":
        client.result.choices[0].finish_reason = "length"
    elif problem == "refusal":
        client.result.choices[0].message.refusal = "refusal"
    elif problem == "tool_call":
        client.result.choices[0].message.tool_calls = ["unexpected"]
    elif problem == "empty":
        client.result.choices = []
    elif problem == "multiple":
        client.result.choices *= 2
    else:
        router = ExternalLLMSkillRouter(memory, replace(router.config, expected_response_model="a-specific-snapshot"),
                                       cache_path=router.cache.path.with_name("drift.db"), max_api_calls=1, client=client)
    with pytest.raises(RouterResponseError):
        route(memory, router)
    assert router.cache.stats()["successful_decisions"] == 0


def test_router_configuration_cannot_be_replaced_after_protocol_is_locked(setup):
    _, router, _ = setup
    with pytest.raises(AttributeError):
        router.config = replace(router.config, history_length=3)


def test_exception_text_and_secrets_do_not_enter_logs(setup):
    memory, router, client = setup
    private_text = "private credential content must never be logged"
    client.result = RuntimeError(private_text)
    with pytest.raises(ExternalRouterError) as error:
        route(memory, router)
    assert private_text not in str(error.value)
    with sqlite3.connect(router.cache.path) as connection:
        dumped = "\n".join(connection.iterdump())
    assert private_text not in dumped
    assert len(client.calls) == 1


def test_shared_budget_includes_failed_calls_and_hits_are_free(setup):
    memory, router, client = setup
    router.cache.max_api_calls = 1
    route(memory, router)
    route(memory, router)
    with pytest.raises(RouterBudgetExceeded):
        route(memory, router, current_observation="a different state")
    assert len(client.calls) == 1


def test_concurrent_same_input_causes_one_model_call(setup):
    memory, router, client = setup
    other = ExternalLLMSkillRouter(memory, RouterConfig(), cache_path=router.cache.path, max_api_calls=1, client=client)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda item: route(memory, item), [router, other]))
    assert len(client.calls) == 1
    assert sorted(item["skill_router_api"]["cache_hit"] for item in results) == [False, True]


def test_changed_router_protocol_cannot_reuse_database(setup):
    memory, router, client = setup
    with pytest.raises(RouterCacheError, match="protocol mismatch"):
        ExternalLLMSkillRouter(memory, replace(RouterConfig(), history_length=3), cache_path=router.cache.path, max_api_calls=1, client=client)


def test_corrupted_cache_is_not_silently_replaced_by_new_model_output(setup):
    memory, router, client = setup
    route(memory, router)
    with sqlite3.connect(router.cache.path) as connection:
        connection.execute("UPDATE decisions SET hash=?", ("0" * 64,))
    with pytest.raises(RouterCacheError, match="integrity"):
        route(memory, router)
    assert len(client.calls) == 1


def test_input_limit_fails_without_silent_truncation(setup):
    memory, router, client = setup
    with pytest.raises(ExternalRouterError, match="byte cap"):
        route(memory, router, current_observation="x" * 70000)
    assert client.calls == []


@pytest.mark.parametrize("changes", [
    {"base_url": "http://api.zhizengzeng.com/v1"},
    {"base_url": "https://user:password@api.zhizengzeng.com/v1"},
    {"base_url": "https://api.zhizengzeng.com"},
    {"model": "gpt-4o-mini"}, {"reasoning_effort": "high"}, {"temperature": 1},
    {"history_length": -1}, {"max_completion_tokens": 0}, {"timeout_seconds": 61},
])
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ExternalRouterError):
        RouterConfig(**changes)


def test_zero_history_really_excludes_all_history(bank, tmp_path):
    memory = FrozenSkillBankMemory(bank)
    client = FakeClient(response(bank.skill_ids[0]))
    router = ExternalLLMSkillRouter(memory, replace(RouterConfig(), history_length=0), cache_path=tmp_path / "zero.db", max_api_calls=1, client=client)
    route(memory, router)
    assert json.loads(client.calls[0]["messages"][1]["content"])["state"]["history"] == []


def test_live_smoke_logic_with_mock_does_one_request_and_one_replay(setup):
    memory, router, client = setup
    result = run_smoke(memory, router)
    assert result["candidate_count"] == 37
    assert result["live_api_calls_this_invocation"] == 1
    assert result["replay_cache_hit"] is True
    assert len(client.calls) == 1


def test_missing_dedicated_key_never_uses_shared_openai_key(bank, tmp_path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-test-credential")
    monkeypatch.setattr("importlib.metadata.version", lambda package: "3.14.1")
    memory = FrozenSkillBankMemory(bank)
    router = ExternalLLMSkillRouter(memory, RouterConfig(), cache_path=tmp_path / "missing.db", max_api_calls=1)
    with pytest.raises(ExternalRouterError, match=API_KEY_ENV):
        route(memory, router)
    assert router.cache.stats()["api_attempts"] == 0


def test_current_sdk_serializes_request_and_parses_response_without_network(bank, tmp_path):
    import openai
    import httpx2

    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        assert str(request.url) == "https://api.zhizengzeng.com/v1/chat/completions"
        return httpx2.Response(200, headers={"x-request-id": "req-mock"}, json={
            "id": "chatcmpl-mock", "object": "chat.completion", "created": 1, "model": "gpt-5.4-mini",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps({"skill_id": bank.skill_ids[0]})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        })

    client = openai.OpenAI(base_url="https://api.zhizengzeng.com/v1", api_key="unit-test-not-a-real-key", max_retries=0,
                           http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(handler), follow_redirects=False))
    try:
        memory = FrozenSkillBankMemory(bank)
        router = ExternalLLMSkillRouter(memory, RouterConfig(), cache_path=tmp_path / "sdk.db", max_api_calls=1, client=client)
        result = route(memory, router)
        assert captured[0]["model"] == "gpt-5.4-mini"
        assert result["skill_router_api"]["request_id"] == "req-mock"
    finally:
        client.close()
