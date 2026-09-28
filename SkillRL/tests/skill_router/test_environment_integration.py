"""CPU fake-environment integration; never initializes ALFWorld or a policy."""

import json
from types import SimpleNamespace as NS

import pytest
from omegaconf import OmegaConf

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.memory import FrozenStepSkillRouter
from agent_system.memory.external_skill_router import ExternalLLMSkillRouter, RouterConfig
from agent_system.memory.frozen_skill_bank import FrozenSkillBankMemory, load_skillnet37
from agent_system.memory import skillnet_runtime
from phase1.archive import archive_rollout_batch


class FakeEnvironment:
    def __init__(self, terminal=False):
        self.terminal = terminal
        self.get_admissible_commands = [["take mug 1 from countertop 1", "help"]]

    def reset(self):
        return ["You see mug 1.\nYour task is to: put a clean mug in cabinet 1"], None, [{"extra.gamefile": "pick_clean_then_place_in_recep/test/game.z8"}]

    def step(self, actions):
        self.get_admissible_commands = [["go to sinkbasin 1", "inventory"]]
        return ["You picked up mug 1."], None, [0.0], [self.terminal], [{"won": False}]


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    memory = FrozenSkillBankMemory(load_skillnet37())
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return NS(choices=[NS(finish_reason="stop", message=NS(content=json.dumps({"skill_id": memory.bank.skill_ids[0]})))],
                  model="gpt-5.4-mini", id="unit", usage=None)

    client = NS(chat=NS(completions=NS(create=create)))
    router = ExternalLLMSkillRouter(memory, RouterConfig(), cache_path=tmp_path / "integration.db", max_api_calls=3, client=client)
    monkeypatch.setattr(skillnet_runtime, "create_skillnet37_runtime", lambda **kwargs: (memory, router))
    return memory, router, calls


def config(tmp_path, history_length=2):
    return OmegaConf.create({"env": {
        "history_length": history_length, "use_skills_only_memory": True, "alfworld": {"action_only_prompt": True},
        "skills_only_memory": {"top_k": 1, "step_routing": {
            "enabled": True, "backend": "external_llm", "cache_path": str(tmp_path / "integration.db"), "max_api_calls": 3,
        }},
    }})


@pytest.mark.parametrize("history_length", [0, 2])
def test_training_manager_switches_to_external_bank_and_routes_every_live_step(runtime, tmp_path, history_length):
    memory, router, calls = runtime
    manager = AlfWorldEnvironmentManager(FakeEnvironment(), lambda acts, commands: (acts, [True]), config(tmp_path, history_length))
    observation, _ = manager.reset({})
    assert manager.external_skill_routing is True
    assert len(manager.retrieved_memories[0]["candidate_skill_ids"]) == 37
    assert memory.bank.skills[0].payload in observation["text"][0]
    next_observation, _, _, infos = manager.step(["take mug 1 from countertop 1"])
    assert len(calls) == 2
    assert memory.bank.skills[0].payload in next_observation["text"][0]
    assert infos[0]["skill_router_api"]["protocol_hash"] == router.protocol_hash
    state = json.loads(calls[1]["messages"][1]["content"])["state"]
    assert state["history"][0]["action"] == "take mug 1 from countertop 1"
    assert set(state["history"][0]) == {"observation", "action"}
    assert len(manager.prompt_skill_metadata[0]["candidate_skill_ids"]) == 37


def test_terminal_observation_does_not_trigger_an_unused_paid_selection(runtime, tmp_path):
    _, _, calls = runtime
    manager = AlfWorldEnvironmentManager(FakeEnvironment(terminal=True), lambda acts, commands: (acts, [True]), config(tmp_path))
    manager.reset({})
    _, _, done, infos = manager.step(["take mug 1 from countertop 1"])
    assert done[0]
    assert len(calls) == 1
    assert infos[0]["selected_skill_id"]
    assert manager.prompt_skill_metadata[0]["injected_skill_ids"] == []


def test_router_audit_survives_existing_trajectory_archive(runtime, tmp_path):
    _, router, _ = runtime
    manager = AlfWorldEnvironmentManager(FakeEnvironment(terminal=True), lambda acts, commands: (acts, [True]), config(tmp_path))
    manager.reset({})
    _, _, _, infos = manager.step(["take mug 1 from countertop 1"])
    assert archive_rollout_batch(
        output_dir=tmp_path, run_id="unit", split="train", global_step=1,
        total_batch_list=[[{"active_masks": True, "rewards": 0, "attention_mask": [1], "responses": [1]}]],
        total_infos=[[infos[0]]], episode_rewards=[0], episode_lengths=[1], trajectory_ids=["unit-trajectory"],
    ) == 1
    result = json.loads((tmp_path / "trajectories/unit/unit-trajectory.json").read_bytes())
    assert result["steps"][0]["skill_router_api"]["protocol_hash"] == router.protocol_hash
    assert result["steps"][0]["skill_router_api"]["api_calls_this_step"] == 1


def test_historical_phase_backend_remains_the_default(tmp_path):
    bank = tmp_path / "historical.json"
    bank.write_text(json.dumps({"general_skills": [{"skill_id": "gen_001", "title": "Explore", "principle": "Explore the room", "when_to_apply": "initial"}],
                                "task_specific_skills": {}, "common_mistakes": []}), encoding="utf-8")
    cfg = config(tmp_path)
    cfg.env.skills_only_memory = {"skills_json_path": str(bank), "step_routing": {"enabled": True}}
    manager = AlfWorldEnvironmentManager(FakeEnvironment(), lambda acts, commands: (acts, [True]), cfg)
    assert isinstance(manager.step_skill_router, FrozenStepSkillRouter)
    assert manager.external_skill_routing is False
    manager.reset({})
    assert "skill_router_api" not in manager.prompt_skill_metadata[0]


@pytest.mark.parametrize("change", ["disabled", "no_cache", "no_budget", "unknown_backend"])
def test_invalid_opt_in_configuration_fails_before_environment_reset(tmp_path, change):
    cfg = config(tmp_path)
    route = cfg.env.skills_only_memory.step_routing
    if change == "disabled":
        route.enabled = False
    elif change == "no_cache":
        del route.cache_path
    elif change == "no_budget":
        del route.max_api_calls
    else:
        route.backend = "unknown"
    with pytest.raises(ValueError):
        AlfWorldEnvironmentManager(FakeEnvironment(), lambda acts, commands: (acts, [True]), cfg)
