"""Versioned branch router factories; legacy mini remains explicit, never fallback."""
from __future__ import annotations

from agent_system.memory.external_skill_router import ExternalLLMSkillRouter, RouterConfig, SYSTEM_PROMPT
from agent_system.memory.frozen_skill_bank import FrozenSkillBankMemory

from .common import require


class BranchMemory(FrozenSkillBankMemory):
    """Editing occurs transactionally through Bank.apply, not memory mutation."""
    def selected_bundle(self, skill_id):
        result = super().selected_bundle(skill_id)
        result["skill_version_sha256"] = self.bank.get(skill_id).version_sha256 if skill_id else None
        return result


class BranchRouter:
    version = "phase3-bank-versioned-mini-v1"

    def __init__(self, bank, api):
        require(api.config.stage == "router", "Wrong router API role")
        self.memory, self.api = BranchMemory(bank), api
        # Keep exactly the original prompt when the candidate count is 37.
        self.system = SYSTEM_PROMPT.replace("All 37 candidates", f"All {len(bank)} candidates")
        self.config = RouterConfig()

    def route(self, candidate_bundle, **visible):
        expected = self.memory.retrieve("")
        for key in ("bank_id", "bank_manifest_sha256", "bank_content_sha256", "candidate_skill_ids",
                    "general_skills", "task_specific_skills", "mistakes_to_avoid"):
            require(candidate_bundle.get(key) == expected[key], "Foreign or filtered branch catalog")
        require(not candidate_bundle.get("disabled_skill_ids"), "Router candidates cannot be masked")
        state = ExternalLLMSkillRouter._visible_input(self, **visible)
        bank = self.memory.bank
        schema = {"type": "object", "properties": {"skill_id": {"type": "string", "enum": list(bank.skill_ids)}},
                  "required": ["skill_id"], "additionalProperties": False}

        def validate(value):
            require(isinstance(value, dict) and set(value) == {"skill_id"} and value["skill_id"] in bank.skill_ids,
                    "Router returned an invalid skill ID")

        result, cost = self.api.request(identity={"router_version": self.version, "bank_sha256": bank.manifest_sha256},
                                        system=self.system, payload={"candidates": bank.router_catalog(), "state": state},
                                        schema=schema, validate=validate)
        bundle = self.memory.selected_bundle(result["skill_id"])
        bundle.update(skill_router_version=self.version, skill_router_api=cost,
                      skill_router_scores={}, skill_router_score_details={}, skill_router_state_flags=[],
                      skill_router_selection_reason="independent external LLM; complete current branch catalog")
        return bundle


def create_runtime(config):
    """Training opt-in; the full versioned catalog is never task-filtered."""
    from .api import APIConfig, JSONClient
    from .bank import Bank
    bank = Bank.load(config["bank_path"], config["bank_sha256"])
    if config.get("backend") == "phase3_skillrl_embedding_state":
        from .embedding_routing import EmbeddingRouterPool
        settings = {key: config[key] for key in ("model_path", "device", "profile_sha256", "max_local_calls")}
        settings["backend"] = "skillrl_embedding_state"
        if "execution" in config:
            settings["execution"] = config["execution"]
        router = EmbeddingRouterPool(settings, config["cache_path"], bank.branch_id).for_bank(bank)
        return router.memory, router
    api = JSONClient(APIConfig(stage="router", model="gpt-5.4-mini",
        max_input_tokens=int(config["max_input_tokens"]),
        max_completion_tokens=int(config["max_completion_tokens"]),
        max_api_calls=int(config["max_api_calls"])), config["cache_path"], allow_live=True)
    router = BranchRouter(bank, api)
    return router.memory, router


def router_backend(settings):
    backend = settings.get("backend", "external_llm")
    require(backend in ("external_llm", "skillrl_embedding_state"), "Unknown Phase3 router backend")
    return backend


def create_provider(settings, root, branch):
    if router_backend(settings) == "skillrl_embedding_state":
        from .embedding_routing import EmbeddingRouterPool
        return EmbeddingRouterPool(settings, root / "router-local.sqlite3", branch)
    from .api import APIConfig, JSONClient
    return JSONClient(APIConfig(stage="router", model="gpt-5.4-mini", **settings), root / "router.sqlite3", allow_live=True)
