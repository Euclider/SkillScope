"""Explicit opt-in factory shared by ALFWorld manager and readout callers."""

from pathlib import Path

from .external_skill_router import ExternalLLMSkillRouter, RouterConfig
from .frozen_skill_bank import FrozenSkillBankMemory, load_skillnet37


DEFAULT_ROUTER_PROFILE = Path(__file__).resolve().parents[2] / "configs/skillnet37_router_gpt54mini_v1.json"
DEFAULT_LOCAL_ROUTER_PROFILE = Path(__file__).resolve().parents[2] / "configs/skillnet37_router_qwen35_local_v1.json"
DEFAULT_EMBEDDING_ROUTER_PROFILE = Path(__file__).resolve().parents[2] / "configs/skillnet37_router_qwen3_embedding_v1.json"
DEFAULT_BATCH_EMBEDDING_ROUTER_PROFILE = Path(__file__).resolve().parents[2] / "configs/skillnet37_router_qwen3_embedding_batch_v1.json"


def create_skillnet37_runtime(*, cache_path, max_api_calls, profile_path=DEFAULT_ROUTER_PROFILE, client=None):
    """No request at construction. Credentials are read only on a cache miss."""
    memory = FrozenSkillBankMemory(load_skillnet37())
    config = RouterConfig.from_profile(profile_path)
    router = ExternalLLMSkillRouter(memory, config, cache_path=cache_path, max_api_calls=max_api_calls, client=client)
    return memory, router


def create_embedding_skillnet37_runtime(*, model_path, device, cache_path, max_local_calls,
                                       profile_path=DEFAULT_EMBEDDING_ROUTER_PROFILE):
    """Frozen official encoder, state-aware top-1 adaptation; no model load yet."""
    from .skillrl_embedding_router import SkillRLEmbeddingStepRouter, load_profile
    memory = FrozenSkillBankMemory(load_skillnet37())
    config, files = load_profile(profile_path)
    router = SkillRLEmbeddingStepRouter(memory, config, model_files=files, model_path=model_path,
        device=device, cache_path=cache_path, max_local_calls=max_local_calls)
    return memory, router


def create_local_skillnet37_runtime(*, model_path, device, cache_path, max_local_calls,
                                   profile_path=DEFAULT_LOCAL_ROUTER_PROFILE):
    """Shelved draft: the user selected SkillRL retrieval before validation."""
    raise RuntimeError("Local Qwen3.5 router proposal was superseded by the SkillRL retrieval decision; not enabled or validated")
    # Preserve the interrupted draft without activating or completing it.
    from .local_skill_router import LocalFrozenSkillRouter, load_profile
    memory = FrozenSkillBankMemory(load_skillnet37())
    config, files = load_profile(profile_path)
    router = LocalFrozenSkillRouter(memory, config, model_files=files, model_path=model_path,
                                   device=device, cache_path=cache_path, max_local_calls=max_local_calls)
    return memory, router


def create_batched_embedding_skillnet37_runtime(*, model_path, device, cache_path, max_local_calls,
                                               profile_path=DEFAULT_BATCH_EMBEDDING_ROUTER_PROFILE):
    from .skillrl_embedding_batch_router import BatchedEmbeddingStepRouter, load_batch_profile
    memory = FrozenSkillBankMemory(load_skillnet37())
    config, files, execution = load_batch_profile(profile_path)
    return memory, BatchedEmbeddingStepRouter(memory, config, model_files=files, model_path=model_path,
        device=device, cache_path=cache_path, max_local_calls=max_local_calls, execution=execution)
