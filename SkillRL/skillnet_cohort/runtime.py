"""Shared bank identity, token bounds, reproducible RNG, and resource gates."""
from __future__ import annotations

import os
import random
import re
import shutil
import stat
from pathlib import Path

from .common import REPO, file_hash, read_json


ROUTER_PROFILES = {
    "external_llm": REPO / "configs/skillnet37_router_gpt54mini_v1.json",
    "skillrl_embedding_state": REPO / "configs/skillnet37_router_qwen3_embedding_v1.json",
    "skillrl_embedding_state_batch": REPO / "configs/skillnet37_router_qwen3_embedding_batch_v1.json",
}


def is_embedding_backend(backend):
    return backend in ("skillrl_embedding_state", "skillrl_embedding_state_batch")


def embedding_profile(backend):
    if backend == "skillrl_embedding_state_batch":
        from agent_system.memory.skillrl_embedding_batch_router import load_batch_profile
        config, files, _ = load_batch_profile(ROUTER_PROFILES[backend])
        return config, files
    from agent_system.memory.skillrl_embedding_router import load_profile
    return load_profile(ROUTER_PROFILES[backend])


def router_backend(settings):
    # Old manifests remain byte-for-byte valid; new ones must opt in explicitly.
    backend = settings.get("router_backend", "external_llm")
    if backend not in ROUTER_PROFILES:
        raise ValueError("Unknown cohort router backend; no automatic fallback")
    return backend


def router_registration(backend, *, model_path=None, device=None):
    backend = router_backend({"router_backend": backend})
    result = {"router_backend": backend, "router_profile_sha256": file_hash(ROUTER_PROFILES[backend])}
    if is_embedding_backend(backend):
        import re
        if not model_path or not isinstance(device, str) or not (device == "cpu" or re.fullmatch(r"cuda:\d+", device)):
            raise ValueError("Embedding cohorts require an explicit model path and cpu/cuda:N placement")
        result.update(router_model_path=str(Path(model_path).resolve()), router_device=device)
        if backend == "skillrl_embedding_state_batch" and device != "cpu":
            raise ValueError("Batched FP32 embedding protocol requires CPU placement")
    elif model_path is not None or device is not None:
        raise ValueError("Local model settings cannot accompany an API router")
    return result


def seed_process(seed, rank=0):
    import numpy as np
    import torch
    actual = (int(seed) + 100000 * int(rank)) % (2**32)
    random.seed(actual)
    np.random.seed(actual)
    torch.manual_seed(actual)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(actual)
    return actual


def verify_runtime_identity(runtime):
    from .inference import validate
    validate(runtime)
    from agent_system.memory.frozen_skill_bank import load_skillnet37
    bank = load_skillnet37()
    if runtime.get("kind") != "skillnet37" or runtime["bank_manifest_sha256"] != bank.manifest_sha256:
        raise ValueError("Wrong frozen bank identity")
    backend = router_backend(runtime)
    profile = ROUTER_PROFILES[backend]
    if file_hash(profile) != runtime["router_profile_sha256"]:
        raise ValueError("Router request profile changed")
    if is_embedding_backend(backend):
        embedding_profile(backend)  # Validate identity/schema, but do not load weights.
        router_registration(backend, model_path=runtime.get("router_model_path"), device=runtime.get("router_device"))
        if runtime.get("max_api_calls", 0) != 0:
            raise ValueError("Embedding runtime cannot carry a paid API-call budget")
    return bank


def make_runtime(runtime, *, client=None):
    verify_runtime_identity(runtime)
    if is_embedding_backend(router_backend(runtime)):
        if client is not None:
            raise ValueError("An external API client cannot be supplied to embedding routing")
        from agent_system.memory.skillnet_runtime import create_embedding_skillnet37_runtime
        factory = create_embedding_skillnet37_runtime
        if router_backend(runtime) == "skillrl_embedding_state_batch":
            from agent_system.memory.skillnet_runtime import create_batched_embedding_skillnet37_runtime
            factory = create_batched_embedding_skillnet37_runtime
        return factory(model_path=runtime["router_model_path"],
            device=runtime["router_device"], cache_path=runtime["cache_path"],
            max_local_calls=runtime["max_local_calls"])
    from agent_system.memory.skillnet_runtime import create_skillnet37_runtime
    return create_skillnet37_runtime(cache_path=runtime["cache_path"],
                                    max_api_calls=runtime["max_api_calls"], client=client)


def runtime_settings(spec, cache_path, max_api_calls=0, *, max_local_calls=0):
    result = {"kind": "skillnet37", "bank_manifest_sha256": spec["bank_manifest_sha256"],
            "router_profile_sha256": spec["router_profile_sha256"],
            "cache_path": str(Path(cache_path).resolve()), "max_api_calls": int(max_api_calls)}
    if is_embedding_backend(router_backend(spec)):
        if max_api_calls != 0:
            raise ValueError("Embedding runtime requires zero external API calls")
        if type(max_local_calls) is not int or max_local_calls < 0:
            raise ValueError("Explicit nonnegative local router-call budget required")
        result.update({key: spec[key] for key in ("router_backend", "router_model_path", "router_device")})
        result["max_local_calls"] = max_local_calls
    elif max_local_calls != 0:
        raise ValueError("Local-call budget cannot authorize a paid router")
    if spec.get('inference_profile'):
        from copy import deepcopy
        from .inference import validate
        result['inference_profile'] = deepcopy(validate(spec))
    return result


def authorized_runtime(spec, cache_path, permit):
    """Same budget/backend contract for training, performance and O/P/N."""
    return runtime_settings(spec, cache_path, permit.get("router_max_api_calls", 0),
                            max_local_calls=permit.get("router_max_local_calls", 0))


class BoundedPolicy:
    """Applies the SAME nontruncating actor context limit to performance and O/P/N."""
    def __init__(self, policy, max_prompt_tokens=4096):
        self.policy = policy
        self.tokenizer = policy.tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)

    def generate(self, prompt, *args, **kwargs):
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        count = len(self.tokenizer.encode(rendered, add_special_tokens=False))
        if count > self.max_prompt_tokens:
            raise ValueError(f"Full actor prompt exceeds frozen limit: {count}>{self.max_prompt_tokens}; no truncation")
        return self.policy.generate(prompt, *args, **kwargs)


def storage_bytes(root):
    """Count owned files without following links or racing progress publication.

    ``atomic_write_json`` renames a short-lived .rank-N.json.<nonce> sibling.
    It may disappear between directory enumeration and lstat. Only that known
    progress-file race is tolerated; missing evidence and other I/O errors must
    still fail closed. This is capacity accounting, not an integrity audit.
    """
    used = 0
    for parent, _, names in os.walk(root):
        for name in names:
            path = Path(parent) / name
            try:
                entry = path.lstat()
            except FileNotFoundError:
                if ((path.parent.name == 'forward_progress' and re.fullmatch(r'\.rank-\d+\.json\.[a-zA-Z0-9_\-]+', name))
                        or re.fullmatch(r'\.runtime-status\.json\.[a-zA-Z0-9_\-]+', name)
                        or re.fullmatch(r'\.publish-[a-zA-Z0-9_\-]+', name)):
                    continue
                raise
            if not stat.S_ISLNK(entry.st_mode):
                used += entry.st_size
    return used


def disk_gate(root, required_bytes=0, *, minimum_free_bytes, maximum_run_bytes):
    root = Path(root).resolve()
    resource_path = root / 'resource_limits.json'
    if resource_path.is_file():
        cohort = read_json(resource_path).get('cohort_storage_root')
        if cohort is not None:
            cohort = Path(cohort).resolve()
            if root == cohort or not root.is_relative_to(cohort) or not (cohort / 'queue_launch.json').is_file():
                raise ValueError('Aggregate storage accounting requires this run\'s launched parent queue')
            root = cohort
    existing = root if root.exists() else root.parent
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    # Never follow directory symlinks into shared models/data or historical runs.
    used = storage_bytes(root)
    if free - required_bytes < minimum_free_bytes or used + required_bytes > maximum_run_bytes:
        raise OSError("Storage budget would be exceeded; no automatic deletion or budget waiver")
    return {"used_bytes": used, "free_bytes": free, "reserved_next_bytes": required_bytes}


def admit_capture(root, tokens, *, rows=None, copies=2):
    """Old+new exact FP32 vocabulary rows, before either distribution is saved."""
    path = Path(root) / "resource_limits.json"
    if not path.exists():
        return  # Historical capture behavior is unchanged.
    limits = read_json(path)
    if copies not in (1, 2):
        raise ValueError('Only explicit old-only or old+new full-vocabulary capture is supported')
    from .capture_storage import settings, encoded_bound
    compression = settings(root)
    required = (encoded_bound(compression, tokens, rows) if compression else
                int(tokens) * int(limits["vocab_size"]) * 4) * copies
    required += int(limits["checkpoint_reserve_bytes"])
    return disk_gate(root, required, minimum_free_bytes=int(limits["minimum_free_bytes"]),
                     maximum_run_bytes=int(limits["maximum_run_bytes"]))
