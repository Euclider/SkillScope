"""SHELVED / UNVALIDATED draft, superseded by the SkillRL retrieval choice.

No profile was frozen and no inference was run. The public factory is blocked.
Retained as interrupted work, not an available or verified experiment backend.

Opt-in, hash-pinned Qwen3.5-4B router; never shares a live policy module.

The visible-state, complete-catalog and payload contracts match the external
router. Inference is local-only, greedy, non-thinking and constrained to one
canonical JSON skill ID. Historical profiles and caches remain unchanged.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from .external_skill_router import ExternalLLMSkillRouter, SYSTEM_PROMPT, SYSTEM_PROMPT_SHA256
from .frozen_skill_bank import SKILLNET37_MANIFEST_SHA256
from .router_cache import RouterCache, canonical_json, digest, utc_now


LOCAL_ROUTER_VERSION = "skillnet37-local-qwen35-step-v1"
MODEL_ID = "Qwen/Qwen3.5-4B"


class LocalRouterError(ValueError):
    pass


@dataclass(frozen=True)
class LocalRouterConfig:
    model: str = MODEL_ID
    history_length: int = 2
    max_input_utf8_bytes: int = 65536
    max_prompt_tokens: int = 8192
    max_completion_tokens: int = 128
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    enable_thinking: bool = False
    do_sample: bool = False
    generation_batch_size: int = 1
    transformers_version: str = "5.10.4"
    torch_version: str = "2.11.0"
    tokenizers_version: str = "0.22.2"
    safetensors_version: str = "0.8.0"

    def __post_init__(self):
        if (self.model != MODEL_ID or self.dtype != "bfloat16" or self.attn_implementation != "sdpa"
                or self.enable_thinking is not False or self.do_sample is not False
                or type(self.generation_batch_size) is not int or self.generation_batch_size != 1):
            raise LocalRouterError("v1 fixes the independent Qwen3.5-4B BF16/SDPA non-thinking greedy batch-1 router")
        for name in ("max_input_utf8_bytes", "max_prompt_tokens", "max_completion_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise LocalRouterError(f"Invalid {name}")
        if type(self.history_length) is not int or self.history_length < 0:
            raise LocalRouterError("history_length must be a nonnegative integer")

    @property
    def expected_response_model(self):
        return self.model


def load_profile(path):
    value = json.loads(Path(path).read_bytes())
    if (set(value) != {"schema_version", "router_version", "bank_manifest_sha256", "system_prompt_sha256", "model_files", "router"}
            or value["schema_version"] != "skillrl.local_router_profile.v1"
            or value["router_version"] != LOCAL_ROUTER_VERSION
            or value["bank_manifest_sha256"] != SKILLNET37_MANIFEST_SHA256
            or value["system_prompt_sha256"] != SYSTEM_PROMPT_SHA256):
        raise LocalRouterError("Invalid local router profile or identity")
    files = value["model_files"]
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                "merges.txt", "vocab.json", "model.safetensors.index.json"}
    if not isinstance(files, dict) or not required <= set(files) or not any(n.endswith(".safetensors") for n in files):
        raise LocalRouterError("Incomplete frozen model inventory")
    for name, sha in files.items():
        if (not isinstance(name, str) or Path(name).name != name or name in (".", "..")
                or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)):
            raise LocalRouterError("Unsafe model filename or invalid hash")
    try:
        config = LocalRouterConfig(**value["router"])
    except TypeError:
        raise LocalRouterError("Unknown local router configuration field") from None
    return config, files


def _file_signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def verify_model_files(model_path, files):
    """Hash every declared byte; reject extra weights/tokenizer configuration."""
    root = Path(model_path).resolve(strict=True)
    actual = {p.name for p in root.iterdir()
              if p.suffix in (".safetensors", ".bin", ".json", ".jinja") or p.name == "merges.txt"}
    # Vision preprocessors are irrelevant to the text-only router.
    actual -= {"preprocessor_config.json", "video_preprocessor_config.json"}
    if actual != set(files):
        raise LocalRouterError("Local snapshot file inventory differs from the frozen profile")
    signatures = {}
    for name, expected in files.items():
        path = root / name
        before = _file_signature(path)
        with path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        after = _file_signature(path)
        if actual_hash != expected or before != after:
            raise LocalRouterError(f"Frozen router snapshot changed: {name}")
        signatures[name] = after
    index = json.loads((root / "model.safetensors.index.json").read_bytes())
    if set(index["weight_map"].values()) != {n for n in files if n.endswith(".safetensors")}:
        raise LocalRouterError("Frozen model index disagrees with its weight shards")
    config = json.loads((root / "config.json").read_bytes())
    if config.get("model_type") != "qwen3_5":
        raise LocalRouterError("Expected a Qwen3.5 snapshot, not a trained policy checkpoint")
    return signatures


class SkillIDTrie:
    """Finite grammar: exactly one of the 37 canonical JSON objects, then EOS."""
    def __init__(self, sequences, eos_token_id):
        self.root = {}
        self.maximum_length = 0
        self.eos = eos_token_id
        for seq in sequences:
            if not seq or eos_token_id in seq:
                raise LocalRouterError("Invalid skill-ID token sequence")
            node = self.root
            self.maximum_length = max(self.maximum_length, len(seq) + 1)
            for token in [*seq, eos_token_id]:
                node = node.setdefault(int(token), {})

    def allowed(self, prefix):
        node = self.root
        for token in prefix:
            if int(token) not in node:
                raise LocalRouterError("Generation escaped the fixed skill-ID grammar")
            node = node[int(token)]
        return sorted(node) if node else [self.eos]


class FrozenQwenGenerator:
    """Owns a fresh checkpoint load, not a policy/reference-model pointer.

    Loading and inference preserve the caller's PyTorch RNG. No downloads,
    adapters, API credentials, optimizer or policy weight synchronization.
    """
    def __init__(self, config, model_path, model_files, device):
        if not re.fullmatch(r"cuda:\d+", device):
            raise LocalRouterError("v1 requires an explicit CUDA device; no automatic placement/CPU fallback")
        self.config, self.model_path, self.model_files = config, Path(model_path).resolve(), dict(model_files)
        self.device = device
        self._model = self._tokenizer = None
        self._signatures = None
        self._lock = threading.RLock()

    def _assert_snapshot_unchanged(self):
        for name, signature in self._signatures.items():
            if _file_signature(self.model_path / name) != signature:
                raise LocalRouterError("Frozen router files changed after verification; no policy reload")

    def _load(self):
        if self._model is not None:
            return
        for package in ("torch", "transformers", "tokenizers", "safetensors"):
            if importlib.metadata.version(package) != getattr(self.config, package + "_version"):
                raise LocalRouterError(f"{package} differs from the frozen local router profile")
        self._signatures = verify_model_files(self.model_path, self.model_files)
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
        device = torch.device(self.device)
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise LocalRouterError("Assigned router GPU is unavailable; no fallback")
        # Model loading may initialize transient tensors: isolate it from actor RNG.
        with torch.random.fork_rng(devices=[device.index]):
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True,
                                                      trust_remote_code=False)
            model = Qwen3_5ForCausalLM.from_pretrained(
                str(self.model_path), dtype=torch.bfloat16, attn_implementation="sdpa",
                local_files_only=True, trust_remote_code=False, device_map={"": self.device})
            model.requires_grad_(False)
            model.eval()
        self._assert_snapshot_unchanged()
        self._tokenizer, self._model = tokenizer, model

    def generate(self, messages, skill_ids):
        import torch
        from transformers import GenerationConfig
        with self._lock:
            self._load()
            self._assert_snapshot_unchanged()
            model, tokenizer = self._model, self._tokenizer
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise LocalRouterError("Router is no longer frozen in evaluation mode")
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False)
            inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
            prompt_length = int(inputs["input_ids"].shape[1])
            if prompt_length > self.config.max_prompt_tokens:
                raise LocalRouterError("Router prompt exceeds frozen token limit; no truncation")
            outputs = [canonical_json({"skill_id": skill_id}) for skill_id in skill_ids]
            sequences = [tokenizer.encode(text, add_special_tokens=False) for text in outputs]
            if any(tokenizer.decode(seq, skip_special_tokens=True) != text for seq, text in zip(sequences, outputs)):
                raise LocalRouterError("Tokenizer cannot round-trip a canonical skill-ID response")
            trie = SkillIDTrie(sequences, tokenizer.eos_token_id)
            if trie.maximum_length > self.config.max_completion_tokens:
                raise LocalRouterError("Output limit cannot accommodate every skill ID")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            generation = GenerationConfig(
                do_sample=False, num_beams=1, max_new_tokens=self.config.max_completion_tokens,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id, use_cache=True, repetition_penalty=1.0)
            with torch.random.fork_rng(devices=[torch.device(self.device).index]), torch.inference_mode():
                result = model.generate(**inputs, generation_config=generation,
                    prefix_allowed_tokens_fn=lambda batch_id, ids: trie.allowed(ids[prompt_length:].tolist()))
            tokens = result[0, prompt_length:].tolist()
            self._assert_snapshot_unchanged()
            if not tokens or tokens[-1] != tokenizer.eos_token_id:
                raise LocalRouterError("Local router response was truncated")
            return {"content": tokenizer.decode(tokens, skip_special_tokens=True),
                    "prompt_tokens": prompt_length, "completion_tokens": len(tokens),
                    "hardware": torch.cuda.get_device_name(torch.device(self.device)),
                    "trainable_parameters": 0, "thinking_enabled": False}

    def close(self):
        with self._lock:
            self._model = self._tokenizer = None
            gc.collect()


class LocalFrozenSkillRouter(ExternalLLMSkillRouter):
    """Reuses only the validated catalog/state/schema contract, never its API path.

    RouterCache retains its historical column names. Here its attempts count
    local inference attempts; metadata explicitly distinguishes them from APIs.
    """
    version = LOCAL_ROUTER_VERSION

    def __init__(self, memory, config, *, model_files, model_path, device, cache_path,
                 max_local_calls, _generator=None):
        if memory.bank.manifest_sha256 != SKILLNET37_MANIFEST_SHA256:
            raise LocalRouterError("Local v1 requires the complete frozen SkillNet-37 bank")
        if not re.fullmatch(r"cuda:\d+", device):
            raise LocalRouterError("Explicit CUDA placement is required")
        self.memory, self._config = memory, config
        self._generator = _generator
        self._generator_args = (config, model_path, dict(model_files), device)
        self.protocol = {
            "version": self.version, "bank_manifest_sha256": memory.bank.manifest_sha256,
            "config": asdict(config), "model_files": dict(model_files), "device": device,
            "system_prompt": SYSTEM_PROMPT, "catalog": memory.bank.router_catalog(),
            "response_format": self._response_format(), "selection_count": 1,
            "decoding": "greedy-canonical-json-id-trie-v1", "automatic_retries": 0,
            "backend": "local_frozen_hf", "api_calls": 0,
        }
        self.cache = RouterCache(cache_path, self.protocol, max_local_calls)

    def _get_client(self):
        raise LocalRouterError("A local router must never construct an external API client")

    def route(self, candidate_bundle, *, task_description, current_observation, admissible_actions, history, step_index):
        self._validate_candidates(candidate_bundle)
        visible = self._visible_input(task_description, current_observation, admissible_actions, history, step_index)
        user = canonical_json({"candidates": self.memory.bank.router_catalog(), "state": visible})
        if len((SYSTEM_PROMPT + user).encode("utf-8")) > self.config.max_input_utf8_bytes:
            raise LocalRouterError("Router input exceeds explicit byte cap; no truncation")
        input_hash = digest(visible)
        key = digest({"protocol_hash": self.protocol_hash, "input_hash": input_hash})
        cache_hit = False
        with self.cache.input_lock(key):
            record = self.cache.lookup(key)
            if record is not None:
                if record.get("visible_input") != visible or record.get("selected_skill_id") not in self.memory.bank.skill_ids:
                    raise LocalRouterError("Cached local selection/input mismatch")
                self.cache.note_hit(key)
                cache_hit = True
            else:
                # Check the local-call cap BEFORE any model load or CUDA allocation.
                attempt = self.cache.reserve(key, visible)
                started = time.monotonic()
                try:
                    if self._generator is None:
                        self._generator = FrozenQwenGenerator(*self._generator_args)
                    output = self._generator.generate(
                        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                        self.memory.bank.skill_ids)
                    response = SimpleNamespace(model=self.config.model, choices=[SimpleNamespace(
                        finish_reason="stop", message=SimpleNamespace(content=output["content"]))])
                    selected = self._parse_response(response)
                    if output.get("trainable_parameters") != 0 or output.get("thinking_enabled") is not False:
                        raise LocalRouterError("Local engine did not attest frozen non-thinking inference")
                    for name in ("prompt_tokens", "completion_tokens"):
                        if type(output.get(name)) is not int or output[name] <= 0:
                            raise LocalRouterError("Missing local token accounting")
                    usage = {"prompt_tokens": output["prompt_tokens"], "completion_tokens": output["completion_tokens"],
                             "total_tokens": output["prompt_tokens"] + output["completion_tokens"],
                             "cached_input_tokens": 0, "reasoning_tokens": 0}
                except Exception as error:
                    self.cache.fail(attempt, {"error_type": type(error).__name__, "external_api_calls": 0,
                                             "latency_ms": round((time.monotonic() - started) * 1000, 3)})
                    raise LocalRouterError(f"Local router failed ({type(error).__name__}); no retry/API fallback") from error
                record = {
                    "cache_key": key, "protocol_hash": self.protocol_hash, "visible_input": visible,
                    "input_hash": input_hash, "selected_skill_id": selected, "created_at_utc": utc_now(),
                    "response": {"requested_model": self.config.model, "response_model": self.config.model,
                        "backend": "local_frozen_hf", "provider_endpoint": None, "provider_cost": 0.0,
                        "provider_cost_currency": "USD", "provider_cost_scope": "external_api_only",
                        "model_files_sha256": digest(self.protocol["model_files"]), "hardware": output.get("hardware"),
                        "latency_ms": round((time.monotonic() - started) * 1000, 3), "usage": usage},
                }
                self.cache.finish(attempt, record)
        bundle = self.memory.selected_bundle(record["selected_skill_id"])
        bundle.update({"disabled_skill_ids": [], "skill_router_version": self.version,
            "skill_router_scores": {}, "skill_router_score_details": {}, "skill_router_state_flags": [],
            "skill_router_selection_reason": "independent frozen local Qwen3.5-4B ID selection; canonical payload only",
            # Existing archives already propagate this carrier; backend prevents
            # local inference from being misreported as a paid API call.
            "skill_router_api": {**record["response"], "protocol_hash": self.protocol_hash, "input_hash": input_hash,
                "cache_key": key, "cache_hit": cache_hit, "api_calls_this_step": 0,
                "local_calls_this_step": 0 if cache_hit else 1,
                "source_decision_latency_ms": record["response"]["latency_ms"],
                "latency_ms": 0.0 if cache_hit else record["response"]["latency_ms"],
                "source_decision_usage": record["response"]["usage"],
                "usage": {name: 0 for name in record["response"]["usage"]} if cache_hit else record["response"]["usage"]},
        })
        return bundle

    def stats(self):
        stats = self.cache.stats()
        return {"local_attempts": stats["api_attempts"], "max_local_calls": stats["max_api_calls"],
                "successful_decisions": stats["successful_decisions"], "failed_attempts": stats["failed_attempts"],
                "cache_hits": stats["cache_hits"], "external_api_calls": 0}

    def close(self):
        if self._generator is not None:
            self._generator.close()
            self._generator = None
