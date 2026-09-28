"""SkillRL embedding retrieval adapted to state-aware, full-bank top-1.

Uses the official Qwen3-Embedding-0.6B encoder and normalized dot product.
Changes from upstream are explicit: visible-state query, flat SkillNet catalog,
one ID per step, canonical-order tie break, and auditable immutable decisions.
No external API, reranker, generated advice, policy sharing or online training.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from .external_skill_router import ExternalLLMSkillRouter
from .frozen_skill_bank import SKILLNET37_MANIFEST_SHA256
from .router_cache import RouterBudgetExceeded, RouterCache, canonical_json, digest, utc_now
from .skills_only_memory import SkillsOnlyMemory


VERSION = "skillrl-embedding-state-top1-v1"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
UPSTREAM_COMMIT = "8e66726ed866a4e0a7f053586a41022798192e6c"
_RNG_LOCK = threading.RLock()


class EmbeddingRouterError(ValueError):
    pass


@contextmanager
def isolated_encoder_rng(device):
    """Also covers lazy dependency imports, which consume Python RNG on import.

Routing is synchronous with policy sampling in the manager/evaluator. This
lock serializes encoder instances, not arbitrary unrelated sampling threads.
"""
    import torch
    devices = [torch.device(device).index] if device.startswith("cuda:") else []
    with _RNG_LOCK:
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=devices):
                yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


@dataclass(frozen=True)
class EmbeddingRouterConfig:
    model: str = MODEL_ID
    revision: str = MODEL_REVISION
    history_length: int = 2
    max_input_utf8_bytes: int = 65536
    max_prompt_tokens: int = 8192
    embedding_dim: int = 1024
    encode_batch_size: int = 8
    dtype: str = "float32"
    attn_implementation: str = "sdpa"
    sentence_transformers_version: str = "6.0.1"
    transformers_version: str = "5.10.4"
    torch_version: str = "2.11.0"
    tokenizers_version: str = "0.22.2"

    def __post_init__(self):
        if (self.model != MODEL_ID or self.revision != MODEL_REVISION or self.dtype != "float32"
                or self.attn_implementation != "sdpa" or self.embedding_dim != 1024):
            raise EmbeddingRouterError("Unregistered encoder/revision/numeric protocol")
        for name in ("max_input_utf8_bytes", "max_prompt_tokens", "encode_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise EmbeddingRouterError(f"Invalid {name}")
        if type(self.history_length) is not int or self.history_length < 0:
            raise EmbeddingRouterError("Invalid visible-history window")


def load_profile(path):
    data = json.loads(Path(path).read_bytes())
    if (set(data) != {"schema_version", "router_version", "upstream_commit", "bank_manifest_sha256", "model_files", "router"}
            or data["schema_version"] != "skillrl.embedding_router_profile.v1" or data["router_version"] != VERSION
            or data["upstream_commit"] != UPSTREAM_COMMIT or data["bank_manifest_sha256"] != SKILLNET37_MANIFEST_SHA256):
        raise EmbeddingRouterError("Invalid embedding router profile")
    files = data["model_files"]
    required = {"config.json", "config_sentence_transformers.json", "modules.json", "1_Pooling/config.json",
                "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
    if not isinstance(files, dict) or not required <= set(files):
        raise EmbeddingRouterError("Incomplete encoder snapshot inventory")
    for name, sha in files.items():
        p = PurePosixPath(name)
        if (p.is_absolute() or p.as_posix() != name or ".." in p.parts or "\\" in name
                or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)):
            raise EmbeddingRouterError("Invalid encoder snapshot path/hash")
    try:
        return EmbeddingRouterConfig(**data["router"]), files
    except TypeError:
        raise EmbeddingRouterError("Unknown embedding router option") from None


def skill_texts(memory):
    # SkillNet has name/description, not the original three SkillRL fields.
    # Do not invent applicability text or include interventions in this mapping.
    return [SkillsOnlyMemory._skill_to_text({"title": item["name"], "principle": item["description"],
                                           "when_to_apply": ""}) for item in memory.bank.router_catalog()]


def _signature(path):
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def verify_snapshot(model_path, files):
    root = Path(model_path).resolve(strict=True)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and ".cache" not in p.relative_to(root).parts
              and (p.suffix in (".json", ".safetensors", ".bin", ".jinja") or p.name == "merges.txt")}
    if actual != set(files):
        raise EmbeddingRouterError("Encoder file inventory changed")
    result = {}
    for name, expected in files.items():
        path = root / name
        before = _signature(path)
        with path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        after = _signature(path)
        if before != after or actual_hash != expected:
            raise EmbeddingRouterError(f"Encoder file hash mismatch: {name}")
        result[name] = after
    return result


class FrozenSentenceEncoder:
    """Local-only SentenceTransformer load; parameters independent of policy."""
    def __init__(self, config, model_path, model_files, device):
        if device != "cpu" and not re.fullmatch(r"cuda:\d+", device):
            raise EmbeddingRouterError("Explicit cpu/cuda:N placement required; no auto fallback")
        self.config, self.root, self.files, self.device = config, Path(model_path).resolve(), dict(model_files), device
        self.model = None
        self.signatures = None
        self.lock = threading.RLock()

    def _check_files(self):
        if any(_signature(self.root / name) != sig for name, sig in self.signatures.items()):
            raise EmbeddingRouterError("Encoder snapshot changed after load")

    def _load(self):
        if self.model is not None:
            return
        for name in ("sentence-transformers", "transformers", "torch", "tokenizers"):
            if importlib.metadata.version(name) != getattr(self.config, name.replace("-", "_") + "_version"):
                raise EmbeddingRouterError(f"{name} differs from the frozen router profile")
        self.signatures = verify_snapshot(self.root, self.files)
        import torch
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(self.root), device=self.device, local_files_only=True,
            trust_remote_code=False, model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"})
        model.requires_grad_(False)
        model.eval()
        # Match SkillRL's encode(texts, ...) without a query-specific prompt.
        if model.default_prompt_name is not None:
            raise EmbeddingRouterError("Unexpected implicit query prompt in the frozen encoder")
        if model.get_embedding_dimension() != self.config.embedding_dim:
            raise EmbeddingRouterError("Wrong encoder output dimension")
        model.max_seq_length = self.config.max_prompt_tokens
        self._check_files()
        self.model = model

    def encode(self, texts):
        import torch
        with self.lock, isolated_encoder_rng(self.device):
            self._load()
            self._check_files()
            if self.model.training or any(p.requires_grad for p in self.model.parameters()):
                raise EmbeddingRouterError("Encoder must stay frozen in eval mode")
            encoded = self.model.tokenizer(list(texts), add_special_tokens=True, truncation=False)
            lengths = [len(row) for row in encoded["input_ids"]]
            if any(length > self.config.max_prompt_tokens for length in lengths):
                raise EmbeddingRouterError("Embedding input exceeds token cap; no silent truncation")
            started = time.monotonic()
            with torch.inference_mode(), torch.autocast(
                    device_type=self.device.split(":")[0], enabled=False):
                embeddings = self.model.encode(list(texts), batch_size=self._forward_batch_size(),
                    normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            self._check_files()
            return np.asarray(embeddings, dtype=np.float32), {
                "input_tokens": sum(lengths), "texts": len(texts),
                "encode_latency_ms": round((time.monotonic() - started) * 1000, 3)}

    def close(self):
        with self.lock:
            self.model = None

    def _forward_batch_size(self):
        return self.config.encode_batch_size


class SkillRLEmbeddingStepRouter(ExternalLLMSkillRouter):
    """Only inherits visible-state/candidate validation; never uses an API client."""
    version = VERSION
    backend = "skillrl_embedding_state"

    def _validate_memory(self, memory):
        if memory.bank.manifest_sha256 != SKILLNET37_MANIFEST_SHA256:
            raise EmbeddingRouterError("Phase1/2 v1 requires the complete frozen SkillNet-37 bank")

    def __init__(self, memory, config, *, model_files, model_path, device, cache_path,
                 max_local_calls, _encoder=None):
        self._validate_memory(memory)
        if device != "cpu" and not re.fullmatch(r"cuda:\d+", device):
            raise EmbeddingRouterError("Explicit encoder device required")
        self.memory, self._config, self._encoder = memory, config, _encoder
        self._encoder_args = config, model_path, dict(model_files), device
        self._skill_embeddings = None
        self._index_lock = threading.RLock()
        self.protocol = {"version": self.version, "config": asdict(config), "device": device,
            "upstream_commit": UPSTREAM_COMMIT, "bank_manifest_sha256": memory.bank.manifest_sha256,
            "model_files": dict(model_files), "catalog": memory.bank.router_catalog(),
            "skill_texts": skill_texts(memory), "query_formatter": "canonical-visible-state-json-v1-no-extra-prompt",
            "similarity": "l2-normalized-fp32-dot-product", "tie_break": "first-in-canonical-bank-order",
            "selection_count": 1, "external_api_calls": 0, "automatic_retries": 0}
        # Historical cache column names are retained; they count LOCAL query
        # attempts for this protocol, never paid API calls. See stats().
        self.cache = RouterCache(cache_path, self.protocol, max_local_calls)

    def _get_client(self):
        raise EmbeddingRouterError("Embedding routing never uses an external API")

    def _vectors(self, texts):
        vectors, accounting = self._encoder.encode(texts)
        vectors = np.asarray(vectors, dtype=np.float32)
        if (vectors.shape != (len(texts), self.config.embedding_dim) or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
            raise EmbeddingRouterError("Invalid/non-normalized encoder vectors")
        if type(accounting.get("input_tokens")) is not int or accounting["input_tokens"] < 0:
            raise EmbeddingRouterError("Missing embedding token accounting")
        return vectors, accounting

    def _validate_record(self, record, visible):
        ids = self.memory.bank.skill_ids
        scores = record.get("scores", {})
        if record.get("visible_input") != visible or set(scores) != set(ids):
            raise EmbeddingRouterError("Cached state/full-bank score mismatch")
        values = np.asarray([scores[sid] for sid in ids], dtype=np.float32)
        if not np.isfinite(values).all() or record.get("selected_skill_id") != ids[int(np.argmax(values))]:
            raise EmbeddingRouterError("Cached selection is not the canonical cosine top-1")

    def route(self, candidate_bundle, *, task_description, current_observation, admissible_actions, history, step_index):
        self._validate_candidates(candidate_bundle)
        visible = self._visible_input(task_description, current_observation, admissible_actions, history, step_index)
        query = canonical_json(visible)
        if len(query.encode("utf-8")) > self.config.max_input_utf8_bytes:
            raise EmbeddingRouterError("Visible state exceeds byte cap; no truncation")
        input_hash = digest(visible)
        key = digest({"protocol_hash": self.protocol_hash, "input_hash": input_hash})
        cache_hit = False
        with self.cache.input_lock(key):
            record = self.cache.lookup(key)
            if record is not None:
                self._validate_record(record, visible)
                self.cache.note_hit(key)
                cache_hit = True
            else:
                try:
                    attempt = self.cache.reserve(key, visible)  # Cap checked before any load/forward.
                except RouterBudgetExceeded:
                    raise RouterBudgetExceeded("Local embedding-call budget exhausted; cached decisions remain usable") from None
                started = time.monotonic()
                index_usage = {"input_tokens": 0, "texts": 0, "encode_latency_ms": 0.0}
                query_usage = None
                try:
                    with self._index_lock:
                        if self._encoder is None:
                            self._encoder = FrozenSentenceEncoder(*self._encoder_args)
                        if self._skill_embeddings is None:
                            self._skill_embeddings, index_usage = self._vectors(self.protocol["skill_texts"])
                            self._skill_embeddings.setflags(write=False)
                        vectors, query_usage = self._vectors([query])
                        sims = self._skill_embeddings @ vectors[0]
                    winner = int(np.argmax(sims))
                    selected = self.memory.bank.skill_ids[winner]
                    scores = {sid: float(score) for sid, score in zip(self.memory.bank.skill_ids, sims)}
                    token_count = index_usage["input_tokens"] + query_usage["input_tokens"]
                    usage = {"prompt_tokens": token_count, "completion_tokens": 0, "total_tokens": token_count,
                             "cached_input_tokens": 0, "reasoning_tokens": 0}
                except Exception as error:
                    self.cache.fail(attempt, {"error_type": type(error).__name__, "external_api_calls": 0,
                        "completed_index_encoding": index_usage, "completed_query_encoding": query_usage,
                        "failed_forward_tokens_unknown": True,
                        "latency_ms": round((time.monotonic() - started) * 1000, 3)})
                    raise EmbeddingRouterError(f"Embedding router failed ({type(error).__name__}); no retry or fallback") from error
                record = {"cache_key": key, "protocol_hash": self.protocol_hash, "input_hash": input_hash,
                    "visible_input": visible, "selected_skill_id": selected, "scores": scores, "created_at_utc": utc_now(),
                    "response": {"backend": self.backend, "requested_model": self.config.model,
                        "response_model": self.config.model, "model_revision": self.config.revision,
                        "model_files_sha256": digest(self.protocol["model_files"]), "provider_endpoint": None,
                        "provider_cost": 0.0, "provider_cost_currency": "USD", "provider_cost_scope": "external_api_only",
                        "latency_ms": round((time.monotonic() - started) * 1000, 3), "usage": usage,
                        "index_encoding": index_usage, "query_encoding": query_usage}}
                self.cache.finish(attempt, record)
        return self._bundle_from_record(record, cache_hit)

    def _bundle_from_record(self, record, cache_hit):
        key, input_hash = record["cache_key"], record["input_hash"]
        bundle = self.memory.selected_bundle(record["selected_skill_id"])
        bundle.update({"disabled_skill_ids": [], "skill_router_version": self.version,
            "skill_router_scores": dict(record["scores"]), "skill_router_score_details": {}, "skill_router_state_flags": [],
            "skill_router_selection_reason": "frozen SkillRL embedding; state-aware all-bank cosine top-1 adaptation",
            "skill_router_api": {**record["response"], "protocol_hash": self.protocol_hash, "input_hash": input_hash,
                "cache_key": key, "cache_hit": cache_hit, "api_calls_this_step": 0, "local_calls_this_step": int(not cache_hit),
                "source_decision_latency_ms": record["response"]["latency_ms"],
                "latency_ms": 0.0 if cache_hit else record["response"]["latency_ms"],
                "source_decision_index_encoding": record["response"]["index_encoding"],
                "index_encoding": {k: 0 for k in record["response"]["index_encoding"]} if cache_hit else record["response"]["index_encoding"],
                "source_decision_query_encoding": record["response"]["query_encoding"],
                "query_encoding": {k: 0 for k in record["response"]["query_encoding"]} if cache_hit else record["response"]["query_encoding"],
                "source_decision_usage": record["response"]["usage"],
                "usage": {k: 0 for k in record["response"]["usage"]} if cache_hit else record["response"]["usage"]}})
        return bundle

    def stats(self):
        stats = self.cache.stats()
        return {"local_attempts": stats["api_attempts"], "max_local_calls": stats["max_api_calls"],
                "successful_decisions": stats["successful_decisions"], "failed_attempts": stats["failed_attempts"],
                "cache_hits": stats["cache_hits"], "external_api_calls": 0}

    def close(self):
        if self._encoder is not None:
            self._encoder.close()
        self._encoder = self._skill_embeddings = None
