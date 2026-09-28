"""Opt-in CPU-batched execution of the same frozen state-aware retrieval rule.

FP32, model revision, query and catalog mapping are unchanged. Batch shape and
CPU execution are a NEW numerical protocol, with a new cache and no API fallback.
"""
from contextlib import ExitStack
from dataclasses import asdict
import json
import time

import numpy as np

from .router_cache import RouterCache, canonical_json, digest, utc_now
from .skillrl_embedding_router import (
    EmbeddingRouterError, FrozenSentenceEncoder, SkillRLEmbeddingStepRouter,
    _RNG_LOCK, load_profile, skill_texts,
)


VERSION = "skillrl-embedding-state-batch-top1-v1"


class BatchedSentenceEncoder(FrozenSentenceEncoder):
    def __init__(self, *args, intra_op_threads, forward_microbatch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.intra_op_threads = intra_op_threads
        if forward_microbatch_size is not None and (
                type(forward_microbatch_size) is not int
                or not 1 <= forward_microbatch_size <= self.config.encode_batch_size):
            raise EmbeddingRouterError("Invalid forward microbatch size")
        self.forward_microbatch_size = forward_microbatch_size

    def _forward_batch_size(self):
        return self.forward_microbatch_size or self.config.encode_batch_size

    def encode(self, texts):
        import torch
        # Only this synchronous encoding call uses extra CPU threads. Preserve
        # the training/evaluation process's original thread and RNG settings.
        with _RNG_LOCK:
            previous = torch.get_num_threads()
            try:
                torch.set_num_threads(self.intra_op_threads)
                vectors, usage = super().encode(texts)
                lengths = [len(ids) for ids in self.model.tokenizer(
                    list(texts), add_special_tokens=True, truncation=False)["input_ids"]]
                return vectors, {**usage, "input_token_counts": lengths}
            finally:
                torch.set_num_threads(previous)


def load_batch_profile(path):
    from pathlib import Path
    from .skillnet_runtime import DEFAULT_EMBEDDING_ROUTER_PROFILE
    data = json.loads(Path(path).read_bytes())
    base_bytes = DEFAULT_EMBEDDING_ROUTER_PROFILE.read_bytes()
    import hashlib
    if (set(data) != {"schema_version", "router_version", "base_profile_sha256", "execution"}
            or data["schema_version"] != "skillrl.embedding_batch_profile.v1"
            or data["router_version"] != VERSION
            or data["base_profile_sha256"] != hashlib.sha256(base_bytes).hexdigest()):
        raise EmbeddingRouterError("Invalid batched embedding profile/base identity")
    execution = data["execution"]
    if (set(execution) != {"device", "intra_op_threads", "request_order", "batch_accounting"}
            or execution["device"] != "cpu"
            or type(execution["intra_op_threads"]) is not int
            or not 1 <= execution["intra_op_threads"] <= 16
            or execution["request_order"] != "first_occurrence_deduplicated"
            or execution["batch_accounting"] != "equal_share_latency_exact_per_query_tokens"):
        raise EmbeddingRouterError("Unregistered batch execution settings")
    config, files = load_profile(DEFAULT_EMBEDDING_ROUTER_PROFILE)
    return config, files, execution


class BatchedEmbeddingStepRouter(SkillRLEmbeddingStepRouter):
    version = VERSION
    backend = "skillrl_embedding_state_batch"

    def __init__(self, memory, config, *, model_files, model_path, device,
                 cache_path, max_local_calls, execution, _encoder=None):
        self._validate_memory(memory)
        if device != "cpu" or execution["device"] != device:
            raise EmbeddingRouterError("This registered batched router is CPU-only")
        self.memory, self._config, self._encoder = memory, config, _encoder
        self._encoder_args = config, model_path, dict(model_files), device
        self._skill_embeddings = None
        import threading
        self._index_lock = threading.RLock()
        self.execution = dict(execution)
        from .skillrl_embedding_router import UPSTREAM_COMMIT
        self.protocol = {"version": self.version, "config": asdict(config), "device": device,
            "execution": self.execution, "upstream_commit": UPSTREAM_COMMIT,
            "bank_manifest_sha256": memory.bank.manifest_sha256, "model_files": dict(model_files),
            "catalog": memory.bank.router_catalog(), "skill_texts": skill_texts(memory),
            "query_formatter": "canonical-visible-state-json-v1-no-extra-prompt",
            "similarity": "l2-normalized-fp32-dot-product", "tie_break": "first-in-canonical-bank-order",
            "selection_count": 1, "external_api_calls": 0, "automatic_retries": 0}
        self.cache = RouterCache(cache_path, self.protocol, max_local_calls)

    def route(self, candidate_bundle, **state):
        return self.route_many([{"candidate_bundle": candidate_bundle, **state}])[0]

    def route_many(self, requests):
        requests = list(requests)
        if not requests:
            return []
        unique, ordered_keys = {}, []
        for request in requests:
            state = dict(request)
            self._validate_candidates(state.pop("candidate_bundle"))
            visible = self._visible_input(**state)
            query = canonical_json(visible)
            if len(query.encode("utf-8")) > self.config.max_input_utf8_bytes:
                raise EmbeddingRouterError("Visible state exceeds byte cap; no truncation")
            key = digest({"protocol_hash": self.protocol_hash, "input_hash": digest(visible)})
            unique.setdefault(key, (visible, query))
            ordered_keys.append(key)
        records, new_keys = {}, set()
        with ExitStack() as stack:
            # Canonical locking prevents deadlock between different job batches.
            for key in sorted(unique):
                stack.enter_context(self.cache.input_lock(key))
            pending = []
            for key, (visible, query) in unique.items():
                record = self.cache.lookup(key)
                if record is None:
                    pending.append(key)
                else:
                    self._validate_record(record, visible)
                    records[key] = record
            reservations = self.cache.reserve_many_local([(key, unique[key][0]) for key in pending])
            unfinished = dict(reservations)
            try:
                with self._index_lock:
                    if pending and self._encoder is None:
                        self._encoder = BatchedSentenceEncoder(*self._encoder_args,
                            intra_op_threads=self.execution["intra_op_threads"])
                    for start in range(0, len(pending), self.config.encode_batch_size):
                        keys = pending[start:start + self.config.encode_batch_size]
                        started = time.monotonic()
                        index = {"input_tokens": 0, "texts": 0, "encode_latency_ms": 0.0}
                        if self._skill_embeddings is None:
                            self._skill_embeddings, index_all = self._vectors(self.protocol["skill_texts"])
                            index = {key: index_all[key] for key in index}
                            self._skill_embeddings.setflags(write=False)
                        vectors, query_all = self._vectors([unique[key][1] for key in keys])
                        counts = query_all.get("input_token_counts")
                        if (not isinstance(counts, list) or len(counts) != len(keys)
                                or any(type(n) is not int or n < 0 for n in counts)
                                or sum(counts) != query_all["input_tokens"]):
                            raise EmbeddingRouterError("Exact per-query token accounting required")
                        # Deliberately keep the v1 FP32 vector dot for each query.
                        # Matrix GEMM would introduce a second numerical change.
                        similarities = [self._skill_embeddings @ vector for vector in vectors]
                        batch_ms = (time.monotonic() - started) * 1000
                        batch_id = digest({"protocol": self.protocol_hash, "queries": keys})
                        for i, key in enumerate(keys):
                            visible = unique[key][0]
                            sims = similarities[i]
                            index_usage = index if i == 0 else {name: 0 for name in index}
                            query_usage = {"input_tokens": counts[i], "texts": 1,
                                "encode_latency_ms": query_all["encode_latency_ms"] / len(keys)}
                            tokens = counts[i] + index_usage["input_tokens"]
                            record = {"cache_key": key, "protocol_hash": self.protocol_hash,
                                "input_hash": digest(visible), "visible_input": visible,
                                "selected_skill_id": self.memory.bank.skill_ids[int(np.argmax(sims))],
                                "scores": {sid: float(score) for sid, score in zip(self.memory.bank.skill_ids, sims)},
                                "created_at_utc": utc_now(), "response": {
                                    "backend": self.backend, "requested_model": self.config.model,
                                    "response_model": self.config.model, "model_revision": self.config.revision,
                                    "model_files_sha256": digest(self.protocol["model_files"]),
                                    "provider_endpoint": None, "provider_cost": 0.0,
                                    "provider_cost_currency": "USD", "provider_cost_scope": "external_api_only",
                                    "latency_ms": batch_ms / len(keys), "batch_id": batch_id,
                                    "batch_size": len(keys), "batch_wall_latency_ms": batch_ms,
                                    "batch_timing_scope": "encoding_and_scoring_excludes_sqlite_commit",
                                    "index_encoding": index_usage, "query_encoding": query_usage,
                                    "usage": {"prompt_tokens": tokens, "completion_tokens": 0,
                                        "total_tokens": tokens, "cached_input_tokens": 0, "reasoning_tokens": 0}}}
                            self.cache.finish(reservations[key], record)
                            unfinished.pop(key)
                            records[key] = record
                            new_keys.add(key)
            except BaseException as error:
                for attempt in unfinished.values():
                    self.cache.fail(attempt, {"error_type": type(error).__name__, "external_api_calls": 0,
                        "batch_forward_tokens_unknown": True, "automatic_retry": False})
                if not isinstance(error, Exception):
                    raise
                raise EmbeddingRouterError(f"Batched embedding failed ({type(error).__name__}); no retry or fallback") from error
            result = []
            for key in ordered_keys:
                cache_hit = key not in new_keys
                if cache_hit:
                    self.cache.note_hit(key)
                else:
                    new_keys.remove(key)
                result.append(self._bundle_from_record(records[key], cache_hit))
            return result
