"""The Phase1/2 encoder/query rule over immutable, evolving Phase3 snapshots.

Every bank gets a separate protocol-bound decision cache and index. A single
process-safe branch ledger caps LOCAL attempts across all bank versions,
training, gate and final evaluation. No paid router API is constructed.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from agent_system.memory.frozen_skill_bank import SKILLNET37_MANIFEST_SHA256
from agent_system.memory.router_cache import RouterBudgetExceeded, RouterCache
from agent_system.memory.skillnet_runtime import DEFAULT_EMBEDDING_ROUTER_PROFILE
from agent_system.memory.skillrl_embedding_batch_router import BatchedEmbeddingStepRouter, BatchedSentenceEncoder
from agent_system.memory.skillrl_embedding_router import (
    FrozenSentenceEncoder, SkillRLEmbeddingStepRouter, UPSTREAM_COMMIT, load_profile, skill_texts,
)
from skillnet_cohort.common import file_hash
from .bank import Bank
from .common import canonical, digest, positive_int, require, strict_json
from .routing import BranchMemory


BACKEND = "skillrl_embedding_state"
VERSION = "phase3-skillrl-embedding-state-top1-v1"
BATCH_VERSION = "phase3-skillrl-embedding-state-batch-top1-v2"
MICROBATCH_VERSION = "phase3-skillrl-embedding-state-batch-top1-v3"


def batch_execution(settings):
    execution = settings.get("execution")
    if execution is None:
        return None
    mode = execution.get("mode") if isinstance(execution, Mapping) else None
    legacy_keys = ({"mode", "intra_op_threads"},
                   {"mode", "intra_op_threads", "shared_gpu_physical_id", "transport"})
    micro_keys = {"mode", "intra_op_threads", "shared_gpu_physical_id", "transport",
                  "forward_microbatch_size"}
    require(isinstance(execution, Mapping)
            and ((mode == "state_batch_fp32_v1" and set(execution) in legacy_keys)
                 or (mode == "state_batch_fp32_micro_v2" and set(execution) == micro_keys
                     and type(execution["forward_microbatch_size"]) is int
                     and 1 <= execution["forward_microbatch_size"] <= 8))
            and type(execution["intra_op_threads"]) is int
            and 1 <= execution["intra_op_threads"] <= 16,
            "Invalid registered FP32 state-batch execution settings")
    if settings["device"] == "cpu":
        require("shared_gpu_physical_id" not in execution,
                "CPU batch must not declare a shared GPU")
    else:
        require(settings["device"] == "cuda:0"
                and type(execution.get("shared_gpu_physical_id")) is int
                and execution["shared_gpu_physical_id"] >= 0
                and execution.get("transport") == "subprocess_pipe_v1",
                "GPU batch requires local cuda:0 and an isolated physical-GPU sidecar")
    if mode == "state_batch_fp32_micro_v2":
        require(settings["device"] == "cuda:0", "Microbatch protocol requires GPU sidecar")
    return dict(execution)


def validate_settings(settings):
    base = {"backend", "model_path", "device", "profile_sha256", "max_local_calls"}
    require(base <= set(settings) <= base | {"execution", "python_executable", "runtime_variant"},
            "Embedding router requires explicit model/device/profile/local-call limit; no API fields")
    if "python_executable" in settings:
        require(isinstance(settings["python_executable"], str) and Path(settings["python_executable"]).is_absolute(),
                "Router interpreter must be an absolute path")
    require(settings["backend"] == BACKEND, "Unknown local router")
    require(isinstance(settings["model_path"], str) and Path(settings["model_path"]).is_absolute(),
            "Absolute local embedding model path required")
    import re
    require(isinstance(settings["device"], str) and (settings["device"] == "cpu" or re.fullmatch(r"cuda:\d+", settings["device"])),
            "Explicit cpu/cuda:N router device required")
    require(file_hash(DEFAULT_EMBEDDING_ROUTER_PROFILE) == settings["profile_sha256"], "Embedding profile changed")
    batch_execution(settings)
    positive_int(settings["max_local_calls"], "branch-wide local router budget")
    config, files = load_profile(DEFAULT_EMBEDDING_ROUTER_PROFILE)
    return runtime_profile(config, settings['device'], settings.get('runtime_variant')), files


def runtime_profile(config, device, variant=None):
    if variant is None:
        return config
    require(variant == 'cpu_torch_2_11_0_v1' and device == 'cpu', 'Invalid explicit CPU runtime variant')
    from dataclasses import replace
    return replace(config, torch_version='2.11.0+cpu')


class LocalLedger:
    def __init__(self, path, profile, max_calls):
        self.path, self.limit = Path(path), max_calls
        require(not any(p.is_symlink() for p in (self.path, *self.path.parents)), "Symlinked local router ledger")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            require(not tables or tables == {"profile", "local_attempts", "local_hits"}, "Foreign local router ledger schema")
            db.execute("CREATE TABLE IF NOT EXISTS profile (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS local_attempts (key TEXT PRIMARY KEY, bank TEXT NOT NULL, result TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS local_hits (key TEXT NOT NULL)")
            value = canonical({**profile, "max_local_calls": max_calls})
            db.execute("INSERT OR IGNORE INTO profile VALUES (1, ?)", (value,))
            require(db.execute("SELECT value FROM profile WHERE id=1").fetchone() == (value,), "Changed branch encoder/budget profile")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(str(self.path), timeout=60)
        try:
            with db:
                db.execute("BEGIN IMMEDIATE")
                yield db
        finally:
            db.close()

    def reserve(self, key, bank):
        with self.connection() as db:
            require(db.execute("SELECT key FROM local_attempts WHERE key=?", (key,)).fetchone() is None,
                    "Prior failed/ambiguous local decision requires reconciliation; no automatic retry")
            if db.execute("SELECT COUNT(*) FROM local_attempts").fetchone()[0] >= self.limit:
                raise RouterBudgetExceeded("Branch-wide local router budget exhausted")
            db.execute("INSERT INTO local_attempts (key, bank) VALUES (?, ?)", (key, bank))

    def reserve_many(self, keys, bank):
        keys = list(keys)
        require(len(keys) == len(set(keys)), "Duplicate batch-local ledger reservation")
        with self.connection() as db:
            for key in keys:
                require(db.execute("SELECT key FROM local_attempts WHERE key=?", (key,)).fetchone() is None,
                        "Prior failed/ambiguous local decision requires reconciliation; no automatic retry")
            if db.execute("SELECT COUNT(*) FROM local_attempts").fetchone()[0] + len(keys) > self.limit:
                raise RouterBudgetExceeded("Branch-wide local router budget exhausted")
            db.executemany("INSERT INTO local_attempts (key, bank) VALUES (?, ?)",
                           [(key, bank) for key in keys])

    def finish(self, key, status, accounting):
        record = {"status": status, "accounting": accounting}
        record["sha256"] = digest(record)
        with self.connection() as db:
            require(db.execute("SELECT result FROM local_attempts WHERE key=?", (key,)).fetchone() == (None,),
                    "Missing/already-completed local reservation")
            db.execute("UPDATE local_attempts SET result=? WHERE key=?", (canonical(record), key))

    def hit(self, key):
        with self.connection() as db:
            row = db.execute("SELECT result FROM local_attempts WHERE key=?", (key,)).fetchone()
            require(row is not None and row[0] is not None and strict_json(row[0])["status"] == "success",
                    "Cache/branch ledger disagreement requires reconciliation")
            db.execute("INSERT INTO local_hits VALUES (?)", (key,))


class BranchCache(RouterCache):
    def __init__(self, path, protocol, limit, ledger):
        super().__init__(path, protocol, limit)
        self.ledger, self.bank, self.keys = ledger, protocol["bank_manifest_sha256"], {}

    def reserve(self, key, visible_input):
        self.ledger.reserve(key, self.bank)
        try:
            attempt = super().reserve(key, visible_input)
        except BaseException as error:
            self.ledger.finish(key, "failed", {"error_type": type(error).__name__, "external_api_calls": 0})
            raise
        self.keys[attempt] = key
        return attempt

    def reserve_many_local(self, items):
        items = list(items)
        keys = [key for key, _ in items]
        self.ledger.reserve_many(keys, self.bank)
        try:
            attempts = super().reserve_many_local(items)
        except BaseException as error:
            for key in keys:
                self.ledger.finish(key, "failed", {"error_type": type(error).__name__, "external_api_calls": 0})
            raise
        self.keys.update({attempt: key for key, attempt in attempts.items()})
        return attempts

    def finish(self, attempt, record):
        super().finish(attempt, record)
        self.ledger.finish(self.keys.pop(attempt), "success", record["response"])

    def fail(self, attempt, safe_error):
        super().fail(attempt, safe_error)
        self.ledger.finish(self.keys.pop(attempt), "failed", safe_error)

    def note_hit(self, key):
        self.ledger.hit(key)
        super().note_hit(key)


class BranchEmbeddingRouter(SkillRLEmbeddingStepRouter):
    version = VERSION
    backend = "phase3_skillrl_embedding_state"

    def _validate_memory(self, memory):
        require(isinstance(memory, BranchMemory) and isinstance(memory.bank, Bank)
                and memory.bank.source == SKILLNET37_MANIFEST_SHA256, "Expected a SkillNet-derived branch snapshot")

    def _validate_candidates(self, candidates):
        require(self.memory.bank.manifest_sha256 == self.protocol["bank_manifest_sha256"], "Bank mutated after router binding")
        super()._validate_candidates(candidates)


class BranchBatchedEmbeddingRouter(BatchedEmbeddingStepRouter):
    """Same frozen top-1 rule, but batch shape and device have new identities."""
    version = BATCH_VERSION
    backend = "phase3_skillrl_embedding_state_batch"

    _validate_memory = BranchEmbeddingRouter._validate_memory

    def _execution_version(self, execution):
        return MICROBATCH_VERSION if execution["mode"] == "state_batch_fp32_micro_v2" else BATCH_VERSION

    def _validate_candidates(self, candidates):
        require(self.memory.bank.manifest_sha256 == self.protocol["bank_manifest_sha256"],
                "Bank mutated after router binding")
        SkillRLEmbeddingStepRouter._validate_candidates(self, candidates)

    def __init__(self, memory, config, *, model_files, model_path, device,
                 cache_path, max_local_calls, execution, _encoder):
        self.version = self._execution_version(execution)
        self._validate_memory(memory)
        self.memory, self._config, self._encoder = memory, config, _encoder
        self._encoder_args = config, model_path, dict(model_files), device
        self._skill_embeddings = None
        import threading
        self._index_lock = threading.RLock()
        self.execution = dict(execution)
        self.protocol = {"version": self.version, "config": asdict(config), "device": device,
            "execution": self.execution, "upstream_commit": UPSTREAM_COMMIT,
            "bank_manifest_sha256": memory.bank.manifest_sha256, "model_files": dict(model_files),
            "catalog": memory.bank.router_catalog(), "skill_texts": skill_texts(memory),
            "query_formatter": self.query_formatter,
            "similarity": "l2-normalized-fp32-dot-product", "tie_break": "first-in-canonical-bank-order",
            "selection_count": 1, "external_api_calls": 0, "automatic_retries": 0}
        self.cache = RouterCache(cache_path, self.protocol, max_local_calls)


class EmbeddingRouterPool:
    """Lazy shared frozen encoder; distinct index/cache per content snapshot."""
    def __init__(self, settings, ledger_path, branch_id, *, _encoder=None):
        config, files = validate_settings(settings)
        self.settings, self.branch_id = dict(settings), branch_id
        self.config, self.files, self.routers = config, files, {}
        execution = batch_execution(settings)
        self.execution = execution
        if execution:
            self.settings["execution"] = execution
        if _encoder is not None:
            self.encoder = _encoder
        elif execution and (settings["device"].startswith("cuda:") or "python_executable" in settings):
            from .gpu_encoder_service import shared_gpu_encoder
            self.encoder = shared_gpu_encoder(model_path=settings["model_path"],
                profile_sha256=settings["profile_sha256"], intra_op_threads=execution["intra_op_threads"],
                shared_gpu_physical_id=execution.get("shared_gpu_physical_id"), ledger_path=ledger_path,
                forward_microbatch_size=execution.get("forward_microbatch_size"),
                **({"python_executable": settings["python_executable"]} if "python_executable" in settings else {}),
                **({"device": "cpu", "runtime_variant": settings.get("runtime_variant")} if settings["device"] == "cpu" else {}))
        elif execution:
            self.encoder = BatchedSentenceEncoder(config, settings["model_path"], files, settings["device"],
                                                  intra_op_threads=execution["intra_op_threads"])
        else:
            self.encoder = FrozenSentenceEncoder(config, settings["model_path"], files, settings["device"])
        ledger_profile = {"version": (MICROBATCH_VERSION if execution.get("mode") == "state_batch_fp32_micro_v2"
                                      else BATCH_VERSION) if execution else VERSION,
            "branch_id": branch_id, "encoder": asdict(config), "model_files": files,
            "device": settings["device"]}
        if execution:
            ledger_profile["execution"] = execution
        if "python_executable" in settings:
            ledger_profile["python_executable"] = settings["python_executable"]
        self.ledger = LocalLedger(ledger_path, ledger_profile, settings["max_local_calls"])

    def for_bank(self, bank):
        require(bank.branch_id == self.branch_id, "Foreign branch cannot share a local budget/cache")
        key = bank.manifest_sha256
        if key not in self.routers:
            path = self.ledger.path.with_suffix(".banks") / f"{key}.sqlite3"
            require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlinked versioned router cache")
            router_type = BranchBatchedEmbeddingRouter if self.execution else BranchEmbeddingRouter
            extra = {"execution": self.execution} if self.execution else {}
            router = router_type(BranchMemory(bank), self.config, model_files=self.files,
                model_path=self.settings["model_path"], device=self.settings["device"], cache_path=path,
                max_local_calls=self.settings["max_local_calls"], _encoder=self.encoder, **extra)
            router.cache = BranchCache(path, router.protocol, self.settings["max_local_calls"], self.ledger)
            self.routers[key] = router
        return self.routers[key]

    def close(self):
        self.encoder.close()
        self.routers.clear()


def local_totals(path):
    """Read-only total across versions, including incomplete/failed attempts."""
    path = Path(path)
    if not path.exists():
        return {"local_attempts": 0, "external_api_calls": 0, "cache_hits": 0, "unreconciled": 0,
                "usage_known_subtotals": {}, "external_api_cost": 0, "scope": "local_embedding_only"}
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        rows = list(db.execute("SELECT bank, result FROM local_attempts"))
        hits = db.execute("SELECT COUNT(*) FROM local_hits").fetchone()[0]
    statuses, usage, seconds = Counter(), Counter(), 0.
    for _, result in rows:
        if result is None:
            statuses["unreconciled"] += 1
            continue
        record = strict_json(result)
        checksum = record.pop("sha256")
        require(digest(record) == checksum, "Changed local router accounting")
        statuses[record["status"]] += 1
        accounting = record["accounting"]
        usage.update(accounting.get("usage", {}))
        seconds += accounting.get("latency_ms", 0.) / 1000
    return {"local_attempts": len(rows), "successful_decisions": statuses["success"], "failed_attempts": statuses["failed"],
            "unreconciled": statuses["unreconciled"], "bank_versions": len({row[0] for row in rows}),
            "cache_hits": hits, "usage_known_subtotals": dict(usage), "latency_seconds": seconds,
            "incomplete_usage_attempts": statuses["failed"] + statuses["unreconciled"],
            "external_api_calls": 0, "external_api_cost": 0,
            "scope": "local_embedding_only; failed/unreconciled compute not inferred as zero"}
