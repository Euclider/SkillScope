"""Persistent, process-safe single-decision cache and bounded API audit ledger.

Use one local-filesystem database per cohort/router protocol. Per-input flock
locks prevent duplicate paid calls, while different inputs can run concurrently.
API reservations are committed before sending, including failed/crashed calls.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class RouterCacheError(ValueError):
    pass


class RouterBudgetExceeded(RouterCacheError):
    pass


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RouterCache:
    def __init__(self, path: str | Path, protocol: dict, max_api_calls: int):
        if type(max_api_calls) is not int or max_api_calls < 0:
            raise RouterCacheError("max_api_calls must be an explicit nonnegative integer")
        self.path = Path(path)
        if self.path.is_symlink():
            raise RouterCacheError("Router cache cannot be a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_dir = self.path.with_name(self.path.name + ".locks")
        if self.lock_dir.is_symlink():
            raise RouterCacheError("Router lock directory cannot be a symlink")
        self.lock_dir.mkdir(mode=0o700, exist_ok=True)
        self.protocol_hash = digest(protocol)
        self.max_api_calls = max_api_calls
        with self._transaction() as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not tables:
                connection.execute("CREATE TABLE protocol (id INTEGER PRIMARY KEY CHECK (id=1), hash TEXT NOT NULL, data TEXT NOT NULL)")
                connection.execute("CREATE TABLE decisions (key TEXT PRIMARY KEY, record TEXT NOT NULL, hash TEXT NOT NULL)")
                connection.execute("CREATE TABLE attempts (id INTEGER PRIMARY KEY, key TEXT NOT NULL, started TEXT NOT NULL, status TEXT NOT NULL, data TEXT NOT NULL)")
                connection.execute("CREATE TABLE cache_hits (id INTEGER PRIMARY KEY, key TEXT NOT NULL, at TEXT NOT NULL)")
                connection.execute("INSERT INTO protocol VALUES (1, ?, ?)", (self.protocol_hash, canonical_json(protocol)))
            elif tables != {"protocol", "decisions", "attempts", "cache_hits"}:
                raise RouterCacheError("Unexpected or incomplete router cache schema")
            row = connection.execute("SELECT hash, data FROM protocol WHERE id=1").fetchone()
            if row != (self.protocol_hash, canonical_json(protocol)):
                raise RouterCacheError("Cache protocol mismatch; use a new cohort/cache path")

    @contextmanager
    def _transaction(self):
        connection = sqlite3.connect(str(self.path), timeout=60)
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def input_lock(self, key: str):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise RouterCacheError("Invalid cache key")
        descriptor = os.open(self.lock_dir / (key + ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def lookup(self, key: str):
        with self._transaction() as connection:
            row = connection.execute("SELECT record, hash FROM decisions WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        try:
            record = json.loads(row[0])
            valid = digest(record) == row[1] and record["cache_key"] == key and record["protocol_hash"] == self.protocol_hash
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            raise RouterCacheError("Cached decision failed integrity validation; no API fallback")
        return record

    def note_hit(self, key: str):
        with self._transaction() as connection:
            connection.execute("INSERT INTO cache_hits (key, at) VALUES (?, ?)", (key, utc_now()))

    def reserve(self, key: str, visible_input: dict) -> int:
        with self._transaction() as connection:
            count = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            if count >= self.max_api_calls:
                raise RouterBudgetExceeded("Router API-call budget exhausted; cached decisions remain usable")
            cursor = connection.execute(
                "INSERT INTO attempts (key, started, status, data) VALUES (?, ?, 'started', ?)",
                (key, utc_now(), canonical_json({"visible_input": visible_input})),
            )
            return cursor.lastrowid

    def finish(self, attempt: int, record: dict):
        with self._transaction() as connection:
            row = connection.execute("SELECT key, status FROM attempts WHERE id=?", (attempt,)).fetchone()
            if row != (record["cache_key"], "started"):
                raise RouterCacheError("Invalid API reservation")
            # Never replace an existing decision. input_lock serializes its creation.
            connection.execute("INSERT INTO decisions VALUES (?, ?, ?)", (record["cache_key"], canonical_json(record), digest(record)))
            connection.execute("UPDATE attempts SET status='success', data=? WHERE id=?", (canonical_json(record), attempt))

    def reserve_many_local(self, items):
        """Atomic local-query admission; caller holds sorted per-input locks.

        Unlike legacy single-call reservations, a failed/incomplete batch query
        cannot be submitted again. Never use this to bypass an API cost guard.
        """
        items = list(items)
        if len({key for key, _ in items}) != len(items):
            raise RouterCacheError("Duplicate batch reservation")
        with self._transaction() as connection:
            count = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            if count + len(items) > self.max_api_calls:
                raise RouterBudgetExceeded("Local embedding-call budget exhausted before batch forward")
            for key, _ in items:
                if connection.execute("SELECT 1 FROM attempts WHERE key=?", (key,)).fetchone():
                    raise RouterCacheError("Prior failed/incomplete local query; no automatic retry")
            result = {}
            for key, visible in items:
                cursor = connection.execute(
                    "INSERT INTO attempts (key, started, status, data) VALUES (?, ?, 'started', ?)",
                    (key, utc_now(), canonical_json({"visible_input": visible})),
                )
                result[key] = cursor.lastrowid
            return result

    def fail(self, attempt: int, safe_error: dict):
        with self._transaction() as connection:
            # Keep the reserved visible input; never store exception text/headers.
            row = connection.execute("SELECT data FROM attempts WHERE id=?", (attempt,)).fetchone()
            data = json.loads(row[0])
            data["failure"] = safe_error
            connection.execute("UPDATE attempts SET status='failed', data=? WHERE id=? AND status='started'", (canonical_json(data), attempt))

    def stats(self) -> dict:
        with self._transaction() as connection:
            return {
                "api_attempts": connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
                "successful_decisions": connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0],
                "failed_attempts": connection.execute("SELECT COUNT(*) FROM attempts WHERE status='failed'").fetchone()[0],
                "cache_hits": connection.execute("SELECT COUNT(*) FROM cache_hits").fetchone()[0],
                "max_api_calls": self.max_api_calls,
            }
