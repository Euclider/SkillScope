"""Bounded OpenAI-compatible JSON requests; no secret or exception-body logging."""
from __future__ import annotations

import importlib.metadata
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .common import ProtocolError, canonical, digest, positive_int, require, safe_label, strict_json


@dataclass(frozen=True)
class APIConfig:
    stage: str
    model: str
    max_input_tokens: int
    max_completion_tokens: int
    max_api_calls: int
    base_url: str = "https://api.zhizengzeng.com/v1"
    sdk_version: str = "3.15.0"
    timeout_seconds: float = 60.

    def __post_init__(self):
        require((self.stage, self.model) in (("editor", "o3"), ("editor", "gpt-5.5"),
                                              ("router", "gpt-5.4-mini")), "Unregistered stage/model")
        url = urlsplit(self.base_url)
        require(url.scheme == "https" and url.hostname and not any((url.username, url.password, url.query, url.fragment))
                and url.path.rstrip("/") == "/v1", "Expected a credential-free HTTPS /v1 endpoint")
        for key in ("max_input_tokens", "max_completion_tokens"):
            positive_int(getattr(self, key), key)
        positive_int(self.max_api_calls, "API budget", zero=True)
        require(0 < self.timeout_seconds <= 60, "Invalid request timeout")

    @property
    def key_env(self):
        return "SKILLRL_PHASE3_EDITOR_API_KEY" if self.stage == "editor" else "SKILLNET_ROUTER_API_KEY"


class JSONClient:
    """Per-branch, per-stage total budget shared by every evolving bank version.

    A failed/ambiguous reservation cannot be resent automatically. Resume reuses
    completed responses, not partially completed calls. SQLite prevents two
    processes spending the same call slot or duplicating the same request.
    """
    def __init__(self, config: APIConfig, ledger_path, *, allow_live=False, client=None, token_counter=None):
        self.config, self.allow_live, self.client = config, allow_live, client
        self.token_counter = token_counter
        self.path = Path(ledger_path)
        require(not any(path.is_symlink() for path in (self.path, *self.path.parents)), "Symlinked API ledger")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS profile (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL, request TEXT NOT NULL, result TEXT)")
            profile = canonical(asdict(config))
            db.execute("INSERT OR IGNORE INTO profile VALUES (1, ?)", (profile,))
            require(db.execute("SELECT value FROM profile WHERE id=1").fetchone() == (profile,), "Changed API profile/budget")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(str(self.path), timeout=60)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _client(self):
        if self.client is None:
            require(self.allow_live is True, "Live API execution is not authorized")
            require(importlib.metadata.version("openai") == self.config.sdk_version, "OpenAI SDK version mismatch")
            key = os.environ.get(self.config.key_env, "")
            require(bool(key.strip()), f"Set dedicated environment variable {self.config.key_env}; no key fallback")
            import openai
            self.client = openai.OpenAI(base_url=self.config.base_url, api_key=key,
                                       timeout=self.config.timeout_seconds, max_retries=0,
                                       http_client=openai.DefaultHttpxClient(follow_redirects=False))
        return self.client

    def request(self, *, identity, system, payload, schema, validate):
        request = {"identity": identity, "system": system, "payload": payload, "schema": schema,
                   "profile": asdict(self.config)}
        key = digest(request)
        with self.connection() as db:
            saved = db.execute("SELECT result FROM attempts WHERE key=?", (key,)).fetchone()
        if saved:
            require(saved[0] is not None, "Ambiguous prior API call; manual reconciliation required")
            record = strict_json(saved[0])
            checksum = record.pop("record_sha256", None)
            require(checksum == digest(record), "Changed API result ledger")
            require(record["status"] == "success", "Prior API failure; automatic retry is prohibited")
            validate(record["value"])
            return record["value"], {**record["accounting"], "cache_hit": True, "api_calls": 0,
                                      "latency_seconds": 0.,
                                      "usage": {name: 0 for name in record["accounting"]["usage"]}}
        client = self._client()
        if self.token_counter is None:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.config.model)
            self.token_counter = lambda text: len(encoding.encode(text, disallowed_special=()))
        # Provider chat/schema framing is not fully observable. Keep the
        # estimate for accounting; the editor's historical input cap is not
        # enforced. The router retains its separate local input protection.
        estimate = self.token_counter(canonical({"system": system, "payload": payload, "schema": schema})) + 256
        if self.config.stage == "router":
            require(estimate <= self.config.max_input_tokens, "Router input estimate exceeds frozen token cap")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            require(db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] < self.config.max_api_calls, "API budget exhausted")
            try:
                db.execute("INSERT INTO attempts (key, request) VALUES (?, ?)", (key, canonical(request)))
            except sqlite3.IntegrityError:
                raise ProtocolError("Identical API request already reserved by another process") from None
        started, response = time.monotonic(), None
        kwargs = {"model": self.config.model, "messages": [{"role": "system", "content": system},
                                                            {"role": "user", "content": canonical(payload)}],
                  "response_format": {"type": "json_schema", "json_schema": {"name": "phase3_response", "strict": True, "schema": schema}},
                  "max_completion_tokens": self.config.max_completion_tokens, "n": 1, "stream": False, "store": False,
                  "reasoning_effort": "medium" if self.config.stage == "editor" else "none"}
        if self.config.stage == "router":
            kwargs["temperature"] = 0
        failure, value = None, None
        try:
            response = client.chat.completions.create(**kwargs)
            require(len(response.choices) == 1 and response.choices[0].finish_reason == "stop", "Incomplete API response")
            message = response.choices[0].message
            require(not getattr(message, "refusal", None) and not getattr(message, "tool_calls", None), "API refusal or tools")
            value = strict_json(message.content)
            validate(value)
        except Exception as error:
            status = getattr(error, "status_code", None)
            failure = {"type": type(error).__name__, "http_status": status if type(status) is int else None}
        from agent_system.memory.external_skill_router import _usage
        usage = _usage(response)
        if (self.config.stage == "router" and usage["prompt_tokens"] is not None
                and usage["prompt_tokens"] > self.config.max_input_tokens):
            failure = {"type": "ReportedInputTokenCapExceeded", "http_status": None}
        accounting = {"stage": self.config.stage, "requested_model": self.config.model,
                      "response_model": safe_label(getattr(response, "model", None)),
                      "request_id": safe_label(getattr(response, "_request_id", None)),
                      "completion_id": safe_label(getattr(response, "id", None)),
                      "system_fingerprint": safe_label(getattr(response, "system_fingerprint", None)),
                      "usage": usage, "input_token_estimate": estimate,
                      "latency_seconds": time.monotonic() - started, "provider_cost": None,
                      "api_calls": 1, "cache_hit": False, "request_sha256": key}
        record = {"status": "failed" if failure else "success", "value": None if failure else value,
                  "accounting": accounting, "failure": failure}
        record["record_sha256"] = digest(record)
        with self.connection() as db:
            db.execute("UPDATE attempts SET result=? WHERE key=?", (canonical(record), key))
        if failure:
            raise ProtocolError(f"External {self.config.stage} request failed ({failure['type']}); see sanitized ledger, no retry") from None
        return value, accounting

    def close(self):
        if self.client is not None and hasattr(self.client, "close"):
            self.client.close()
