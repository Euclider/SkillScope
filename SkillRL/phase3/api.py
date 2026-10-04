"""Bounded OpenAI-compatible JSON requests; no secret or exception-body logging."""
from __future__ import annotations

import importlib.metadata
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .common import ProtocolError, canonical, digest, positive_int, require, safe_label, sha256, strict_json


# Only fixed local validation messages may be logged; never exception bodies
# from the gateway or arbitrary model text.
SAFE_CONSTRAINTS = frozenset({
    'Incomplete API response', 'API refusal or tools', 'Duplicate JSON key',
    'Non-finite JSON number', 'Invalid operation budget/schema', 'Invalid operation count',
    'Unexpected operation fields', 'Unknown operation or missing rationale',
    'Unregistered edit evidence', 'Mutations require evidence references',
    'Targets must be versioned IDs', 'Invalid edit target', 'Unknown/repeated edit target',
    'Stale edit target', 'Wrong target cardinality', 'Normalized mutation budget exceeded',
    'DELETE/NOOP cannot create content', 'Invalid skill content', 'Empty editor content',
    'New IDs must not be reused', 'NOOP must be the only operation',
    'Editor targeted an unexposed skill', 'Empty or duplicate active skills',
})


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
        require(0 < self.timeout_seconds <= (600 if self.stage == "editor" else 60), "Invalid request timeout")

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
            db.execute("CREATE TABLE IF NOT EXISTS retry_authorizations (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS transport_timeout (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
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

    @staticmethod
    def _checked_result(raw):
        require(raw is not None, "Ambiguous prior API call; manual reconciliation required")
        record = strict_json(raw)
        checksum = record.pop("record_sha256", None)
        require(checksum == digest(record), "Changed API result ledger")
        return record

    def authorize_timeout_retry(self, request_key, *, timeout_seconds=600):
        """Persist one explicit retry of an editor timeout; preserve its attempt.

        The retry shares the original call budget. A second failure remains
        terminal, and ordinary resumes may only reuse this same authorization.
        """
        return self._authorize_retry(request_key, timeout_seconds=timeout_seconds, response_retry=False)

    def authorize_response_retry(self, request_key, *, timeout_seconds=600):
        """One explicitly approved retry of a rejected response; no rule changes."""
        return self._authorize_retry(request_key, timeout_seconds=timeout_seconds, response_retry=True)

    def _authorize_retry(self, request_key, *, timeout_seconds, response_retry):
        sha256(request_key)
        require(self.config.stage == "editor", "Only editor timeout recovery is supported")
        require(type(timeout_seconds) in (int, float) and 0 < timeout_seconds <= 600,
                "Invalid retry timeout")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT request, result FROM attempts WHERE key=?", (request_key,)).fetchone()
            require(row is not None, "Unknown API request for timeout recovery")
            request = strict_json(row[0])
            require(digest(request) == request_key and request["profile"] == asdict(self.config),
                    "Changed API request/profile")
            require("explicit_retry" not in request, "A second retry requires separate reconciliation")
            record = self._checked_result(row[1])
            failure_type = 'ProtocolError' if response_retry else 'APITimeoutError'
            require(record["status"] == "failed" and record.get("failure", {}).get("type") == failure_type,
                    "Explicit recovery requires a recorded API timeout")
            if response_retry:
                require(bool(record['accounting'].get('completion_id')), 'No received response to reconcile')
            receipt = {"schema_version": ("skillrl.phase3.editor_response_retry.v1" if response_retry
                                           else "skillrl.phase3.editor_timeout_retry.v1"),
                       "original_request_sha256": request_key, "original_result_sha256": digest(record),
                       "timeout_seconds": float(timeout_seconds), "additional_attempts": 1,
                       "reason": ("user_authorized_resume_after_rejected_editor_response" if response_retry
                                  else "user_authorized_resume_after_editor_timeout"),
                       "authorized_utc": datetime.now(timezone.utc).isoformat()}
            existing = db.execute("SELECT value FROM retry_authorizations WHERE key=?", (request_key,)).fetchone()
            if existing:
                saved = strict_json(existing[0])
                require({k: v for k, v in saved.items() if k != "authorized_utc"}
                        == {k: v for k, v in receipt.items() if k != "authorized_utc"},
                        "Changed retry authorization")
                return saved
            db.execute("INSERT INTO retry_authorizations VALUES (?, ?)", (request_key, canonical(receipt)))
        return receipt

    def authorize_transport_timeout(self, *, timeout_seconds=600):
        """Explicit operational override, without changing historical request keys.

        This changes waiting time only. Failed calls still require a separate
        one-shot retry authorization and every call retains the original cap.
        """
        require(self.config.stage == 'editor', 'Only editor transport timeout may be extended')
        require(type(timeout_seconds) in (int, float)
                and self.config.timeout_seconds <= timeout_seconds <= 600, 'Invalid transport timeout')
        receipt = {'schema_version': 'skillrl.phase3.transport_timeout.v1',
                   'profile_sha256': digest(asdict(self.config)), 'timeout_seconds': float(timeout_seconds),
                   'authorized_utc': datetime.now(timezone.utc).isoformat(),
                   'reason': 'user_authorized_editor_timeout_extension'}
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute('SELECT value FROM transport_timeout WHERE id=1').fetchone()
            if existing:
                saved = self._checked_result(existing[0])
                require({k: v for k, v in saved.items() if k != 'authorized_utc'}
                        == {k: v for k, v in receipt.items() if k != 'authorized_utc'},
                        'Changed transport timeout authorization')
                return saved
            db.execute('INSERT INTO transport_timeout VALUES (1, ?)',
                       (canonical({**receipt, 'record_sha256': digest(receipt)}),))
        return receipt

    def request(self, *, identity, system, payload, schema, validate):
        request = {"identity": identity, "system": system, "payload": payload, "schema": schema,
                   "profile": asdict(self.config)}
        key = digest(request)
        with self.connection() as db:
            saved = db.execute("SELECT result FROM attempts WHERE key=?", (key,)).fetchone()
            authorization = db.execute("SELECT value FROM retry_authorizations WHERE key=?", (key,)).fetchone()
            transport_row = db.execute('SELECT value FROM transport_timeout WHERE id=1').fetchone()
        transport = self._checked_result(transport_row[0]) if transport_row else None
        if transport:
            require(self.config.stage == 'editor'
                    and transport['schema_version'] == 'skillrl.phase3.transport_timeout.v1'
                    and transport['profile_sha256'] == digest(asdict(self.config))
                    and self.config.timeout_seconds <= transport['timeout_seconds'] <= 600,
                    'Changed transport timeout profile')
        retry = None
        if authorization:
            require(saved is not None, "Missing original retry attempt")
            original = self._checked_result(saved[0])
            retry = strict_json(authorization[0])
            expected_type = {'skillrl.phase3.editor_timeout_retry.v1': 'APITimeoutError',
                             'skillrl.phase3.editor_response_retry.v1': 'ProtocolError'}.get(retry.get('schema_version'))
            require(self.config.stage == "editor" and original["status"] == "failed"
                    and expected_type is not None and original.get("failure", {}).get("type") == expected_type
                    and retry["original_request_sha256"] == key
                    and retry["original_result_sha256"] == digest(original)
                    and retry["additional_attempts"] == 1 and 0 < retry["timeout_seconds"] <= 600,
                    "Changed timeout recovery authorization")
            request["explicit_retry"] = retry
            key = digest(request)
            with self.connection() as db:
                saved = db.execute("SELECT result FROM attempts WHERE key=?", (key,)).fetchone()
        if saved:
            record = self._checked_result(saved[0])
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
        if transport:
            kwargs['timeout'] = transport['timeout_seconds']
        if retry:
            kwargs["timeout"] = retry["timeout_seconds"]
        failure, value = None, None
        phase, diagnostic = 'transport', {}
        try:
            response = client.chat.completions.create(**kwargs)
            phase = 'completion'
            diagnostic['choice_count'] = len(response.choices)
            diagnostic['finish_reason'] = safe_label(response.choices[0].finish_reason) if response.choices else None
            require(len(response.choices) == 1 and response.choices[0].finish_reason == "stop", "Incomplete API response")
            message = response.choices[0].message
            phase = 'message'
            require(not getattr(message, "refusal", None) and not getattr(message, "tool_calls", None), "API refusal or tools")
            phase = 'json'
            if isinstance(message.content, str):
                diagnostic['content_sha256'] = digest(message.content)
                diagnostic['content_characters'] = len(message.content)
            value = strict_json(message.content)
            phase = 'validation'
            validate(value)
        except Exception as error:
            status = getattr(error, "status_code", None)
            failure = {"type": type(error).__name__, "http_status": status if type(status) is int else None}
            failure['phase'] = phase
            if isinstance(error, ProtocolError) and str(error) in SAFE_CONSTRAINTS:
                failure['constraint'] = str(error)
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
        if transport:
            accounting.update(effective_timeout_seconds=kwargs['timeout'],
                              transport_timeout_sha256=digest(transport))
        if retry:
            accounting.update(explicit_retry_of=retry["original_request_sha256"],
                              effective_timeout_seconds=retry["timeout_seconds"])
        record = {"status": "failed" if failure else "success", "value": None if failure else value,
                  "accounting": accounting, "failure": failure}
        if failure:
            record['response_diagnostic'] = diagnostic
        record["record_sha256"] = digest(record)
        with self.connection() as db:
            db.execute("UPDATE attempts SET result=? WHERE key=?", (canonical(record), key))
        if failure:
            raise ProtocolError(f"External {self.config.stage} request failed ({failure['type']}); see sanitized ledger, no retry") from None
        return value, accounting

    def close(self):
        if self.client is not None and hasattr(self.client, "close"):
            self.client.close()
