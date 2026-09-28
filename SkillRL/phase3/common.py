"""Small immutable records, strict JSON and non-overwriting persistence."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path


class ProtocolError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def content_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha256(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Expected a SHA-256 identity")
    return value


def positive_int(value, name, *, zero=False):
    require(type(value) is int and value >= (0 if zero else 1), f"Invalid {name}")
    return value


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid {name}")
    return float(value)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ProtocolError("Non-finite JSON number")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, TypeError):
        raise ProtocolError("Invalid strict JSON") from None


def write_new(path, value):
    """Exact idempotent replay only; never replace a prior record or follow links."""
    path = Path(path)
    require(not any(parent.is_symlink() for parent in (path, *path.parents)),
            "Record paths must not contain symlinks")
    data = (canonical(value) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == data:
            return path
        raise FileExistsError("Refusing to overwrite an existing Phase3 record") from None
    fd, temporary = tempfile.mkstemp(prefix=".phase3-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(not path.is_symlink() and path.read_bytes() == data, "Concurrent record publication differs")
    finally:
        os.unlink(temporary)  # Only the freshly allocated private temporary file.
    return path


def safe_label(value):
    if not isinstance(value, str):
        return None
    return "<redacted>" if re.search(r"sk-[A-Za-z0-9_-]{12,}", value) else value[:256]
