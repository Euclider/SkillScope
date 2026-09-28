"""Immutable artifacts and explicit, scoped execution gates."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCHEMA = "skillnet.phase12.v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new_json(path, value):
    """Atomic no-clobber publication; an identical existing artifact is reusable."""
    content = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    return write_new_bytes(path, content)


def write_new_bytes(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # link() publishes an already-fsynced file without replacing a concurrent writer.
    import tempfile
    descriptor, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise
    finally:
        Path(temporary).unlink(missing_ok=True)  # Only our own temporary publication file.


def checked_relative(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or path == root or not path.is_relative_to(root):
        raise ValueError("Artifact path escapes its declared root")
    return path


def load_preparation(path, verify=True):
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get("schema_version") != SCHEMA or manifest.get("kind") != "preparation":
        raise ValueError("Not a SkillNet Phase1/2 preparation manifest")
    if verify:
        for row in manifest["assets"]:
            if file_hash(checked_relative(path.parent, row["path"])) != row["sha256"]:
                raise ValueError(f"Changed preparation asset: {row['path']}")
        from .assets import validate_controls
        validate_controls(path)
        from .day_budget import validate_budget_binding
        validate_budget_binding(read_json(path.parent / 'spec.json'))
        from .inference import validate
        validate(read_json(path.parent / 'spec.json'))
    return manifest


def require_authorization(path, preparation_path, operation):
    """Never inferred from configuration validity, environment variables, or tests."""
    if path is None:
        raise PermissionError("An explicit execution authorization is required")
    override = os.environ.get('SKILLNET_PHASE12_RECOVERY_AUTHORIZATION')
    if override and Path(path).resolve() != Path(override).resolve():
        replacement = read_json(override)
        recovery = replacement.get('readout_recovery', {})
        if str(Path(path).resolve()) == recovery.get('original_authorization'):
            if (operation not in ('evaluation', 'readout')
                    or file_hash(path) != recovery.get('original_authorization_sha256')
                    or replacement.get('preparation_sha256') != file_hash(preparation_path)
                    or str(Path(replacement.get('run_root', '')).resolve()) != recovery.get('source_root')):
                raise PermissionError('Invalid readout-continuation authorization replacement')
            path = override  # Original protocol and permit bytes remain immutable.
    value = read_json(path)
    if value.get("preparation_sha256") != file_hash(preparation_path):
        raise PermissionError("Authorization belongs to another preparation")
    if value.get("approved") is not True or operation not in value.get("operations", []):
        raise PermissionError(f"Execution is not approved for {operation}")
    spec = read_json(Path(preparation_path).resolve().parent / "spec.json")
    if spec.get('budget_profile'):
        from .day_budget import validate_admission
        # Each stage must retain the same evidence and absolute deadline, but
        # only supervisor admission needs enough time for the ENTIRE pipeline.
        validate_admission(value, spec, preparation_path, require_full_budget=False)
    if operation in {"training", "evaluation"}:
        from .runtime import is_embedding_backend, router_backend
        local = is_embedding_backend(router_backend(spec))
        field = "router_max_local_calls" if local else "router_max_api_calls"
        limit = value.get(field, 0)
        if type(limit) is not int or limit <= 0:
            raise PermissionError(f"A positive, explicit {field} budget is required")
        if local and value.get("router_max_api_calls", 0) != 0:
            raise PermissionError("Embedding authorization must not permit paid router calls")
        if not local and value.get("router_max_local_calls", 0) != 0:
            raise PermissionError("Local-call budget cannot authorize an API router")
    return value


@contextmanager
def exclusive_writer(output):
    """Fail, rather than duplicate paid work, if another evaluator owns the path."""
    import fcntl
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Evaluation output already has an active writer") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
