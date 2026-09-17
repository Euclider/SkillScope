"""Durable, idempotent manifests and per-episode trajectory archives."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def repo_commit(repo_root: str | Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def repo_worktree_fingerprint(repo_root: str | Path) -> str:
    """Hash tracked diffs plus untracked file contents for exact provenance."""
    root = Path(repo_root)
    digest = hashlib.sha256()
    digest.update(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root))
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
    ).split(b"\0")
    for raw_path in sorted(path for path in untracked if path):
        path = root / os.fsdecode(raw_path)
        if not path.is_file() or raw_path.startswith(b"artifacts/"):
            continue
        digest.update(raw_path)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl_idempotent(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    unique_fields: Sequence[str],
) -> int:
    """Append only unseen keys under an advisory file lock."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with target.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing = set()
        for line in handle:
            if line.strip():
                row = json.loads(line)
                existing.add(tuple(jsonable(row.get(field)) for field in unique_fields))
        handle.seek(0, os.SEEK_END)
        for record in records:
            serial = jsonable(record)
            key = tuple(serial.get(field) for field in unique_fields)
            if key in existing:
                continue
            handle.write(json.dumps(serial, ensure_ascii=False, sort_keys=True) + "\n")
            existing.add(key)
            appended += 1
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return appended


def write_run_manifest(
    output_dir: str | Path,
    run_id: str,
    config: Any,
    repo_root: str | Path,
    skill_bank_path: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    config_value = jsonable(config)
    payload = {
        "run_id": run_id,
        "created_at": utc_now(),
        "repo_commit": repo_commit(repo_root),
        "repo_worktree_fingerprint": repo_worktree_fingerprint(repo_root),
        "repo_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=repo_root, text=True
        ).splitlines(),
        "config": config_value,
        "config_hash": stable_hash(config_value),
        "skill_bank_path": str(skill_bank_path) if skill_bank_path else None,
        "skill_bank_hash": sha256_file(skill_bank_path) if skill_bank_path else None,
        **jsonable(extra or {}),
    }
    path = Path(output_dir) / "manifests" / f"{run_id}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != payload["config_hash"]:
            raise ValueError(f"run_id {run_id!r} already exists with a different config")
        return path
    atomic_write_json(path, payload)
    return path


def _token_count(step: Mapping[str, Any], key: str) -> int:
    value = jsonable(step.get(key, []))
    if not isinstance(value, list):
        return 0
    if key == "attention_mask":
        return sum(int(item) for item in value)
    return len(value)


def archive_rollout_batch(
    *,
    output_dir: str | Path,
    run_id: str,
    split: str,
    global_step: int,
    total_batch_list: Sequence[Sequence[Mapping[str, Any]]],
    total_infos: Sequence[Sequence[Mapping[str, Any]]],
    episode_rewards: Sequence[Any],
    episode_lengths: Sequence[Any],
    trajectory_ids: Sequence[Any],
) -> int:
    """Write one lossless JSON document per episode and an idempotent index."""
    root = Path(output_dir)
    index_rows = []
    for index, (batch_steps, info_steps) in enumerate(zip(total_batch_list, total_infos)):
        trajectory_id = str(trajectory_ids[index])
        steps = []
        all_retrieved: set[str] = set()
        all_candidates: set[str] = set()
        all_injected: set[str] = set()
        selected_ids: list[str] = []
        router_versions: set[str] = set()
        game_id = None
        context_id = None
        for step_index, (batch_step, info) in enumerate(zip(batch_steps, info_steps)):
            if not bool(jsonable(batch_step.get("active_masks", True))):
                continue
            retrieved = list(info.get("retrieved_skill_ids", []))
            candidates = list(info.get("candidate_skill_ids", []))
            injected = list(info.get("injected_skill_ids", []))
            selected = info.get("selected_skill_id")
            all_retrieved.update(retrieved)
            all_candidates.update(candidates)
            all_injected.update(injected)
            if selected:
                selected_ids.append(str(selected))
            if info.get("skill_router_version"):
                router_versions.add(str(info["skill_router_version"]))
            game_id = game_id or info.get("extra.gamefile")
            context_id = context_id or info.get("skill_task_type")
            steps.append({
                "step_index": step_index,
                "prompt_text": info.get("prompt_text"),
                "task_description": info.get("task_description"),
                "observation": info.get("observation"),
                "admissible_actions": info.get("admissible_actions", []),
                "raw_model_output": info.get("raw_model_output"),
                "projected_action": info.get("projected_action"),
                "is_action_valid": bool(jsonable(info.get("is_action_valid", False))),
                "reward": jsonable(batch_step.get("rewards")),
                "next_observation": info.get("next_observation"),
                "retrieved_skill_ids": retrieved,
                "candidate_skill_ids": candidates,
                "injected_skill_ids": injected,
                "disabled_skill_ids": list(info.get("disabled_skill_ids", [])),
                "selected_skill_id": selected,
                "skill_router_version": info.get("skill_router_version"),
                "skill_router_scores": info.get("skill_router_scores", {}),
                "skill_router_score_details": info.get(
                    "skill_router_score_details", {}
                ),
                "skill_router_state_flags": list(
                    info.get("skill_router_state_flags", [])
                ),
                "skill_router_selection_reason": info.get(
                    "skill_router_selection_reason"
                ),
                "prompt_tokens": _token_count(batch_step, "attention_mask"),
                "completion_tokens": _token_count(batch_step, "responses"),
            })
        trajectory_path = root / "trajectories" / run_id / f"{trajectory_id}.json"
        payload = {
            "schema_version": (
                "phase1.trajectory.v2" if router_versions
                else "phase1.trajectory.v1"
            ),
            "run_id": run_id,
            "trajectory_id": trajectory_id,
            "split": split,
            "global_step": int(global_step),
            "game_id": game_id,
            "context_id": context_id,
            "episode_return": jsonable(episode_rewards[index]),
            "trajectory_length": int(jsonable(episode_lengths[index])),
            "retrieved_skill_ids": sorted(all_retrieved),
            "candidate_skill_ids": sorted(all_candidates),
            "injected_skill_ids": sorted(all_injected),
            "unique_skill_count": len(all_injected),
            "selected_skill_ids": sorted(set(selected_ids)),
            "skill_selection_counts": dict(sorted(Counter(selected_ids).items())),
            "skill_router_versions": sorted(router_versions),
            "created_at": utc_now(),
            "steps": steps,
        }
        atomic_write_json(trajectory_path, payload)
        index_rows.append({
            key: payload[key]
            for key in (
                "run_id", "trajectory_id", "split", "global_step", "game_id",
                "context_id", "episode_return", "trajectory_length",
                "retrieved_skill_ids", "candidate_skill_ids",
                "injected_skill_ids", "unique_skill_count",
                "selected_skill_ids", "skill_selection_counts",
                "skill_router_versions",
            )
        } | {"trajectory_path": str(trajectory_path)})
    return append_jsonl_idempotent(
        root / "trajectories" / "index.jsonl",
        index_rows,
        unique_fields=("run_id", "trajectory_id"),
    )


def archive_training_step(
    *,
    output_dir: str | Path,
    run_id: str,
    global_step: int,
    batch: Any,
    metrics: Mapping[str, Any],
    update_type: str,
) -> Path:
    """Archive the reward/advantage provenance used by one policy update."""
    batch_size = int(batch.batch.batch_size[0])
    response_mask = batch.batch.get("response_mask")
    trajectory_ids = batch.non_tensor_batch.get("traj_uid", [None] * batch_size)
    group_ids = batch.non_tensor_batch.get("uid", [None] * batch_size)
    rows = []
    for row_index in range(batch_size):
        mask = response_mask[row_index].bool() if response_mask is not None else None
        row = {
            "row_index": row_index,
            "group_id": jsonable(group_ids[row_index]),
            "trajectory_id": jsonable(trajectory_ids[row_index]),
        }
        for field in ("token_level_scores", "token_level_rewards", "returns", "advantages"):
            if field not in batch.batch:
                continue
            values = batch.batch[field][row_index]
            if mask is not None and values.shape == mask.shape:
                values = values[mask]
            values = values.detach().float().cpu()
            row[f"{field}_sum"] = float(values.sum().item())
            row[f"{field}_mean"] = float(values.mean().item()) if values.numel() else None
            row[f"{field}_min"] = float(values.min().item()) if values.numel() else None
            row[f"{field}_max"] = float(values.max().item()) if values.numel() else None
        rows.append(row)
    payload = {
        "schema_version": "phase1.training_step.v1",
        "run_id": run_id,
        "global_step": int(global_step),
        "update_type": update_type,
        "created_at": utc_now(),
        "metrics": jsonable(metrics),
        "trajectory_update_summaries": rows,
    }
    target = Path(output_dir) / "training_steps" / run_id / f"step-{global_step:06d}.json"
    atomic_write_json(target, payload)
    append_jsonl_idempotent(
        Path(output_dir) / "training_steps" / "index.jsonl",
        [{
            "run_id": run_id,
            "global_step": int(global_step),
            "update_type": update_type,
            "path": str(target),
            "actor_grad_norm": jsonable(metrics.get("actor/grad_norm")),
        }],
        unique_fields=("run_id", "global_step"),
    )
    return target
