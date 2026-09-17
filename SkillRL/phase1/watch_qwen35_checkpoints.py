#!/usr/bin/env python3
"""Archive formal Qwen3.5 milestones and retain two resumable checkpoints.

The trainer's tracker file is written only after all FSDP shards and data.pt
have been saved.  This watcher therefore treats the tracker as the commit
marker, validates the committed checkpoint, exports analysis milestones via a
temporary directory, verifies the HF artifact, and finally rotates older full
checkpoints.  It never reads validation, Skill utility, or action-flip data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


MIN_FULL_CHECKPOINT_BYTES = 40_000_000_000
MIN_MODEL_ONLY_BYTES = 8_000_000_000


def log(message: str) -> None:
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def validate_full_checkpoint(path: Path) -> dict[str, int]:
    actor = path / "actor"
    model_pattern = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")
    model_matches = []
    for shard in actor.glob("model_world_size_*_rank_*.pt"):
        match = model_pattern.match(shard.name)
        if match:
            model_matches.append((int(match.group(1)), int(match.group(2)), shard))
    world_sizes = {world_size for world_size, _, _ in model_matches}
    if len(world_sizes) != 1:
        raise RuntimeError(f"cannot infer one FSDP world size in {path}: {sorted(world_sizes)}")
    world_size = world_sizes.pop()
    expected_ranks = set(range(world_size))
    actual_ranks = {rank for _, rank, _ in model_matches}
    if actual_ranks != expected_ranks:
        raise RuntimeError(f"incomplete model ranks in {path}: {sorted(actual_ranks)} vs {sorted(expected_ranks)}")
    required_groups = {
        "model_shards": [item for _, _, item in model_matches],
        "optimizer_shards": list(actor.glob(f"optim_world_size_{world_size}_rank_*.pt")),
        "extra_state_shards": list(actor.glob(f"extra_state_world_size_{world_size}_rank_*.pt")),
    }
    missing = [name for name, files in required_groups.items() if len(files) != world_size]
    if missing:
        raise RuntimeError(f"incomplete shard groups in {path}: {', '.join(missing)}")
    if not (path / "data.pt").is_file():
        raise RuntimeError(f"missing data.pt in {path}")
    if not (actor / "config.json").is_file():
        raise RuntimeError(f"missing actor/config.json in {path}")
    total_bytes = checkpoint_size(path)
    if total_bytes < MIN_FULL_CHECKPOINT_BYTES:
        raise RuntimeError(f"checkpoint too small: {total_bytes} bytes in {path}")
    return {"total_bytes": total_bytes, "world_size": world_size, **{name: len(files) for name, files in required_groups.items()}}


def validate_model_only(path: Path) -> dict[str, int | str]:
    model = path / "model.safetensors"
    config = path / "config.json"
    if not model.is_file() or model.stat().st_size < MIN_MODEL_ONLY_BYTES:
        raise RuntimeError(f"missing or undersized model.safetensors in {path}")
    if not config.is_file():
        raise RuntimeError(f"missing config.json in {path}")
    with config.open(encoding="utf-8") as handle:
        config_data = json.load(handle)
    model_type = config_data.get("model_type")
    if model_type != "qwen3_5_text":
        raise RuntimeError(f"unexpected model_type {model_type!r} in {config}")
    from safetensors import safe_open

    with safe_open(model, framework="pt", device="cpu") as handle:
        tensor_count = len(handle.keys())
    if tensor_count < 400:
        raise RuntimeError(f"unexpected tensor count {tensor_count} in {model}")
    return {
        "model_bytes": model.stat().st_size,
        "tensor_count": tensor_count,
        "model_type": model_type,
    }


def read_committed_step(root: Path) -> int | None:
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        return None
    try:
        return int(tracker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def export_milestone(
    repo_root: Path,
    checkpoint: Path,
    target: Path,
    run_id: str,
    step: int,
    full_metadata: dict[str, int],
) -> None:
    if target.exists():
        metadata = validate_model_only(target)
        log(f"milestone update {step} already verified at {target} ({metadata['model_bytes']} bytes)")
        return

    temporary = target.with_name(f".{target.name}.partial-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(repo_root / "scripts" / "model_merger.py"),
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(checkpoint / "actor"),
        "--target_dir",
        str(temporary),
    ]
    log(f"exporting analysis milestone update {step} to temporary directory {temporary}")
    subprocess.run(command, cwd=repo_root, check=True)
    model_metadata = validate_model_only(temporary)
    manifest = {
        "schema_version": "phase1.model_only_archive.v1",
        "run_id": run_id,
        "global_update": step,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_bytes": full_metadata["total_bytes"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **model_metadata,
    }
    manifest_path = temporary / "phase1_archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["config_sha256"] = file_sha256(temporary / "config.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.rename(target)
    log(
        f"verified and atomically archived update {step}: {target} "
        f"({model_metadata['model_bytes']} bytes, {model_metadata['tensor_count']} tensors)"
    )


def rotate_full_checkpoints(
    root: Path,
    committed_step: int,
    keep: int,
    milestones: set[int],
    archive_root: Path,
    archive_prefix: str,
    external_resume_checkpoint: Path | None,
    run_id: str,
) -> None:
    complete: list[tuple[int, Path]] = []
    for path in root.glob("global_step_*"):
        try:
            step = int(path.name.removeprefix("global_step_"))
        except ValueError:
            continue
        if step > committed_step:
            continue
        try:
            validate_full_checkpoint(path)
        except RuntimeError:
            continue
        complete.append((step, path))

    complete.sort()
    keep_steps = {step for step, _ in complete[-keep:]}
    for step, path in complete:
        if step in keep_steps:
            continue
        if step in milestones:
            ordinal = sorted(milestones).index(step) + 1
            archive = archive_root / f"{archive_prefix}-c{ordinal}-update{step}"
            validate_model_only(archive)
        log(f"removing rotated full recovery checkpoint {path}")
        shutil.rmtree(path)

    # The trainer may already have removed actor shards. Remove only stale shell
    # directories that are older than both retained committed checkpoints.
    floor = min(keep_steps) if keep_steps else committed_step
    for path in root.glob("global_step_*"):
        try:
            step = int(path.name.removeprefix("global_step_"))
        except ValueError:
            continue
        if step < floor and path.is_dir() and not (path / "actor").exists():
            shutil.rmtree(path)

    # A continuation uses a new run ID so its immutable manifest can record the
    # changed save frequency.  Keep its source checkpoint until the continuation
    # has committed and verified its first replacement.  At that point the old
    # source and new checkpoint are two valid copies; deleting the old one keeps
    # peak full-checkpoint storage near two slots while the next save is written.
    if external_resume_checkpoint is not None and external_resume_checkpoint.exists() and len(complete) >= 1:
        validate_full_checkpoint(external_resume_checkpoint)
        marker = external_resume_checkpoint.parent / "external_resume_checkpoint_rotation.json"
        marker.write_text(
            json.dumps(
                {
                    "schema_version": "phase1.external_resume_rotation.v1",
                    "continued_by_run_id": run_id,
                    "removed_checkpoint": str(external_resume_checkpoint),
                    "replacement_checkpoints": [str(path) for _, path in complete[-keep:]],
                    "removed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        log(f"removing superseded external resume checkpoint {external_resume_checkpoint}")
        shutil.rmtree(external_resume_checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--model-only-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-prefix", required=True)
    parser.add_argument("--milestones", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--keep-full", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--external-resume-checkpoint", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.keep_full < 2:
        raise ValueError("keep-full must be at least 2 to preserve a fallback checkpoint")
    root = args.checkpoint_root.resolve()
    repo_root = args.repo_root.resolve()
    expected_parent = (repo_root / "artifacts" / "checkpoints").resolve()
    if root.parent != expected_parent:
        raise ValueError(f"refusing checkpoint root outside {expected_parent}: {root}")
    milestones = set(args.milestones)
    external_resume_checkpoint = args.external_resume_checkpoint.resolve() if args.external_resume_checkpoint else None
    if external_resume_checkpoint is not None and external_resume_checkpoint.parent.parent != expected_parent:
        raise ValueError(f"refusing external resume checkpoint outside {expected_parent}: {external_resume_checkpoint}")
    last_reported: int | None = None
    log(f"watching {root}; milestones={sorted(milestones)}; keep_full={args.keep_full}")

    while True:
        step = read_committed_step(root)
        if step is None:
            time.sleep(args.poll_seconds)
            continue
        checkpoint = root / f"global_step_{step}"
        try:
            metadata = validate_full_checkpoint(checkpoint)
        except RuntimeError as error:
            if step != last_reported:
                log(f"tracker points to an unverified checkpoint; waiting: {error}")
                last_reported = step
            time.sleep(args.poll_seconds)
            continue
        if step != last_reported:
            log(f"verified committed recovery checkpoint update {step}: {metadata['total_bytes']} bytes")
            last_reported = step

        if step in milestones:
            ordinal = sorted(milestones).index(step) + 1
            target = args.model_only_root / f"{args.archive_prefix}-c{ordinal}-update{step}"
            export_milestone(repo_root, checkpoint, target, args.run_id, step, metadata)
        rotate_full_checkpoints(
            root,
            step,
            args.keep_full,
            milestones,
            args.model_only_root,
            args.archive_prefix,
            external_resume_checkpoint,
            args.run_id,
        )
        if step >= max(milestones):
            final_target = args.model_only_root / f"{args.archive_prefix}-c{len(milestones)}-update{max(milestones)}"
            validate_model_only(final_target)
            log("final milestone verified; watcher complete")
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
