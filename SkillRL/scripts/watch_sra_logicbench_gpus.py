"""Queue a prepared LogicBench Phase 1/2 command until enough GPUs are free.

The queue is deliberately fail-closed: an explicit runnable manifest, a real
matching training preflight receipt, and sufficient persistent disk are all
required before the command can start. It never retries a failed experiment.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "skillscope.logicbench_gpu_queue.v1"
REPO = Path(__file__).resolve().parents[1]


def parse_gpu_snapshot(gpu_text: str, process_text: str) -> list[dict]:
    """Parse UUID-bound GPU and compute-process rows from NVIDIA SMI."""
    processes: dict[str, list[int]] = {}
    try:
        for line in process_text.splitlines():
            if not line.strip():
                continue
            uuid, pid = (part.strip() for part in line.split(",", 1))
            processes.setdefault(uuid, []).append(int(pid))
        rows = []
        for line in gpu_text.splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 5:
                raise ValueError("Wrong GPU field count")
            index, uuid, used, total, utilization = parts
            rows.append({"index": int(index), "uuid": uuid,
                         "used_mib": int(used), "total_mib": int(total),
                         "utilization": int(utilization),
                         "compute_pids": sorted(processes.get(uuid, []))})
        if not rows or len({row["uuid"] for row in rows}) != len(rows):
            raise ValueError("Empty or duplicate GPU inventory")
        if set(processes) - {row["uuid"] for row in rows}:
            raise ValueError("Unknown GPU UUID in compute process inventory")
        return sorted(rows, key=lambda row: row["index"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid GPU snapshot: {error}") from error


def gpu_snapshot() -> list[dict]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10)
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10)
    return parse_gpu_snapshot(gpu.stdout, processes.stdout)


def eligible_gpus(snapshot: list[dict], *, max_used_mib: int,
                  max_utilization: int) -> list[int]:
    return [row["index"] for row in snapshot
            if row["used_mib"] <= max_used_mib
            and row["utilization"] <= max_utilization
            and not row["compute_pids"]]


def stable_gpu_choice(history: list[list[int]], *, count: int,
                      required_polls: int) -> list[int] | None:
    if count < 2 or required_polls < 2:
        raise ValueError("Require at least two GPUs and two stable polls")
    if len(history) < required_polls:
        return None
    common = set(history[-required_polls])
    for sample in history[-required_polls + 1:]:
        common.intersection_update(sample)
    selected = sorted(common)
    return selected[:count] if len(selected) >= count else None


def launch_blockers(manifest: dict, disk_free_bytes: int) -> list[str]:
    blockers = []
    if manifest.get("schema_version") != SCHEMA:
        blockers.append("invalid_manifest_schema")
    if manifest.get("runnable_experiment") is not True:
        blockers.append("experiment_not_ready")
    required_gpus = manifest.get("required_gpu_count", 2)
    if type(required_gpus) is not int or required_gpus < 2:
        blockers.append("invalid_gpu_count")
    command = manifest.get("command")
    if (not isinstance(command, list) or not command
            or not all(isinstance(part, str) and part for part in command)
            or shutil.which(command[0]) is None):
        blockers.append("launch_command_missing")
    receipt_path = manifest.get("preflight_receipt")
    if not isinstance(receipt_path, str) or not Path(receipt_path).is_file():
        blockers.append("preflight_receipt_missing")
    else:
        try:
            receipt = json.loads(Path(receipt_path).read_text())
            validated_gpus = receipt.get(
                "training_gpu_count", 2 if receipt.get("two_gpu_training") is True else 0)
            if (receipt.get("status") != "PASS"
                    or type(validated_gpus) is not int
                    or type(required_gpus) is not int
                    or validated_gpus < required_gpus
                    or receipt.get("logicbench_phase12_pipeline") is not True):
                blockers.append("preflight_receipt_invalid")
        except (OSError, ValueError, TypeError):
            blockers.append("preflight_receipt_invalid")
    run_root = manifest.get("run_root")
    if not isinstance(run_root, str) or not Path(run_root).is_absolute():
        blockers.append("run_root_missing")
    elif Path(run_root).exists() and any(Path(run_root).iterdir()):
        blockers.append("run_root_not_empty")
    minimum = manifest.get("minimum_free_bytes")
    if type(minimum) is not int or minimum <= 0:
        blockers.append("disk_budget_missing")
    elif disk_free_bytes < minimum:
        blockers.append("insufficient_disk")
    return blockers


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _disk_free(manifest: dict) -> int:
    root = manifest.get("run_root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        return 0
    candidate = Path(root)
    while not candidate.exists():
        if candidate == candidate.parent:
            return 0
        candidate = candidate.parent
    return shutil.disk_usage(candidate).free


def _snapshot_status(manifest: dict, history: list[list[int]], *, count: int,
                     stable_polls: int, max_used_mib: int,
                     max_utilization: int) -> tuple[dict, list[int] | None]:
    snapshot = gpu_snapshot()
    free = eligible_gpus(snapshot, max_used_mib=max_used_mib,
                         max_utilization=max_utilization)
    history.append(free)
    del history[:-stable_polls]
    selected = stable_gpu_choice(history, count=count,
                                 required_polls=stable_polls)
    disk_free = _disk_free(manifest)
    blockers = launch_blockers(manifest, disk_free)
    if selected is None:
        blockers.append("waiting_for_stable_free_gpus")
    status = {"updated_at": datetime.now(timezone.utc).isoformat(),
              "state": "ready" if not blockers else "waiting",
              "blockers": blockers, "eligible_gpu_ids": free,
              "selected_gpu_ids": selected, "disk_free_bytes": disk_free,
              "gpus": snapshot}
    return status, selected


def watch(manifest_path: Path, state_dir: Path, *, count: int = 2,
          stable_polls: int = 3, max_used_mib: int = 512,
          max_utilization: int = 10, poll_seconds: float = 30,
          once: bool = False) -> dict:
    if count < 2 or stable_polls < 2 or poll_seconds <= 0:
        raise ValueError("Invalid GPU watch settings")
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "watch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        history: list[list[int]] = []
        while True:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("required_gpu_count", 2) != count:
                raise ValueError("GPU queue count differs from experiment manifest")
            try:
                status, selected = _snapshot_status(
                    manifest, history, count=count, stable_polls=stable_polls,
                    max_used_mib=max_used_mib, max_utilization=max_utilization)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                history.clear()
                status = {"updated_at": datetime.now(timezone.utc).isoformat(),
                          "state": "waiting", "blockers": ["gpu_probe_failed"],
                          "error": f"{type(error).__name__}: {error}"}
                selected = None
            _write_json(state_dir / "status.json", status)
            with (state_dir / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(status, sort_keys=True) + "\n")
            if once:
                return status
            if status["state"] == "ready" and selected is not None:
                # Close the observation-to-launch race as far as read-only
                # NVIDIA queries permit; another user's process can still start.
                final = eligible_gpus(gpu_snapshot(), max_used_mib=max_used_mib,
                                      max_utilization=max_utilization)
                if not set(selected).issubset(final) or launch_blockers(manifest, _disk_free(manifest)):
                    history.clear()
                    time.sleep(poll_seconds)
                    continue
                root = Path(manifest["run_root"])
                root.mkdir(parents=True, exist_ok=True)
                launch = root / "launch.json"
                payload = {"started_at": datetime.now(timezone.utc).isoformat(),
                           "manifest": str(manifest_path.resolve()),
                           "gpu_ids": selected, "command": manifest["command"]}
                with launch.open("x") as stream:
                    json.dump(payload, stream, indent=2)
                    stream.write("\n")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected))
                environment["PYTHONDONTWRITEBYTECODE"] = "1"
                with (root / "run.log").open("x") as log:
                    process = subprocess.Popen(manifest["command"], cwd=REPO,
                                               env=environment, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    status.update({"state": "running", "pid": process.pid,
                                   "selected_gpu_ids": selected, "blockers": []})
                    _write_json(state_dir / "status.json", status)
                    exit_code = process.wait()
                status.update({"state": "finished" if exit_code == 0 else "failed",
                               "exit_code": exit_code,
                               "finished_at": datetime.now(timezone.utc).isoformat()})
                _write_json(state_dir / "status.json", status)
                return status
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "configs/sra_logicbench_phase12_gpu_queue.json")
    parser.add_argument("--state-dir", type=Path,
                        default=REPO / "artifacts/logicbench/gpu_queue")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--stable-polls", type=int, default=3)
    parser.add_argument("--max-used-mib", type=int, default=512)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    status = watch(args.manifest, args.state_dir, count=args.gpus,
                   stable_polls=args.stable_polls, max_used_mib=args.max_used_mib,
                   max_utilization=args.max_utilization,
                   poll_seconds=args.poll_seconds, once=args.once)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
