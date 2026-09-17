"""Useful multi-process evaluation concurrency; no dummy GPU reservations."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from phase1.archive import atomic_write_json, append_jsonl_idempotent, sha256_file, utc_now
from phase2.resource_watch import gpu_snapshot

AMENDMENT = "evaluation_resource_amendment.json"
DISK_WAIVER = "evaluation_disk_waiver_20260914.json"


def load_policy(root):
    root = Path(root)
    path = root/AMENDMENT
    if not path.exists(): return None
    policy = json.loads(path.read_text())
    if policy["root"] != str(root.resolve()) or policy["protocol_sha256"] != sha256_file(root/"protocol.json"):
        raise ValueError("Evaluation resource amendment does not match frozen protocol")
    waiver_path = root/DISK_WAIVER
    if waiver_path.exists():
        waiver = json.loads(waiver_path.read_text())
        if (waiver["root"] != str(root.resolve()) or waiver["protocol_sha256"] != policy["protocol_sha256"]
                or waiver["superseded_resource_sha256"] != sha256_file(path) or not waiver["user_authorized"]):
            raise ValueError("Disk waiver is not bound to the authorized evaluation")
        policy = {**policy, "disk_budget_waived": True, "minimum_write_safety_mib": waiver["minimum_write_safety_mib"]}
    return policy


def check_disk(root, policy):
    waived = policy.get("disk_budget_waived", False)
    required = (policy["minimum_write_safety_mib"]*2**20 if waived else
                (policy["disk_reserve_gib"] + policy["remaining_write_allowance_gib"])*2**30)
    if shutil.disk_usage(root).free < required:
        if waived: raise RuntimeError("Filesystem nearly full: insufficient actual write space (not a disk-budget stop)")
        raise RuntimeError("Insufficient disk for remaining evaluation plus unchanged 200GiB reserve")


def process_inventory():
    lookup = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    devices = {uid.strip(): int(index) for index, uid in (line.split(",") for line in lookup.splitlines())}
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    result = {gpu: set() for gpu in devices.values()}
    for line in apps.splitlines():
        uid, pid = (s.strip() for s in line.split(","))
        if uid in devices: result[devices[uid]].add(int(pid))
    return result


def extra_slots(row, owned_pids, observed_pids, budget, headroom, limit):
    """Only idle devices or devices occupied exclusively by these workers."""
    if observed_pids - owned_pids: return 0
    n = len(owned_pids)
    if not n and (row["used_mib"] >= 1500 or row["utilization"] >= 10): return 0
    # Include not-yet-loaded workers' full planned budgets, not merely their
    # current small CUDA allocations. Also allow for actual usage > estimates.
    committed = max(row["used_mib"], n*budget)
    return max(0, min(limit-n, (row["total_mib"]-headroom-committed)//budget))


def retryable(output):
    return "out of memory" in output.lower() and not any(
        text in output for text in ("token alignment failed", "Live prompt/tokenizer mismatch", "Unregistered endpoint window"))


def run_parallel(pipeline, module, update, start_update=None):
    policy = load_policy(pipeline.root)
    if policy is None: raise ValueError("Multi-worker evaluation must be explicitly enabled")
    kind = module.rsplit(".", 1)[-1]
    settings = policy["stages"][kind]
    name = f"{kind}-u{update}" + (f"-from{start_update}" if start_update is not None else "")
    shards = pipeline.config["evaluation"].get("shards", 8)
    pending = [s for s in range(shards) if not pipeline.shard_complete(module, update, s, start_update)]
    workers, cooldowns, limits = {}, {}, {}
    last_report = time.monotonic()
    try:
        while pending or workers:
            check_disk(pipeline.root, policy)
            for shard, worker in list(workers.items()):
                process = worker["process"]
                code = process.poll()
                if code is None: continue
                worker["log"].close()
                del workers[shard]
                if code == 0 and pipeline.shard_complete(module, update, shard, start_update): continue
                with worker["path"].open("rb") as source:
                    source.seek(worker["offset"])
                    output = source.read().decode("utf-8", errors="replace")
                can_retry = retryable(output)
                archive = pipeline.root/"evaluation_attempts"/f"{name}-shard{shard}-{time.time_ns()}"
                archive.mkdir(parents=True)
                shutil.copy2(worker["path"], archive/"attempt.log")
                atomic_write_json(archive/"failure.json", {"at": utc_now(), "stage": name, "shard": shard,
                    "gpu": worker["gpu"], "exit_code": code, "log_offset": worker["offset"], "oom_retryable": can_retry})
                if not can_retry: raise RuntimeError(f"{name} shard {shard} failed; see {archive}")
                # Read-only signal computation is reproducible. Evaluation
                # resumes only missing trajectory IDs, each with its fixed RNG.
                pending.append(shard)
                gpu = worker["gpu"]
                cooldowns[gpu] = time.monotonic() + policy["oom_cooldown_seconds"]
                limits[gpu] = max(1, limits.get(gpu, settings["max_workers_per_gpu"])-1)
                print(f"[{utc_now()}] {name} shard {shard}: OOM archived; queued for safe retry", flush=True)
            snapshot = gpu_snapshot()
            inventory = process_inventory()
            capacity = {}
            for row in snapshot:
                gpu = row["gpu"]
                owned = {w["process"].pid for w in workers.values() if w["gpu"] == gpu}
                capacity[gpu] = 0 if time.monotonic() < cooldowns.get(gpu, 0) else extra_slots(
                    row, owned, inventory.get(gpu, set()), settings["worker_budget_mib"],
                    policy["gpu_headroom_mib"], limits.get(gpu, settings["max_workers_per_gpu"]))
            while pending and any(capacity.values()):
                # Spread across available GPUs before adding another local
                # process, then fill useful slots with distinct logical shards.
                gpu = min((g for g, n in capacity.items() if n),
                          key=lambda g: (sum(w["gpu"] == g for w in workers.values()), g))
                shard = pending.pop(0)
                path = pipeline.root/"logs"/f"{name}-shard{shard}.log"
                offset = path.stat().st_size if path.exists() else 0
                log = path.open("a")
                args = [sys.executable, "-u", "-m", module, "--root", str(pipeline.root),
                        "--update", str(update), "--shard", str(shard), "--shards", str(shards)]
                if start_update is not None: args += ["--start-update", str(start_update)]
                process = subprocess.Popen(args, cwd=pipeline.repo,
                    env={**pipeline.env, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                workers[shard] = {"process": process, "log": log, "path": path, "offset": offset, "gpu": gpu}
                capacity[gpu] -= 1
                append_jsonl_idempotent(pipeline.root/"evaluation_allocation_history.jsonl", [{
                    "at": utc_now(), "stage": name, "shard": shard, "pid": process.pid, "gpu": gpu,
                    "worker_budget_mib": settings["worker_budget_mib"], "log_offset": offset}], unique_fields=("pid", "stage", "shard"))
                print(f"[{utc_now()}] {name} shard {shard} -> GPU {gpu}, PID {process.pid}", flush=True)
            pipeline.status(name if workers else f"waiting_for_free_gpu-{name}", evaluation_update=update,
                pending_shards=pending, gpu_snapshot=snapshot,
                workers=[{"gpu": w["gpu"], "shard": s, "pid": w["process"].pid} for s, w in workers.items()],
                resource_mode="concurrent independent useful shards; no memory padding")
            if time.monotonic()-last_report > 300:
                pipeline.refresh_report()
                last_report = time.monotonic()
            if pending or workers: time.sleep(policy["poll_seconds"])
    finally:
        for worker in workers.values():
            if worker["process"].poll() is None:
                worker["process"].terminate()
                try: worker["process"].wait(timeout=10)
                except subprocess.TimeoutExpired: worker["process"].kill(); worker["process"].wait()
            worker["log"].close()
    pipeline.refresh_report()
