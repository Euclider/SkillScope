"""Durable idle-GPU queue for coverage and a NEW pre-update baseline only.

This runner cannot train or open a future endpoint. It leaves all historical
cohorts untouched. Its frozen queue is intentionally distinct from the still
unfrozen larger-window training protocol.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import zipfile

from phase1.archive import atomic_write_json, sha256_file, stable_hash, utc_now
from phase2.protocol import expected_trajectory_ids, seed_streams
from phase2.resource_watch import gpu_snapshot, idle_devices

REPO = Path(__file__).resolve().parents[1]
BASELINE_SCHEMA = "phase2.preupdate_baseline.v1"


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_queue(config, repo=REPO):
    if config.get("schema_version") != "phase2.ranking_preparation.v1" or config.get("status") != "frozen":
        raise ValueError("An explicit frozen preparation queue is required")
    if config["scope"]["launch_new_rl"] or config["scope"]["modify_old_results"]:
        raise ValueError("This queue cannot train or change old results")
    seed_streams(config)
    seeds = config["coverage_seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Unique coverage seeds required")
    if set(seeds) & set(config["evaluation"]["gold_seeds"] + config["evaluation"]["old_evidence_seeds"]):
        raise ValueError("Coverage and continuation seed streams must be distinct")
    if config["resources"]["consecutive_idle_polls"] != 2:
        raise ValueError("Two consecutive idle polls required")
    if config["resources"]["idle_used_mib_below"] != 1500 or config["resources"]["idle_utilization_below"] != 10:
        raise ValueError("Use the conservative shared resource_watch thresholds")
    games = [x.strip() for x in (repo/config["game_ids_file"]).read_text().splitlines() if x.strip() and not x.startswith("#")]
    if len(games) != config["game_count"] or len(set(games)) != len(games):
        raise ValueError("Game support does not match the queue")
    if not 1 <= config["shards"] <= len(games):
        raise ValueError("Coverage shards must be nonempty")
    bank = json.loads((repo/config["skill_bank"]).read_text())
    if config["context_id"] not in bank["task_specific_skills"]:
        raise ValueError("Unknown Skill context")
    if not (Path(config["source_model"])/"model.safetensors").is_file():
        raise ValueError("Complete source model required")
    return games


def prepare(config_path):
    config = json.loads(config_path.read_text())
    validate_queue(config)
    root = Path(config["root"])
    if root.exists():
        if not (root/"queue.json").exists() or json.loads((root/"queue.json").read_text()) != config:
            raise ValueError("Refusing to adopt or overwrite a different run directory")
        verify_manifest(root)
        return root
    root.mkdir(parents=True)
    (root/"logs").mkdir()
    atomic_write_json(root/"queue.json", config)
    source_files = sorted(set(REPO.joinpath("phase2").rglob("*.py")) |
                          set(REPO.joinpath("phase1").glob("*.py")) |
                          set(REPO.joinpath("agent_system/memory").glob("*.py")) |
                          {REPO/"agent_system/environments/prompts/alfworld.py",
                           REPO/"agent_system/environments/env_package/alfworld/projection.py"})
    with zipfile.ZipFile(root/"source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files:
            archive.write(path, str(path.relative_to(REPO)))
    inputs = [REPO/config["skill_bank"], REPO/config["game_ids_file"]]
    inputs += sorted(p for p in Path(config["source_model"]).iterdir() if p.is_file())
    print("Hashing immutable source model and inputs (no GPU allocation)", flush=True)
    atomic_write_json(root/"queue_manifest.json", {
        "created_at": utc_now(), "queue_sha256": sha256_file(root/"queue.json"),
        "source_sha256": {str(p): sha256_file(p) for p in source_files},
        "input_sha256": {str(p): sha256_file(p) for p in inputs},
        "input_stats": {str(p): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in inputs},
        "source_archive_sha256": sha256_file(root/"source.zip"),
    })
    atomic_write_json(root/"status.json", {"at": utc_now(), "stage": "prepared_not_started"})
    return root


def verify_manifest(root):
    manifest = json.loads((root/"queue_manifest.json").read_text())
    if sha256_file(root/"queue.json") != manifest["queue_sha256"]:
        raise ValueError("Frozen queue changed")
    for filename, digest in manifest["source_sha256"].items():
        if sha256_file(filename) != digest:
            raise ValueError(f"Queued source changed; archive an amendment before resuming: {filename}")
    # Large model hashes were computed at registration. Detect file changes on
    # every start without repeatedly reading 19 GB while polling idle GPUs.
    for filename, stats in manifest["input_stats"].items():
        stat = Path(filename).stat()
        if stat.st_size != stats["bytes"] or stat.st_mtime_ns != stats["mtime_ns"]:
            raise ValueError(f"Frozen input changed: {filename}")


def validate_baseline(config, root, update, repo=REPO):
    if config.get("schema_version") != BASELINE_SCHEMA or config.get("status") != "frozen" or update != config["parent_update"]:
        raise ValueError("Preparation evaluation may only open the frozen pre-update endpoint")
    verify_manifest(root)
    manifest = json.loads((root/"baseline_manifest.json").read_text())
    if manifest["protocol_sha256"] != sha256_file(root/"protocol.json"):
        raise ValueError("Baseline protocol changed after support freeze")
    if config["skill_bank_sha256"] != sha256_file(repo/"memory_data/alfworld/claude_style_skills.json"):
        raise ValueError("Frozen baseline Skill Bank changed")


def coverage_command(config, root, shard):
    ev = config["evaluation"]
    # --skill-id is an archive label required by the old CLI, NOT a router
    # override: full_bank + step-routing still chooses among all 18 candidates.
    return [sys.executable, "-u", "-m", "phase1.eval_skill_margin",
            "--checkpoint", config["source_model"], "--skill-bank", str(REPO/config["skill_bank"]),
            "--skill-id", "cle_006", "--context-id", config["context_id"],
            "--game-ids-file", str(REPO/config["game_ids_file"]),
            "--game-shard-index", str(shard), "--game-shard-count", str(config["shards"]),
            "--eval-seeds", *map(str, config["coverage_seeds"]), "--conditions", "full_bank",
            "--temperature", str(ev["temperature"]), "--top-p", str(ev["top_p"]),
            "--max-steps", str(ev["max_steps"]), "--max-new-tokens", str(ev["max_new_tokens"]),
            "--history-length", str(ev["history_length"]), "--step-routing", "--router-general-top-k", "12",
            "--environment-seed", str(config["environment_seed"]), "--rl-seed", str(config["parent_rl_seed"]),
            "--update-id", str(config["parent_update"]), "--split", "valid_unseen",
            "--run-id", config["run_id"]+"-coverage", "--output", str(root/"coverage"/f"shard-{shard}.jsonl")]


def coverage_complete(config, root, shard):
    path = root/"coverage"/f"shard-{shard}.jsonl"
    if not path.exists():
        return False
    games = validate_queue(config)
    from phase1.eval_skill_margin import resolve_game_file
    expected = {(str(resolve_game_file(game)), seed) for game in games[shard::config["shards"]] for seed in config["coverage_seeds"]}
    rows = read_rows(path)
    actual = {(r["game_id"], r["eval_seed"]) for r in rows}
    if len(rows) != len(actual) or not actual <= expected:
        raise ValueError("Invalid coverage index identities")
    if any(r["checkpoint_id"] != Path(config["source_model"]).name or r["context_id"] != config["context_id"] or r["skill_condition"] != "full_bank" for r in rows):
        raise ValueError("Coverage index does not describe the frozen source")
    if any(not Path(r["trajectory_path"]).is_file() for r in rows):
        raise ValueError("Coverage trajectory archive is missing")
    return actual == expected


class PreparationQueue:
    def __init__(self, root):
        self.root = root
        self.config = json.loads((root/"queue.json").read_text())
        validate_queue(self.config)
        verify_manifest(root)
        self.env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "TOKENIZERS_PARALLELISM": "false", "ALFWORLD_DATA": "/home/wangyifan/skill-RL/data/alfworld",
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        os.environ["ALFWORLD_DATA"] = self.env["ALFWORLD_DATA"]

    def status(self, stage, **extra):
        atomic_write_json(self.root/"status.json", {"at": utc_now(), "stage": stage,
                          "pid": os.getpid(), "new_rl_enabled": False, **extra})

    def cpu(self, command, name):
        self.status(name)
        with (self.root/"logs"/f"{name}.log").open("a") as log:
            subprocess.run(command, cwd=REPO, env={**self.env, "CUDA_VISIBLE_DEVICES": ""},
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    def schedule(self, stage, command, complete):
        pending = [s for s in range(self.config["shards"]) if not complete(s)]
        active, previous = {}, set()
        policy = self.config["resources"]
        locks = REPO/"artifacts/phase2/gpu_locks"
        locks.mkdir(parents=True, exist_ok=True)
        try:
            while pending or active:
                for gpu, worker in list(active.items()):
                    proc, log, lock, shard = worker
                    if proc.poll() is None:
                        continue
                    log.close(); lock.close(); del active[gpu]
                    if proc.returncode or not complete(shard):
                        raise RuntimeError(f"{stage} shard {shard} failed ({proc.returncode}); archived partial trajectories can be resumed after inspection")
                snapshot = gpu_snapshot()
                available = set(idle_devices(snapshot))
                stable = sorted((available & previous) - set(active))
                free_gib = shutil.disk_usage(self.root).free / 2**30
                disk_ok = free_gib >= policy["minimum_disk_free_gib"]
                observation = {"at": utc_now(), "stage": stage, "gpus": snapshot,
                               "idle_gpus": sorted(available), "confirmed_idle_gpus": stable,
                               "disk_free_gib": free_gib, "disk_ok": disk_ok}
                with (self.root/"gpu_watch.jsonl").open("a") as handle:
                    handle.write(json.dumps(observation)+"\n"); handle.flush()
                atomic_write_json(self.root/"gpu_watch_latest.json", observation)
                for gpu in stable if disk_ok else []:
                    if not pending or len(active) >= policy["maximum_workers"]:
                        break
                    lock = (locks/f"gpu-{gpu}.lock").open("a")
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close(); continue
                    if gpu not in idle_devices(gpu_snapshot()):
                        lock.close(); continue
                    verify_manifest(self.root)
                    shard = pending.pop(0)
                    args = command(shard)
                    logpath = self.root/"logs"/f"{stage}-shard-{shard}.log"
                    log = logpath.open("a")
                    try:
                        proc = subprocess.Popen(args, cwd=REPO, env={**self.env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                                pass_fds=(lock.fileno(),))
                    except BaseException:
                        log.close(); lock.close(); raise
                    active[gpu] = (proc, log, lock, shard)
                    allocation = {"at": utc_now(), "stage": stage, "gpu": gpu, "shard": shard,
                                  "pid": proc.pid, "command": args, "log": str(logpath)}
                    with (self.root/"allocations.jsonl").open("a") as handle:
                        handle.write(json.dumps(allocation)+"\n")
                    print(json.dumps(allocation), flush=True)
                self.status(stage if active else ("waiting_for_idle_gpu" if disk_ok else "waiting_for_disk_reserve"),
                            queued_stage=stage, pending_shards=pending, gpu_snapshot=snapshot,
                            workers=[{"gpu": g, "pid": p.pid, "shard": s} for g, (p, _, _, s) in active.items()],
                            disk_free_gib=free_gib, poll_seconds=policy["poll_seconds"])
                previous = available
                if pending or active:
                    time.sleep(policy["poll_seconds"])
        finally:
            for proc, log, lock, _ in active.values():
                if proc.poll() is None:
                    # Only subprocess groups launched by this queue are in scope.
                    os.killpg(proc.pid, signal.SIGTERM)
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                log.close(); lock.close()

    def freeze_support(self):
        config, root = self.config, self.root
        checkpoint_id = Path(config["source_model"]).name
        coverage = root/"anchors/coverage.json"
        if not coverage.exists():
            self.cpu([sys.executable, "-m", "phase1.build_all_first_invocation_anchors",
                      "--trajectory-index", *[str(root/"coverage"/f"shard-{s}.jsonl") for s in range(config["shards"])],
                      "--skill-bank", str(REPO/config["skill_bank"]), "--context-id", config["context_id"],
                      "--checkpoint-id", checkpoint_id, "--minimum-occurrences", str(config["minimum_occurrences"]),
                      "--minimum-games", str(config["minimum_games"]), "--maximum-selected", str(config["maximum_anchors_per_skill"]),
                      "--output-dir", str(root/"anchors")], "build-natural-anchors")
        rows = json.loads(coverage.read_text())["coverage"]
        for row in rows:
            if not row["supported"]:
                continue
            skill = row["skill_id"]
            control = root/"controls"/f"{skill}.json"
            if not control.exists():
                anchor = read_rows(root/"anchors/selected"/f"{skill}.jsonl")[0]
                self.cpu([sys.executable, "-m", "phase1.create_payload_placebo", "--model", config["source_model"],
                          "--skill-bank", str(REPO/config["skill_bank"]), "--skill-id", skill,
                          "--task-description", anchor["task_description"], "--output", str(control)], f"placebo-{skill}")
        if not (root/"support/anchor_sets.proposal.json").exists():
            self.cpu([sys.executable, "-m", "phase2.coverage_audit", "--coverage", str(coverage),
                      "--output", str(root/"support"), "--controls", str(root/"controls"), "--checkpoint-id", checkpoint_id,
                      "--minimum-occurrences", str(config["minimum_occurrences"]), "--minimum-games", str(config["minimum_games"]),
                      "--maximum-anchors", str(config["maximum_anchors_per_skill"])], "audit-support")
        support = json.loads((root/"support/anchor_sets.proposal.json").read_text())
        if not support["anchor_sets"]:
            self.status("coverage_complete_no_supported_skills", supported_skills=0)
            return False
        protocol = {"schema_version": BASELINE_SCHEMA, "status": "frozen", "run_id": config["run_id"],
                    "root": str(root), "parent_update": config["parent_update"], "parent_rl_seed": config["parent_rl_seed"],
                    "rl_path_id": "seed303-u35-preparation-only", "primary_control": "placebo",
                    "skill_bank_sha256": sha256_file(REPO/config["skill_bank"]),
                    "evaluation": {**config["evaluation"], "shards": config["shards"],
                                   "anchor_sets": support["anchor_sets"], "anchor_count": support["anchor_count"]}}
        path = root/"protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError("Refusing to change frozen baseline support")
        if not path.exists():
            atomic_write_json(path, protocol)
        suffixes_per_anchor = len(seed_streams(protocol)["evidence"] + seed_streams(protocol)["gold"]) * len(protocol["evaluation"]["arms"])
        atomic_write_json(root/"baseline_manifest.json", {"protocol_sha256": sha256_file(path),
                          "queue_sha256": sha256_file(root/"queue.json"), "supported_skills": len(support["anchor_sets"]),
                          "anchor_count": support["anchor_count"], "expected_suffixes": support["anchor_count"]*suffixes_per_anchor})
        model = root/"models"/f"u{config['parent_update']:04d}"
        model.parent.mkdir(parents=True, exist_ok=True)
        if model.is_symlink() and model.resolve() != Path(config["source_model"]).resolve():
            raise ValueError("Unexpected model alias")
        if not model.exists():
            model.symlink_to(config["source_model"], target_is_directory=True)
        return True

    def run(self):
        self.schedule("coverage", lambda s: coverage_command(self.config, self.root, s),
                      lambda s: coverage_complete(self.config, self.root, s))
        if not self.freeze_support():
            return
        protocol = json.loads((self.root/"protocol.json").read_text())
        update = self.config["parent_update"]
        all_ids = expected_trajectory_ids(protocol, REPO, update)
        def complete(shard):
            directory = self.root/"evaluations"/f"u{update:04d}"
            marker = directory/f"shard-{shard}-complete.json"
            if not marker.exists(): return False
            metadata = json.loads(marker.read_text())
            if metadata["protocol_sha256"] != sha256_file(self.root/"protocol.json") or metadata["max_jobs"] is not None:
                raise ValueError("Invalid baseline completion marker")
            rows = read_rows(directory/f"shard-{shard}.jsonl")
            ids = {r["trajectory_id"] for r in rows}
            if len(ids) != len(rows) or ids != set(all_ids[shard::self.config["shards"]]):
                raise ValueError("Completed baseline index is incomplete or duplicated")
            if any(not Path(r["trajectory_path"]).is_file() for r in rows):
                raise ValueError("Missing baseline trajectory")
            return True
        self.schedule("baseline", lambda s: [sys.executable, "-u", "-m", "phase2.evaluate", "--root", str(self.root),
                      "--update", str(update), "--shard", str(s), "--shards", str(self.config["shards"])], complete)
        self.cpu([sys.executable, "-m", "phase2.preparation_report", "--root", str(self.root)], "baseline-report")
        self.status("preparation_complete_awaiting_new_window_protocol", coverage_trajectories=self.config["game_count"]*len(self.config["coverage_seeds"]),
                    baseline_suffixes=len(all_ids), supported_skills=len(protocol["evaluation"]["anchor_sets"]),
                    anchor_count=protocol["evaluation"]["anchor_count"], new_post_update_results=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    root = prepare(args.config.resolve()) if args.config else args.root
    if root is None: parser.error("--config or --root is required")
    if args.prepare_only:
        print(root); return
    if args.detach:
        verify_manifest(root)
        with (root/"supervisor.log").open("a") as log:
            proc = subprocess.Popen([sys.executable, "-u", "-m", "phase2.queued_preparation", "--root", str(root)],
                                    cwd=REPO, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        atomic_write_json(root/"supervisor_launch.json", {"at": utc_now(), "pid": proc.pid})
        print(f"Supervisor PID {proc.pid}; state: {root/'status.json'}"); return
    with (root/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = PreparationQueue(root)
        def stop(signum, frame):
            raise KeyboardInterrupt(f"Supervisor received signal {signum}")
        signal.signal(signal.SIGTERM, stop)
        try: queue.run()
        except BaseException as error:
            queue.status("stopped_on_error", error=str(error)); raise


if __name__ == "__main__":
    main()
