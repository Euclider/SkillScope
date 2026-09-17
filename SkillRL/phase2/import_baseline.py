"""Reindex an exactly matched, completed pre-update baseline without new rollout."""
import argparse
import json
from pathlib import Path

from phase1.archive import atomic_write_json, sha256_file, stable_hash, utc_now
from phase1.build_all_first_invocation_anchors import write_jsonl
from phase2.protocol import evaluation_identity, evaluation_jobs, parent_update, validate_extended
from phase2.utilities import read_evaluations


def import_baseline(root):
    root = Path(root)
    config = json.loads((root/"protocol.json").read_text())
    repo = Path(__file__).resolve().parents[1]
    validate_extended(config, repo)
    source = Path(config["baseline_source"])
    if source.resolve() == root.resolve(): raise ValueError("Baseline source must be a separate immutable cohort")
    if sha256_file(source/"protocol.json") != config["baseline_protocol_sha256"]:
        raise ValueError("Source baseline protocol changed")
    original = json.loads((source/"protocol.json").read_text())
    if original["evaluation"] != config["evaluation"] or parent_update(original) != parent_update(config):
        raise ValueError("Cannot reuse baseline with changed anchors, controls, seeds, generation, or endpoint")
    marker = root/"baseline_import.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["protocol_sha256"] != sha256_file(root/"protocol.json"): raise ValueError("Import target protocol changed")
        return saved
    data = read_evaluations(source)
    update = parent_update(config)
    if data.empty or set(data["update"]) != {update}: raise ValueError("Source baseline is incomplete")
    keys = ["anchor_id", "skill_id", "purpose", "continuation_seed", "arm"]
    if data.duplicated(keys).any(): raise ValueError("Duplicate source baseline identity")
    rows_by_key = {tuple(row[k] for k in keys): row for row in data.to_dict("records")}
    jobs = evaluation_jobs(config, repo)
    directory = root/"evaluations"/f"u{update:04d}"
    outputs = []
    source_hashes = {}
    for job in jobs:
        identity = evaluation_identity(config, update, job)
        old = rows_by_key[tuple(identity[k] for k in keys)]
        path = Path(old["trajectory_path"])
        payload = json.loads(path.read_text())
        if payload["trajectory_id"] != old["trajectory_id"] or any(payload[k] != old[k] for k in keys):
            raise ValueError("Source baseline trajectory/index identity mismatch")
        source_hashes[str(path)] = sha256_file(path)
        tid = stable_hash(identity)[:24]
        destination = directory/"trajectories"/identity["skill_id"]/f"{tid}.json"
        provenance = {"source_run_id": original["run_id"], "source_trajectory_id": old["trajectory_id"],
                      "source_trajectory_path": str(path), "source_sha256": source_hashes[str(path)],
                      "imported_without_new_rollout": True}
        new = {**old, **identity, "trajectory_id": tid, "trajectory_path": str(destination),
               "rl_path_id": config["rl_path_id"], "baseline_import": provenance}
        atomic_write_json(destination, {**payload, **new})
        outputs.append(new)
    shards = config["evaluation"]["shards"]
    for shard in range(shards):
        part = outputs[shard::shards]
        write_jsonl(directory/f"shard-{shard}.jsonl", part)
        atomic_write_json(directory/f"shard-{shard}-complete.json", {
            "created_at": utc_now(), "jobs": len(part), "shard": shard, "max_jobs": None,
            "protocol_sha256": sha256_file(root/"protocol.json"), "shards": shards, "imported_without_new_rollout": True})
    model = root/"models"/f"u{update:04d}"
    model.parent.mkdir(parents=True, exist_ok=True)
    source_model = source/"models"/f"u{update:04d}"
    if model.exists() and model.resolve() != source_model.resolve(): raise ValueError("Unexpected baseline model target")
    if not model.exists(): model.symlink_to(source_model.resolve(), target_is_directory=True)
    receipt = {"created_at": utc_now(), "protocol_sha256": sha256_file(root/"protocol.json"),
               "source_protocol_sha256": config["baseline_protocol_sha256"], "imported_rows": len(outputs),
               "new_environment_rollouts": 0, "source_model": str(source_model.resolve()), "source_trajectory_sha256": source_hashes}
    atomic_write_json(marker, receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    result = import_baseline(a.root)
    print(json.dumps({k: v for k, v in result.items() if k != "source_trajectory_sha256"}))


if __name__ == "__main__": main()
