"""Explicitly authorized, audited storage rotation; never delete policy evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase1.watch_qwen35_checkpoints import read_committed_step, validate_full_checkpoint, validate_model_only

REPO = Path(__file__).resolve().parents[1]
LEGACY = REPO/"artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1"
APPROVED_CONVERSIONS = ("u0031-w4", "u0033-w1", "u0033-w2", "u0033-w5", "u0033-w6", "u0034-w1")


def checked_tree(path, parent):
    path, parent = Path(path), Path(parent)
    if path.is_symlink() or path.parent.resolve() != parent.resolve() or path.resolve() == parent.resolve():
        raise ValueError(f"Unsafe storage target: {path}")
    if not path.is_dir(): raise ValueError(f"Missing target directory: {path}")
    files = sorted(path.rglob("*"))
    if any(p.is_symlink() for p in files): raise ValueError("Refusing a deletion tree containing symlinks")
    return [p for p in files if p.is_file()]


def archive_and_remove(path, parent, archive, reason, sources):
    path, archive = Path(path), Path(archive)
    files = checked_tree(path, parent)
    if archive.resolve().is_relative_to(path.resolve()): raise ValueError("Audit archive must be outside deleted target")
    archive.mkdir(parents=True, exist_ok=True)
    inventory = [{"relative_path": str(p.relative_to(path)), "bytes": p.stat().st_size} for p in files]
    for p in files:
        # Retain conversion hashes, scheduler, RNG/data state and all small
        # metadata. Large weights/Adam tensors are the explicitly removed data.
        if p.stat().st_size < 32*2**20:
            destination = archive/"metadata"/p.relative_to(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, destination)
    record = {"at": utc_now(), "target": str(path.resolve()), "reason": reason, "sources": sources,
              "files": inventory, "logical_bytes": sum(x["bytes"] for x in inventory), "state": "validated_before_deletion"}
    atomic_write_json(archive/"receipt.json", record)
    shutil.rmtree(path)
    record.update(state="deleted", deleted_at=utc_now())
    atomic_write_json(archive/"receipt.json", record)
    print(json.dumps({k: record[k] for k in ("target", "logical_bytes", "state")}), flush=True)
    return record


def release_legacy_conversions(archive):
    parent = LEGACY/"elastic_resume"
    plans = []
    # Validate ALL native source checkpoints and exports before removing any.
    for name in APPROVED_CONVERSIONS:
        target = parent/name
        if not target.exists():
            receipt = Path(archive)/name/"receipt.json"
            if receipt.exists() and json.loads(receipt.read_text()).get("state") == "deleted": continue
            raise ValueError(f"Expected approved conversion missing without audit: {target}")
        checked_tree(target, parent)
        manifests = list(target.glob("global_step_*/actor/elastic_resume.json"))
        if len(manifests) != 1: raise ValueError("Ambiguous conversion identity")
        meta = json.loads(manifests[0].read_text())
        native, model = Path(meta["source_checkpoint"]), Path(meta["source_model"])
        if native.resolve().is_relative_to(target.resolve()) or model.resolve().is_relative_to(target.resolve()):
            raise ValueError("Deletion would remove the reconstruction source")
        validate_full_checkpoint(native); validate_model_only(model)
        if sha256_file(model/"phase2_export.json") != meta["source_export_sha256"]:
            raise ValueError("Source export metadata changed")
        plans.append((target, {"native_checkpoint": str(native), "fp32_policy": str(model),
                                "conversion_manifest_sha256": sha256_file(manifests[0]),
                                "target_world_size": meta["target_world_size"], "reconstructable": True}))
    return [archive_and_remove(target, parent, Path(archive)/target.name,
                               "User approved release of six obsolete, reconstructable world-size conversions on 2026-09-13", sources)
            for target, sources in plans]


def rotate_recovery(root, config, update):
    settings = config.get("storage", {})
    if not settings.get("rolling_recovery_authorized", False): return []
    if settings.get("keep_full_recovery") != 2: raise ValueError("Retain exactly two verified native recovery checkpoints")
    root = Path(root).resolve()
    parent = root/"checkpoints"
    if read_committed_step(parent) != update: raise ValueError("Latest recovery checkpoint has not committed")
    present = sorted((int(p.name.removeprefix("global_step_")), p) for p in parent.glob("global_step_*") if p.is_dir())
    if any(u > update for u, _ in present): raise ValueError("A later checkpoint exists; do not rotate out of order")
    if len(present) <= 2: return []
    retained = present[-2:]
    for _, path in retained: validate_full_checkpoint(path)
    records = []
    for old, target in present[:-2]:
        if old not in config["post_updates"]: raise ValueError("Refusing to delete a checkpoint outside this new run")
        model = root/"models"/f"u{old:04d}"
        validate_model_only(model)
        export = json.loads((model/"phase2_export.json").read_text())
        if export["dtype"] != "float32": raise ValueError("Every removed recovery state requires a lossless retained policy")
        if not (root/"signals"/f"u{old:04d}"/"committed.json").exists():
            raise ValueError("Never rotate a checkpoint before its single-update signals commit")
        records.append(archive_and_remove(target, parent, root/"storage_audit"/f"recovery-u{old:04d}",
            "Authorized latest-two recovery rotation; old Adam tensors removed, every FP32 policy and training/signal evidence retained",
            {"retained_native_checkpoints": [str(p) for _, p in retained], "retained_fp32_policy": str(model),
             "older_optimizer_reconstructable": False}))
    return records


def release_run_conversion(root, config, update):
    if not config.get("storage", {}).get("release_completed_conversions", False): return
    root = Path(root).resolve()
    validate_full_checkpoint(root/"checkpoints"/f"global_step_{update}")
    allocation = root/"allocations"/f"u{update:04d}.json"
    if not allocation.exists(): return
    value = json.loads(allocation.read_text())
    path = Path(value["resume_checkpoint"])
    parent = root/"elastic_resume"
    if path.parent.parent.resolve() != parent.resolve():
        from phase2.staged_storage import load_amendment
        amendment = load_amendment(root, update)
        if amendment is None: return  # native source; never remove it
        parent = Path(amendment["scratch_parent"])
        if path.parent.parent.resolve() != parent.resolve(): return
    target = path.parent
    if not target.exists(): return
    manifests = list(target.glob("global_step_*/actor/elastic_resume.json"))
    if len(manifests) != 1: raise ValueError("Missing complete conversion manifest")
    meta = json.loads(manifests[0].read_text())
    validate_full_checkpoint(Path(meta["source_checkpoint"]))
    archive_and_remove(target, parent, root/"storage_audit"/f"conversion-used-u{update:04d}",
                       "Temporary world-size conversion released after a new native checkpoint committed",
                       {"source_checkpoint": meta["source_checkpoint"], "new_checkpoint": str(root/"checkpoints"/f"global_step_{update}"),
                        "reconstructable_from_source_while_source_retained": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-approved-legacy-conversions", action="store_true")
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if not args.release_approved_legacy_conversions: parser.error("An explicit approved operation is required")
    release_legacy_conversions(args.archive)


if __name__ == "__main__": main()
