"""Opt-in execution-only admission: durable evidence, tmpfs conversion scratch.

No science config, training tensors, RNG, or retention rule is changed. The
runner commits the new native checkpoint and rotates recovery BEFORE export.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import psutil

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase1.watch_qwen35_checkpoints import read_committed_step, validate_full_checkpoint, validate_model_only

GIB = 2**30
AMENDMENT = "staged_storage_amendment.json"


def staged_peak(total, deferred_export, reclaim):
    """Peak before commit/rotation, or after deferred export; not total writes."""
    if min(total, deferred_export, reclaim) < 0 or deferred_export > total:
        raise ValueError("Invalid staged storage budget")
    return max(total - deferred_export, total - reclaim)


def load_amendment(root, update):
    root = Path(root).resolve()
    file = root/AMENDMENT
    if not file.exists(): return None
    value = json.loads(file.read_text())
    if value["root"] != str(root) or value["protocol_sha256"] != sha256_file(root/"protocol.json"):
        raise ValueError("Storage amendment does not match immutable protocol")
    if value["update"] != update: return None
    scratch = Path(value["scratch_parent"])
    if (scratch.is_symlink() or scratch.parent != Path("/dev/shm") or
            not scratch.name.startswith("skillrl-phase2-") or scratch.stat().st_uid != os.getuid()):
        raise ValueError("Unowned or unsafe tmpfs scratch")
    marker = json.loads((scratch/"owner.json").read_text())
    if marker != {k: value[k] for k in ("root", "update", "protocol_sha256")}:
        raise ValueError("Scratch ownership marker mismatch")
    if not any(p.mountpoint == "/dev/shm" and p.fstype == "tmpfs" for p in psutil.disk_partitions(all=True)):
        raise ValueError("Scratch filesystem is not tmpfs")
    return value


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def storage_inputs(root, config, update):
    root = Path(root)
    settings = config["storage"]
    if not settings.get("rolling_recovery_authorized") or settings.get("keep_full_recovery") != 2:
        raise ValueError("Staged admission requires authorized latest-two retention")
    if read_committed_step(root/"checkpoints") != update-1:
        raise ValueError("Expected a committed parent recovery boundary")
    present = sorted((int(p.name.removeprefix("global_step_")), p)
                     for p in (root/"checkpoints").glob("global_step_*") if p.is_dir())
    if len(present) != 2 or present[-1][0] != update-1:
        raise ValueError("Expected exactly two native parent recovery checkpoints")
    for _, path in present: validate_full_checkpoint(path)
    old, target = present[0]
    if old not in config["post_updates"]: raise ValueError("Recovery target outside this run")
    model = root/"models"/f"u{old:04d}"
    validate_model_only(model)
    if json.loads((model/"phase2_export.json").read_text())["dtype"] != "float32":
        raise ValueError("Lossless older policy must be retained")
    if not (root/"signals"/f"u{old:04d}"/"committed.json").exists():
        raise ValueError("Older single-update signals must already be committed")
    latest_model = root/"models"/f"u{update-1:04d}"
    validate_model_only(latest_model)
    export = json.loads((latest_model/"phase2_export.json").read_text())
    if export["dtype"] != "float32": raise ValueError("Expected FP32 parent export")
    return {"deferred_export_bytes": export["model_bytes"], "reclaim_bytes": tree_bytes(target),
            "source_checkpoint_bytes": tree_bytes(present[-1][1]), "rotation_target": str(target)}


def admit(root, config, update, *, tokens=None, world=None, source_world=None):
    amendment = load_amendment(root, update)
    if amendment is None: return None
    values = storage_inputs(root, config, update)
    settings = config["storage"]
    total = settings["minimum_next_update_allowance_gib"]*GIB
    if tokens is not None:
        # Both live OLD and NEW capture all attention-valid response tokens in
        # FP32, including padded decision rows. Native state allowance includes
        # world-size padding/metadata; miscellaneous allowance covers batch,
        # trajectory archives, serialization headers, and small signal files.
        capture = int(tokens)*int(config["signals"]["vocab_size"])*4*2
        checkpoint = int(values["source_checkpoint_bytes"]*1.05) + GIB
        total = max(total, capture + checkpoint + values["deferred_export_bytes"] + 5*GIB)
        values.update(response_tokens=int(tokens), live_old_new_bytes=capture,
                      native_checkpoint_allowance_bytes=checkpoint)
    peak = staged_peak(total, values["deferred_export_bytes"], values["reclaim_bytes"])
    free = shutil.disk_usage(root).free
    floor = settings["projected_transient_reserve_gib"]*GIB
    record = {"at": utc_now(), "update": update, "protocol_sha256": amendment["protocol_sha256"],
              **values, "total_write_allowance_bytes": total, "peak_net_growth_bytes": peak,
              "disk_free_bytes": free, "reserve_bytes": floor, "admitted": free-peak >= floor,
              "stage": "actual_batch_before_old_forward" if tokens is not None else "prelaunch"}
    atomic_write_json(Path(root)/"storage_audit"/f"admission-u{update:04d}-{record['stage']}.json", record)
    if not record["admitted"]:
        raise RuntimeError(f"Staged disk reserve insufficient: {free/GIB:.1f}GiB free, {peak/GIB:.1f}GiB peak, {floor/GIB:.0f}GiB reserve")
    if world is not None and world != source_world:
        scratch = Path(amendment["scratch_parent"])
        target = scratch/f"u{update-1:04d}-w{world}"/f"global_step_{update-1}"
        needed = 0 if (target/"actor/elastic_resume.json").exists() else 55*GIB
        if shutil.disk_usage(scratch).free-needed < 64*GIB or psutil.virtual_memory().available-needed < 190*GIB:
            raise RuntimeError("Insufficient tmpfs/host memory reserve for recoverable conversion scratch")
    return record


def register(root, update):
    root = Path(root).resolve()
    config = json.loads((root/"protocol.json").read_text())
    if (root/AMENDMENT).exists():
        if load_amendment(root, update) is None: raise ValueError("Amendment already bound to another update")
        return admit(root, config, update)
    if update != config["post_updates"][-1]: raise ValueError("Execution amendment is scoped to final U40 only")
    storage_inputs(root, config, update)
    if shutil.disk_usage(root).free < config["storage"]["boundary_reserve_gib"]*GIB:
        raise RuntimeError("Boundary reserve insufficient")
    scratch = Path(tempfile.mkdtemp(prefix=f"skillrl-phase2-u{update}-", dir="/dev/shm"))
    identity = {"root": str(root), "update": update, "protocol_sha256": sha256_file(root/"protocol.json")}
    atomic_write_json(scratch/"owner.json", identity)
    value = {**identity, "created_at": utc_now(), "scratch_parent": str(scratch),
             "scope": "execution resources only; scientific protocol and 200/250GiB safeguards unchanged",
             "order": "durable captures/native commit -> verified latest-two rotation -> FP32 export",
             "volatile_data": "only reconstructable conversion; native source and every experimental evidence file remain on persistent disk"}
    atomic_write_json(root/AMENDMENT, value)
    return admit(root, config, update)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(register(args.root, args.update), indent=2))
