"""Shared protocol contracts for repeated evaluations and frozen update windows.

The historical fast-v1 file is never edited. Expanded cohorts use explicit
anchor sets, endpoint windows, branch identities, and independent seed streams.
"""
from __future__ import annotations

import json
from pathlib import Path

from phase1.archive import sha256_file, stable_hash


def parent_update(config):
    return int(config.get("parent_update", min(config.get("post_updates", [31])) - 1))


def training_run_id(config, update):
    return f"{config.get('training', {}).get('run_id_prefix', 'phase2-s303-fast')}-u{update}"


def seed_streams(config):
    ev = config["evaluation"]
    streams = {"evidence": ev["old_evidence_seeds"], "gold": ev["gold_seeds"]}
    for name, seeds in streams.items():
        if not seeds or len(seeds) != len(set(seeds)) or not all(isinstance(x, int) for x in seeds):
            raise ValueError(f"Nonempty, unique integer continuation seeds required: {name}")
    if set(streams["gold"]) & set(streams["evidence"]):
        raise ValueError("Evidence and gold continuation seeds must be disjoint")
    if set(ev["arms"]) != {"original", "placebo", "null"} or len(ev["arms"]) != 3:
        raise ValueError("Exactly the ORIGINAL/PLACEBO/NULL arms are required")
    return streams


def anchor_sets(config, repo):
    ev = config["evaluation"]
    sets = ev.get("anchor_sets")
    if sets is None:
        sets = [{"skill_id": skill, "context_id": None,
                 "anchors_path": str(Path(ev["anchors_dir"]) / f"{skill}.jsonl"),
                 "placebo_path": str(Path(ev["placebo_dir"]) / f"{skill}.json")}
                for skill in ev["skills"]]
    if not sets:
        raise ValueError("No frozen anchor sets; run pre-update coverage audit first")
    result, seen = [], set()
    for spec in sets:
        key = (spec["skill_id"], spec.get("context_id"))
        if key in seen:
            raise ValueError(f"Duplicate Skill/context anchor set: {key}")
        seen.add(key)
        path = Path(repo) / spec["anchors_path"]
        placebo_path = Path(repo) / spec["placebo_path"]
        for file, field in [(path, "anchors_sha256"), (placebo_path, "placebo_sha256")]:
            if field in spec and sha256_file(file) != spec[field]:
                raise ValueError(f"Frozen asset changed: {file}")
        anchors = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not anchors or len({a["anchor_id"] for a in anchors}) != len(anchors):
            raise ValueError(f"Empty or duplicate anchor set: {path}")
        for a in anchors:
            if a["skill_id"] != spec["skill_id"] or (spec.get("context_id") is not None and a["context_id"] != spec["context_id"]):
                raise ValueError(f"Anchor Skill/context mismatch: {path}")
        result.append({**spec, "anchors": anchors, "placebo": json.loads(placebo_path.read_text())})
    count = sum(len(s["anchors"]) for s in result)
    if ev.get("anchor_count", count) != count:
        raise ValueError("Declared anchor_count does not match frozen assets")
    return result


def evaluation_jobs(config, repo):
    streams = seed_streams(config)
    jobs = []
    for spec in anchor_sets(config, repo):
        for anchor in spec["anchors"]:
            for purpose, seeds in streams.items():
                for seed in seeds:
                    for arm in config["evaluation"]["arms"]:
                        jobs.append((spec["skill_id"], anchor, purpose, seed, arm, spec["placebo"]))
    return jobs


def evaluation_identity(config, update, job):
    skill, anchor, purpose, seed, arm, _ = job
    # Preserve legacy trajectory IDs exactly; anchor_id already identifies context.
    return {"run_id": config["run_id"], "update": update, "anchor_id": anchor["anchor_id"],
            "skill_id": skill, "purpose": purpose, "continuation_seed": seed, "arm": arm}


def expected_trajectory_ids(config, repo, update):
    return [stable_hash(evaluation_identity(config, update, job))[:24] for job in evaluation_jobs(config, repo)]


def controls_by_skill(config, repo):
    result = {}
    for spec in anchor_sets(config, repo):
        skill, value = spec["skill_id"], spec["placebo"]
        if skill in result and result[skill] != value:
            raise ValueError(f"Different context-specific payloads for {skill}; use distinct bound Skill IDs")
        result[skill] = value
    return result


def signal_directory(root, end, start=None):
    return Path(root) / "signals" / f"u{end:04d}" if start is None else Path(root) / "window_signals" / f"u{start:04d}-u{end:04d}"


def validate_windows(windows):
    seen = set()
    for window in windows:
        key = (window["start"], window["end"])
        if key in seen or key[0] >= key[1]:
            raise ValueError(f"Duplicate or nonpositive window: {key}")
        seen.add(key)
        if window["role"] not in ("development", "boundary", "test", "sensitivity"):
            raise ValueError("Unknown window role")
    primary = [w for w in windows if w["role"] != "sensitivity"]
    primary.sort(key=lambda w: w["start"])
    for left, right in zip(primary, primary[1:]):
        if left["end"] > right["start"]:
            raise ValueError("Primary update windows overlap")
    dev = [w for w in windows if w["role"] == "development"]
    test = [w for w in windows if w["role"] == "test"]
    if dev and test and max(w["end"] for w in dev) >= min(w["start"] for w in test):
        raise ValueError("Development/test windows share an endpoint or cross in time; insert a boundary window")
    return windows


def make_windows(start, horizon, development, boundary, test):
    if min(horizon, development, boundary, test) < 1:
        raise ValueError("Positive horizon and nonempty development/boundary/test segments required")
    roles = ["development"] * development + ["boundary"] * boundary + ["test"] * test
    return validate_windows([{"start": start + i * horizon, "end": start + (i+1) * horizon, "role": role}
                             for i, role in enumerate(roles)])


def validate_extended(config, repo, require_frozen=True):
    if not config.get("schema_version", "").startswith("phase2.extended."):
        raise ValueError("Expected an explicit new extended protocol, not the historical fast cohort")
    if require_frozen and config.get("status") != "frozen":
        raise ValueError("Draft protocol cannot launch; freeze after coverage/calibration and branch selection")
    skillnet = config.get("runtime", {}).get("kind") == "skillnet37"
    if skillnet:
        from skillnet_cohort.runtime import verify_runtime_identity
        verify_runtime_identity(config["runtime"])
    if require_frozen and config.get("root"):
        root=Path(config["root"])
        manifest=root/"manifest.json"
        if manifest.exists():
            saved=json.loads(manifest.read_text())
            if saved.get("registered_protocol_sha256") and sha256_file(root/"protocol.json")!=saved["registered_protocol_sha256"]:
                raise ValueError("Registered protocol changed after preparation")
            bank=Path(repo)/("memory_data/alfworld/skillnet37/manifest.json" if skillnet
                             else "memory_data/alfworld/claude_style_skills.json")
            if saved.get("skill_bank_sha256") and sha256_file(bank)!=saved["skill_bank_sha256"]:
                raise ValueError("Skill Bank changed after protocol preparation")
    seed_streams(config)
    windows = validate_windows(config["windows"])
    if not windows:
        raise ValueError("No predeclared update windows")
    start = parent_update(config)
    if min(w["start"] for w in windows) != start:
        raise ValueError("Window schedule must start at the registered parent update")
    if config["post_updates"] != list(range(start + 1, max(w["end"] for w in windows) + 1)):
        raise ValueError("Capture every intermediate update even when evaluating wider windows")
    if config.get("window_direction") != "start_batch_endpoint_projection":
        raise ValueError("Window direction must explicitly use the first batch's old policy/advantage; never sum D")
    sets = anchor_sets(config, repo)
    if require_frozen:
        for spec in config["evaluation"]["anchor_sets"]:
            if not all(key in spec for key in ("anchors_sha256", "placebo_sha256", "context_id")):
                raise ValueError("Frozen anchor assets require hashes and context IDs")
        if not config.get("rl_path_id") or not config["training"].get("run_id_prefix"):
            raise ValueError("Explicit independent path and training artifact namespace required")
    return sets
