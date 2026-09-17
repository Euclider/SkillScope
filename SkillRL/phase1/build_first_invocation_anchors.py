#!/usr/bin/env python3
"""Build first-invocation anchors from checkpoint-0 FULL_BANK trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase1.archive import append_jsonl_idempotent
from phase1.first_invocation import build_anchor


def resolve_record_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-index", type=Path, required=True)
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--condition", default="full_bank")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).parents[1]
    anchors = []
    for line in args.trajectory_index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        index = json.loads(line)
        if index.get("checkpoint_id") != args.checkpoint_id:
            continue
        if index.get("skill_condition") != args.condition:
            continue
        trajectory_path = resolve_record_path(index["trajectory_path"], repo_root)
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        anchor = build_anchor(
            trajectory=trajectory,
            trajectory_path=str(trajectory_path),
            skill_id=args.skill_id,
            source_index=index,
        )
        if anchor is not None:
            anchors.append(anchor)
    if not anchors:
        raise ValueError("No eligible first-invocation anchors found")
    append_jsonl_idempotent(args.output, anchors, unique_fields=("anchor_id",))


if __name__ == "__main__":
    main()
