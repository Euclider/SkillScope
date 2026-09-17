from __future__ import annotations

import json
import argparse
from pathlib import Path

from phase1.archive import atomic_write_json, repo_worktree_fingerprint, sha256_file, utc_now
from phase2.protocol import validate_extended, evaluation_jobs


def main():
    repo = Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=repo / "phase2/config/semantic_direction_fast_v1.json")
    args=parser.parse_args()
    path=args.config
    config = json.loads(path.read_text())
    extended=config.get("schema_version","").startswith("phase2.extended.")
    if extended:validate_extended(config,repo)
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=True)
    (root/"logs").mkdir(exist_ok=True)
    target = root/"protocol.json"
    if target.exists() and json.loads(target.read_text()) != config:
        raise ValueError("Immutable protocol already exists with different content")
    atomic_write_json(target, config)
    if not (root/"manifest.json").exists():
        atomic_write_json(root/"manifest.json", {
            "created_at":utc_now(),"protocol_sha256":sha256_file(path),
            "registered_protocol_sha256":sha256_file(target),
            "repo_worktree_fingerprint":repo_worktree_fingerprint(repo),
            "phase1_report_sha256":sha256_file(repo.parent/"2026-09-04-qwen35-clean-all-skill-three-seed-utility-results.md"),
            "skill_bank_sha256":sha256_file(repo/"memory_data/alfworld/claude_style_skills.json"),
            "parent_checkpoint":config["parent_checkpoint"],
            "primary_control":"placebo", "training_started":False,
            "expected_suffixes_per_endpoint":len(evaluation_jobs(config,repo)) if extended else 1200,
        })
    print(root)


if __name__ == "__main__":
    main()
