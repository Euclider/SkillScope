"""Compare a new Phase3 batch router against immutable old CPU decisions.

This is an offline numerical/throughput check, not an environment rollout.
The output directory must be new; the old cache is opened read-only.
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

from agent_system.memory.router_cache import canonical_json
from skillnet_cohort.common import file_hash

from .bank import Bank
from .common import require, strict_json, write_new
from .embedding_routing import EmbeddingRouterPool, validate_settings


def run(old_cache, bank_path, bank_sha256, settings_path, output):
    old_cache, output = Path(old_cache).resolve(), Path(output).resolve()
    require(old_cache.is_file() and not output.exists(), "Old cache required; output must be new")
    settings = strict_json(Path(settings_path).read_text())
    validate_settings(settings)
    bank = Bank.load(bank_path, bank_sha256)
    with sqlite3.connect(f"file:{old_cache}?mode=ro", uri=True) as db:
        old_protocol = strict_json(db.execute("SELECT data FROM protocol WHERE id=1").fetchone()[0])
        rows = [strict_json(row[0]) for row in db.execute("SELECT record FROM decisions ORDER BY key")]
    require(rows and old_protocol["bank_manifest_sha256"] == bank.manifest_sha256,
            "Old decisions and supplied bank disagree")
    require(all(row["protocol_hash"] == rows[0]["protocol_hash"] for row in rows),
            "Mixed old decision protocols")
    pool = EmbeddingRouterPool(settings, output / "router-local.sqlite3", bank.branch_id)
    router = pool.for_bank(bank)
    candidate = router.memory.retrieve("")
    mismatches, maximum_abs_score_difference, minimum_old_margin = [], 0.0, float("inf")
    started = time.monotonic()
    try:
        for start in range(0, len(rows), 128):
            chunk = rows[start:start + 128]
            selected = router.route_many([{"candidate_bundle": candidate, **row["visible_input"]}
                                          for row in chunk])
            for offset, (old, new) in enumerate(zip(chunk, selected)):
                if old["selected_skill_id"] != new["selected_skill_id"]:
                    mismatches.append({"row": start + offset, "old": old["selected_skill_id"],
                                       "new": new["selected_skill_id"]})
                old_scores = old["scores"]
                new_scores = new["skill_router_scores"]
                require(set(old_scores) == set(new_scores), "Candidate catalog changed")
                maximum_abs_score_difference = max(maximum_abs_score_difference,
                    max(abs(old_scores[key] - new_scores[key]) for key in old_scores))
                top_two = sorted(old_scores.values(), reverse=True)[:2]
                minimum_old_margin = min(minimum_old_margin, top_two[0] - top_two[1])
    finally:
        pool.close()
    from .gpu_encoder_service import GPUEncoderProxy
    import torch
    report = {"status": "TOP1_PARITY" if not mismatches else "TOP1_MISMATCH",
        "old_cache": str(old_cache), "old_cache_sha256": file_hash(old_cache),
        "old_protocol_sha256": rows[0]["protocol_hash"],
        "new_protocol_sha256": router.protocol_hash,
        "bank_sha256": bank.manifest_sha256, "states": len(rows),
        "top1_matches": len(rows) - len(mismatches), "mismatches": mismatches,
        "maximum_abs_score_difference": maximum_abs_score_difference,
        "minimum_old_top2_margin": minimum_old_margin,
        "wall_seconds": time.monotonic() - started,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576
            if settings["device"].startswith("cuda:") and not isinstance(pool.encoder, GPUEncoderProxy)
            else None,
        "gpu_memory_scope": "sidecar; parent PyTorch allocation is not a valid peak"
            if isinstance(pool.encoder, GPUEncoderProxy) else "current_process",
        "new_router_local_attempts": len(rows), "external_api_calls": 0}
    write_new(output / "report.json", report)
    print(canonical_json({key: value for key, value in report.items() if key != "mismatches"}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-cache", type=Path, required=True)
    parser.add_argument("--bank-path", type=Path, required=True)
    parser.add_argument("--bank-sha256", required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.old_cache, args.bank_path, args.bank_sha256, args.settings, args.output)


if __name__ == "__main__":
    main()
