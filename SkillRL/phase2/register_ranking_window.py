"""Register the user-approved first U35->U40 ranking window, before new RL."""
import argparse
import copy
import json
from pathlib import Path

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
from phase2.protocol import validate_extended


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = repo/"artifacts/phase2/qwen35-clean-s303-u35-ranking-preparation-v1"
    old = json.loads((source/"protocol.json").read_text())
    queue = json.loads((source/"queue.json").read_text())
    cfg = json.loads((repo/"phase2/config/extended_direction_v2.template.json").read_text())
    root = repo/"artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1"
    cfg.update(status="frozen", registered_at=utc_now(),
               note="User approved first 5-update window on 2026-09-13. Direct within-window ranking is primary; no fitted regressor is needed. Not a multi-window or cross-seed result.",
               run_id=root.name, root=str(root), rl_path_id="seed303-continued-u35-to40-ranking-v1",
               parent_rl_seed=303, parent_checkpoint=queue["source_checkpoint"], parent_update=35,
               post_updates=list(range(36, 41)), windows=[{"start": 35, "end": 40, "role": "test"}],
               baseline_source=str(source), baseline_protocol_sha256=sha256_file(source/"protocol.json"),
               report_path=str(repo.parent/"2026-09-13-phase2-u35-to40-ranking-execution.md"))
    cfg.pop("window_design_proposal", None)
    cfg["evaluation"] = copy.deepcopy(old["evaluation"])
    cfg["training"].update(run_id_prefix="phase2-s303-ranking-v1", sampler_seed_base=30300)
    cfg["prediction"]["enable_fitted_models"] = False
    cfg["ranking"] = copy.deepcopy(queue["ranking_plan"])
    cfg["ranking"]["scores"].update({"u_original_norm": 1, "u_control_norm": 1, "old_margin": -1,
                                    **{f"activation_l{i}_norm": 1 for i in (8, 16, 24, 32)}})
    cfg["ranking"]["shared_candidate_pool"] = "Identical naturally supported complete-case Skills for all fixed scores, separately within each window/context/phase"
    cfg["ranking"]["bootstrap_status"] = "Paired game/continuation CIs for each utility change; ranking metrics descriptive in this first single window, not a cross-window significance claim"
    cfg["storage"] = {"rolling_recovery_authorized": True, "keep_full_recovery": 2,
                       "release_completed_conversions": True, "retain_every_fp32_policy": True,
                       "retain_all_live_logprobs_and_alignment": True,
                       "boundary_reserve_gib": 250, "projected_transient_reserve_gib": 200,
                       "minimum_next_update_allowance_gib": 100,
                       "note": "Three full checkpoints can transiently coexist while the newest commits. Rotation happens only after native commit and validation. No old source cohort deletion except separately approved six conversion copies."}
    validate_full_checkpoint(Path(cfg["parent_checkpoint"]))
    validate_extended(cfg, repo)
    if a.output.exists():
        existing = json.loads(a.output.read_text())
        cfg["registered_at"] = existing["registered_at"]
        if existing != cfg: raise ValueError("Immutable window registration already differs")
    else: atomic_write_json(a.output, cfg)
    print(a.output)


if __name__ == "__main__": main()
