"""Build NEW offline assets. Does not initialize Ray, ALFWorld, policy, or API."""
from __future__ import annotations

import argparse
from pathlib import Path

from .assets import LocalTokenizer, build_controls, game_inventory, model_inventory, write_placeholders
from .common import SCHEMA, file_hash, write_new_json
from .runtime import ROUTER_PROFILES, embedding_profile, is_embedding_backend, router_registration


def prepare(output, data_root, model_path, seed=None, *, router_backend="external_llm",
            router_model_path=None, router_device=None, gpu_count=4, budget_profile=None, inference_profile=None):
    from agent_system.memory.frozen_skill_bank import load_skillnet37
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preparation output must be a new, empty directory")
    if gpu_count not in (4, 8):
        raise ValueError("Only registered four/eight-GPU topologies are supported")
    routing = router_registration(router_backend, model_path=router_model_path, device=router_device)
    if is_embedding_backend(router_backend):
        from agent_system.memory.skillrl_embedding_router import verify_snapshot
        _, files = embedding_profile(router_backend)
        verify_snapshot(router_model_path, files)  # Read-only byte hashes, no forward.
    bank = load_skillnet37()
    games = game_inventory(data_root)
    model = model_inventory(model_path)
    controls = build_controls(bank, LocalTokenizer(model_path))
    spec = {
        "schema_version": SCHEMA, "status": "prepared_not_authorized",
        "seed": seed, "seed_confirmed": seed is not None,
        "bank_manifest_sha256": bank.manifest_sha256, **routing,
        "model_path": str(Path(model_path).resolve()), "data_root": str(Path(data_root).resolve()),
        "training": {"sampling": "SkillRL 16 groups x 8 trajectories; complete six-task train pool, not exhaustive traversal",
                     "iterations": 150, "games_per_iteration": 16, "group_size": 8,
                     "ppo_minibatch_rows": 128, "microbatch_per_gpu": 1, "gpus": gpu_count,
                     "save_every": 5, "monitor_every": 5, "monitor_episodes": 64},
        "evaluation": {"splits": ["valid_seen", "valid_unseen"], "full_game_traversal": True,
                       "temperature": 0.4, "top_p": 1.0, "max_steps": 50,
                       "max_new_tokens": 512, "max_prompt_tokens": 4096, "history_length": 2,
                       "performance_seeds": [61001], "anchor_source_seeds": [61011, 61021],
                       "evidence_seeds": [62011, 62021], "gold_seeds": [63011, 63021, 63031, 63041],
                       "minimum_anchor_occurrences": 30, "minimum_anchor_games": 10,
                       "maximum_anchors_per_skill": 50},
        "readout": {"horizon": 5, "windows": [{"start": u, "end": u + 5, "role": "test"} for u in range(0, 150, 5)],
                    "anchor_strategy": "natural calls at each window start, fixed across paired endpoints",
                    "primary_score": "gated_D", "fit_new_models": False,
                    "support": {"minimum_nonzero_decisions": 20, "minimum_games": 4, "minimum_trajectories": 8},
                    "unsupported": "abstain; never assign zero risk",
                    "capture": "actual batch/advantage/old-new FP32 vocabulary distributions at every update",
                    "score_frozen_before_gold": True},
        "prompt_audit": {"payload_min_tokens": min(c["target_token_count"] for c in controls["controls"].values()),
                         "payload_max_tokens": max(c["target_token_count"] for c in controls["controls"].values()),
                         "placebos": len(controls["controls"]), "complete_runtime_prompt_verified": False,
                         "overflow_policy": "raise before generation; no silent truncation/drop"},
        "phase3": {"enabled": False, "required_for_phase12": False},
        "execution": {"approved": False, "router_max_api_calls": 0, "router_max_local_calls": 0,
                      "gpu_forward_validated": False, "storage_budget_bytes": None},
    }
    if budget_profile is not None:
        from .day_budget import apply_budget
        spec = apply_budget(spec, budget_profile)
    if inference_profile is not None:
        from .inference import registration
        spec['inference_profile'] = registration(inference_profile)
    for name, value in [("games.json", games), ("model.json", model), ("placebos.json", controls), ("spec.json", spec)]:
        write_new_json(output / name, value)
    write_placeholders(output / "datasets/train.parquet", 16, "train")
    write_placeholders(output / "datasets/seen-monitor.parquet", 64, "test")
    names = ["games.json", "model.json", "placebos.json", "spec.json",
             "datasets/train.parquet", "datasets/seen-monitor.parquet"]
    manifest = {"schema_version": SCHEMA, "kind": "preparation",
                "assets": [{"path": name, "sha256": file_hash(output / name)} for name in names],
                "experiments_started": False}
    write_new_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--router-backend", choices=list(ROUTER_PROFILES), default="external_llm")
    parser.add_argument("--router-model", type=Path)
    parser.add_argument("--router-device", help="Explicit cpu or cuda:N, relative to visible devices")
    parser.add_argument("--gpus", type=int, choices=[4, 8], default=4)
    parser.add_argument("--budget-profile", type=Path)
    parser.add_argument("--inference-profile", type=Path)
    args = parser.parse_args()
    import json
    print(json.dumps(prepare(args.output, args.data_root, args.model, args.seed,
        router_backend=args.router_backend, router_model_path=args.router_model,
        router_device=args.router_device, gpu_count=args.gpus, budget_profile=args.budget_profile,
        inference_profile=args.inference_profile), indent=2))


if __name__ == "__main__":
    main()
