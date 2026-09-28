"""Read-only static readiness, distinct from authorization and GPU validation."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path

from .common import file_hash, load_preparation, read_json
from .runtime import ROUTER_PROFILES, embedding_profile, is_embedding_backend, router_backend, verify_runtime_identity, runtime_settings
from .training import configuration

PACKAGES = ("torch", "torchvision", "fastapi", "uvicorn", "transformers", "ray", "hydra-core", "tensordict", "torchdata",
            "pandas", "pyarrow", "textworld", "gymnasium", "alfworld", "openai")


def inspect(preparation, run_root):
    preparation = Path(preparation).resolve()
    load_preparation(preparation)
    assets = preparation.parent
    spec, games, controls = (read_json(assets / name) for name in ("spec.json", "games.json", "placebos.json"))
    bank = verify_runtime_identity(runtime_settings(spec, Path(run_root) / "router.sqlite3", 0))
    cfg = configuration(preparation, run_root)
    errors = []
    versions = {}
    backend = router_backend(spec)
    packages = (*PACKAGES, "sentence-transformers", "tokenizers") if is_embedding_backend(backend) else PACKAGES
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
            errors.append(f"Missing runtime distribution: {package}")
    if is_embedding_backend(backend):
        from agent_system.memory.skillrl_embedding_router import verify_snapshot
        router_config, router_files = embedding_profile(backend)
        for name in ("sentence-transformers", "transformers", "torch", "tokenizers"):
            if versions[name] != getattr(router_config, name.replace("-", "_") + "_version"):
                errors.append(f"Frozen embedding dependency changed: {name}")
        verify_snapshot(spec["router_model_path"], router_files)  # Hashing only, no encoder forward.
    import importlib
    entrypoints = {}
    for name in ("verl.trainer.main_ppo", "verl.workers.fsdp_workers", "phase2.measure",
                 "phase2.evaluate", "phase2.aggregate", "phase2.window_forecast"):
        try:
            importlib.import_module(name)  # No cluster, environment or model construction.
            entrypoints[name] = "imported"
        except ImportError as error:
            entrypoints[name] = type(error).__name__
            errors.append(f"Runtime entrypoint cannot import: {name}: {error}")
    from .assets import LocalTokenizer
    tokenizer = LocalTokenizer(spec["model_path"])
    if set(controls["controls"]) != set(bank.skill_ids):
        errors.append("PLACEBO inventory differs from frozen bank")
    for skill in bank.skills:
        control = controls["controls"][skill.skill_id]
        if (control["original_payload_sha256"] != skill.payload_sha256
                or len(tokenizer.encode(skill.payload)) != len(tokenizer.encode(control["text"]))):
            errors.append(f"Invalid token-matched control: {skill.skill_id}")
    if not cfg.phase2.enabled or cfg.actor_rollout_ref.actor.ppo_mini_batch_size != 128:
        errors.append("Readout capture or public-code mini-batch lost")
    for row in read_json(assets / "model.json")["files"]:
        path = Path(spec["model_path"]) / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            errors.append(f"Model inventory changed: {row['path']}")
    return {
        "status": "STATIC_READY_EXECUTION_LOCKED" if not errors else "STATIC_INVALID",
        "errors": errors, "versions": versions, "entrypoint_imports": entrypoints, "seed": spec["seed"],
        "router_backend": backend, "policy_gpu_count": int(cfg.trainer.n_gpus_per_node),
        "bank_count": len(bank.skills), "placebo_count": len(controls["controls"]),
        "game_counts": {split: value["count"] for split, value in games["splits"].items()},
        "training": f"{spec['training']['iterations']} iterations, 16 groups x 8; complete train pool sampling, not exhaustive epochs",
        "full_eval_job_count": sum(games["splits"][split]["count"] for split in spec["evaluation"]["splits"]),
        "readout_windows": len(spec["readout"]["windows"]), "capture_enabled": bool(cfg.phase2.enabled),
        "phase3_required": False, "training_started": False, "api_calls": 0, "writes": 0,
        "remaining_execution_gates": [
            "Explicit router/storage budgets and scoped execution authorization",
            f"Actual {cfg.trainer.n_gpus_per_node}-GPU forward/backward/restore feasibility with this router, not inferred from imports/CPU tests",
            "Runtime full-prompt coverage; strict overflow guard is installed, but no real states were run",
        ],
        "artifacts_generated_during_future_execution": [
            "FP32 B0/endpoint snapshots, natural-anchor support, committed predictions, and O/P/N outcomes"
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--require-execution-ready", action="store_true")
    args = parser.parse_args()
    result = inspect(args.preparation, args.run_root)
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)
    if args.require_execution_ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
