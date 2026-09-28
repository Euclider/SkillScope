"""Prepare the registered 128-row GRPO path. Default invocation is a read-only plan."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from scripts.inspect_skillrl_alignment import load_config
from .common import digest, file_hash, load_preparation, read_json, require_authorization, write_new_json
from .runtime import ROUTER_PROFILES, authorized_runtime, disk_gate, is_embedding_backend, router_backend, runtime_settings, verify_runtime_identity


def configuration(preparation, run_root, router_calls=0, gpu_count=None, *, router_local_calls=0):
    preparation, run_root = Path(preparation).resolve(), Path(run_root).resolve()
    load_preparation(preparation)
    assets = preparation.parent
    spec = read_json(assets / "spec.json")
    if gpu_count is None:
        gpu_count = int(spec["training"]["gpus"])
    if not spec["seed_confirmed"] or spec["seed"] is None:
        raise ValueError("A new RL seed must be explicitly registered before composing a run")
    cfg = load_config()
    cfg.alignment_run = {
        "seed": spec["seed"], "model_path": spec["model_path"],
        "train_files": str(assets / "datasets/train.parquet"),
        "val_files": str(assets / "datasets/seen-monitor.parquet"),
        "run_id": f"skillnet37-s{spec['seed']}", "output_dir": str(run_root / "checkpoints"),
        "archive_dir": str(run_root / "trajectories"),
        "router_cache": str(run_root / "router.sqlite3"),
        "ray_temp_dir": f"/tmp/sn-{digest(str(run_root))[:12]}",  # Ray UNIX-socket path length.
    }
    cfg.phase2 = {"enabled": True, "root": str(run_root)}
    if spec['readout'].get('capture_scope'):
        cfg.phase2.capture_scope = spec['readout']['capture_scope']
        cfg.phase2.capture_updates = [window['start'] + 1 for window in spec['readout']['windows']]
    if spec.get('budget_profile'):
        cfg.phase2.progress_journal = True
    cfg.actor_rollout_ref.cohort_seed = int(spec["seed"])
    cfg.actor_rollout_ref.rollout.seed = int(spec['seed'])
    # A separately registered generation microbatch does not change the 16x8
    # GRPO groups or 128-row optimization minibatch. RNG order is versioned.
    cfg.actor_rollout_ref.rollout.micro_batch_size = spec['training'].get('rollout_microbatch_per_gpu', 1)
    if spec.get('inference_profile'):
        from .inference import apply_training
        apply_training(cfg, spec['inference_profile'])
    if 'actor_optimizer_offload' in spec['training']:
        cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload = spec['training']['actor_optimizer_offload']
    cfg.trainer.total_training_steps = int(spec['training']['iterations'])
    cfg.trainer.total_epochs = int(spec['training']['iterations'])
    cfg.actor_rollout_ref.actor.optim.total_training_steps = int(spec['training']['iterations'])
    if gpu_count not in (4, 8):
        raise ValueError('Only explicitly registered four/eight-GPU topologies are supported')
    cfg.trainer.n_gpus_per_node = gpu_count
    if gpu_count == 8:
        cfg.actor_rollout_ref.actor.fsdp_config.cpu_shard_init = True
        cfg.actor_rollout_ref.ref.fsdp_config.cpu_shard_init = True
    cfg.skillnet_cohort = {"enabled": True, "seed": spec["seed"], "data_root": spec["data_root"],
                          "preparation": str(preparation), "preparation_sha256": file_hash(preparation)}
    runtime = runtime_settings(spec, cfg.alignment_run.router_cache, router_calls, max_local_calls=router_local_calls)
    if is_embedding_backend(router_backend(spec)):
        cfg.env.skills_only_memory.step_routing = {
            "enabled": True, "backend": router_backend(spec),
            "profile_path": str(ROUTER_PROFILES[router_backend(spec)]),
            "model_path": runtime["router_model_path"], "device": runtime["router_device"],
            "cache_path": runtime["cache_path"], "max_local_calls": runtime["max_local_calls"],
        }
    else:
        cfg.env.skills_only_memory.step_routing.max_api_calls = int(router_calls)
    # Fail on actual full prompts; placeholder row filtering cannot remove games.
    cfg.data.filter_overlong_prompts = False
    OmegaConf.resolve(cfg)
    verify_runtime_identity(runtime)
    return cfg


def plan(preparation, run_root):
    cfg = configuration(preparation, run_root)
    return {"mode": "plan_only", "training_started": False, "external_api_calls": 0,
            "configuration": OmegaConf.to_container(cfg, resolve=True),
            "legacy_elastic_launcher_used": False,
            "required_before_execution": ["separate authorization and router/storage limits",
                                          "real GPU feasibility/restore validation; not inferred from CPU tests"]}


def reject_legacy_overrides():
    for name in ("PHASE2_ELASTIC_TRAINING", "PHASE2_CPU_ADAM"):
        if os.environ.get(name) == "1":
            raise ValueError(f"Unset legacy execution override: {name}")


def execute(preparation, run_root, authorization):
    permit = require_authorization(authorization, preparation, "training")
    run_root = Path(run_root).resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError("New-cohort launch will not resume/overwrite an existing run root")
    reject_legacy_overrides()
    gpu_ids = permit.get("gpu_ids", [])
    if len(gpu_ids) not in (4, 8) or len(set(map(str, gpu_ids))) != len(gpu_ids):
        raise ValueError("This recipe requires four/eight explicitly assigned GPU IDs")
    spec = read_json(Path(preparation).parent / "spec.json")
    cfg = configuration(preparation, run_root, permit.get("router_max_api_calls", 0), gpu_count=len(gpu_ids),
                        router_local_calls=permit.get("router_max_local_calls", 0))
    from .assets import model_inventory
    if model_inventory(cfg.alignment_run.model_path) != read_json(Path(preparation).parent / "model.json"):
        raise ValueError("B0 weights/tokenizer changed since preparation")
    if router_backend(spec) == "external_llm" and not os.environ.get("SKILLNET_ROUTER_API_KEY"):
        raise ValueError("Dedicated router credential must be injected into the process environment")
    limits = permit["storage"]
    disk_gate(run_root, int(limits["checkpoint_reserve_bytes"]),
              minimum_free_bytes=int(limits["minimum_free_bytes"]),
              maximum_run_bytes=int(limits["maximum_run_bytes"]))
    os.environ.update({"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
                       "ALFWORLD_DATA": cfg.skillnet_cohort.data_root,
                       "PYTHONDONTWRITEBYTECODE": "1"})
    # Local Ray children inherit the secret from the process environment.
    # Never insert credentials into serialized config/runtime_env or printed kwargs.
    write_new_json(run_root / "launch.json", {
        "preparation_sha256": file_hash(preparation), "authorization_sha256": file_hash(authorization),
        "configuration": OmegaConf.to_container(cfg, resolve=True),
        "runtime": authorized_runtime(spec, cfg.alignment_run.router_cache, permit)})
    write_new_json(run_root / "resource_limits.json", {**limits, "vocab_size": 248320})
    from verl.trainer.main_ppo import run_ppo
    run_ppo(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if args.execute:
        execute(args.preparation, args.run_root, args.authorization)
    else:
        print(json.dumps(plan(args.preparation, args.run_root), indent=2))


if __name__ == "__main__":
    main()
