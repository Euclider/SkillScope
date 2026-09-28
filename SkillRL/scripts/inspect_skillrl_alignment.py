"""Read-only configuration audit. Never creates environments, files, or API clients.

This is intentionally NOT an experiment launcher. The opt-in Hydra config is
prepared for a future approved runner; all unresolved execution gates are visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.errors import InterpolationToMissingValueError, MissingMandatoryValue


CODE_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = CODE_ROOT / "configs/skillrl_public_alignment_v1.json"
CONFIG_PATH = CODE_ROOT / "verl/trainer/config/alfworld_skillnet37_skillrl_v1.yaml"


def load_config(path: Path = CONFIG_PATH):
    """Compose this one documented Hydra defaults list without the training stack."""
    overlay = OmegaConf.load(path)
    if OmegaConf.to_container(overlay.get("defaults")) != ["ppo_trainer", "_self_"]:
        raise ValueError("Unexpected defaults; inspect the changed composition explicitly")
    del overlay["defaults"]
    return OmegaConf.merge(OmegaConf.load(path.parent / "ppo_trainer.yaml"), overlay)


def inspect_alignment(config=None, profile=None):
    if profile is None:
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if config is None:
        config = load_config()
    errors = []
    for section in ("aligned_values", "study_overrides"):
        for key, expected in profile[section].items():
            try:
                actual = OmegaConf.select(config, key, throw_on_missing=True)
                if OmegaConf.is_config(actual):
                    actual = OmegaConf.to_container(actual, resolve=True)
                if actual != expected:
                    errors.append(f"{key}: expected {expected!r}, observed {actual!r}")
            except Exception as error:
                errors.append(f"{key}: {type(error).__name__}")

    derived = {}
    try:
        groups = int(config.data.train_batch_size)
        env_n = int(config.env.rollout.n)
        iterations = int(config.trainer.total_training_steps)
        world = int(config.trainer.nnodes * config.trainer.n_gpus_per_node)
        sp = int(config.actor_rollout_ref.actor.ulysses_sequence_parallel_size)
        mini = int(config.actor_rollout_ref.actor.ppo_mini_batch_size)
        generation_n = int(config.actor_rollout_ref.rollout.n)
        micro = int(config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu)
        if world <= 0 or sp <= 0 or world % sp or mini <= 0 or micro <= 0:
            raise ValueError("invalid batch topology")
        dp = world // sp
        if mini * generation_n % dp:
            raise ValueError("global mini-batch not divisible by data-parallel size")
        local_mini = mini * generation_n // dp
        if local_mini % micro:
            raise ValueError("local mini-batch not divisible by micro-batch")
        derived = {
            "trajectories_per_iteration": groups * env_n,
            "nominal_training_trajectories": groups * env_n * iterations,
            "local_minibatch_rows_full": local_mini,
            "gradient_accumulation_full_minibatch": local_mini // micro,
            "opportunity_steps": list(range(config.trainer.test_freq, iterations + 1, config.trainer.test_freq)),
        }
        for key in ("trajectories_per_iteration", "nominal_training_trajectories",
                    "local_minibatch_rows_full", "gradient_accumulation_full_minibatch"):
            if derived[key] != profile["derived"][key]:
                errors.append(f"Derived {key} differs from the archived contract")
        if derived["opportunity_steps"] != profile["phase3"]["opportunity_steps"]:
            errors.append("Edit opportunities differ from the archived schedule")
    except Exception as error:
        errors.append(f"Batch/schedule calculation failed: {type(error).__name__}")

    bank = profile["bank_router"]
    bank_hash = hashlib.sha256((CODE_ROOT / bank["bank_manifest"]).read_bytes()).hexdigest()
    if bank_hash != bank["bank_manifest_sha256"]:
        errors.append("Frozen initial-bank manifest hash differs")
    router_profile = json.loads((CODE_ROOT / bank["router_profile"]).read_text(encoding="utf-8"))
    if router_profile.get("router", {}).get("model") != bank["router_model"]:
        errors.append("Router model differs from the archived contract")

    # Inspect only the declared inputs: no model/data/cache path is accessed.
    # A mandatory interpolation can raise from OmegaConf.missing_keys(), so
    # check each alias explicitly without resolving the entire training config.
    missing = []
    for key in profile["required_unset_fields"]:
        try:
            value = OmegaConf.select(config, key, throw_on_missing=True)
        except (MissingMandatoryValue, InterpolationToMissingValueError):
            missing.append(key)
        else:
            # This is the distributed setting template, not an authorized run
            # manifest. Runtime inputs must remain visibly unassigned here.
            errors.append(f"{key}: required template input is already assigned ({type(value).__name__})")
    if (profile["phase3"]["edit_operation"] != "external_llm_decides_add_delete_modify"
            or profile["phase3"]["allowed_edit_operations"] != ["add", "delete", "modify"]):
        errors.append("Phase3 must preserve the user-confirmed shared external-editor action space")
    if profile["phase3"]["approved_for_execution"]:
        errors.append("This setting-only profile is not execution approval")
    return {
        "setting_id": profile["setting_id"],
        "status": "CONFIGURATION_VALIDATED_NOT_RUN" if not errors else "CONFIGURATION_INVALID",
        "errors": errors,
        "derived": derived,
        "required_runtime_values_unset": sorted(missing),
        "pending_before_launch": profile["pending_before_launch"],
        "phase3_status": profile["phase3"]["status"],
        "phase3_edit_operation": profile["phase3"]["edit_operation"],
        "full_game_coverage_verified": False,
        "gpu_forward_validated": False,
        "training_started": False,
        "external_api_calls": 0,
        "writes": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-ready", action="store_true",
                        help="Fail if any runtime value or prelaunch gate remains open")
    args = parser.parse_args()
    result = inspect_alignment()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["errors"]:
        return 1
    if args.require_ready and (result["required_runtime_values_unset"] or result["pending_before_launch"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
