"""Two synthetic local embedding decisions plus replay; not a benchmark run.

An explicitly new output directory is required. No policy, ALFWorld, Ray,
external API, old trajectories, old cache or benchmark outcome is used.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agent_system.memory.skillnet_runtime import create_embedding_skillnet37_runtime, DEFAULT_EMBEDDING_ROUTER_PROFILE
from agent_system.memory.router_cache import utc_now
from skillnet_cohort.common import file_hash, write_new_json


SYNTHETIC_STATES = [
    {
        "task_description": "put a clean mug in cabinet 1",
        "current_observation": "You are at sinkbasin 1. You are holding mug 1, which has not yet been cleaned.",
        "admissible_actions": ["clean mug 1 with sinkbasin 1", "go to cabinet 1", "inventory"],
        "history": [
            {"observation": "On countertop 1, you see mug 1.", "action": "take mug 1 from countertop 1"},
            {"observation": "You pick up mug 1 from countertop 1.", "action": "go to sinkbasin 1"},
        ],
        "step_index": 2,
    },
    {
        "task_description": "put a clean mug in cabinet 1",
        "current_observation": "You are at cabinet 1. Cabinet 1 is open. You are holding mug 1, which is now clean.",
        "admissible_actions": ["put mug 1 in/on cabinet 1", "close cabinet 1", "inventory"],
        "history": [
            {"observation": "You are at sinkbasin 1 holding a dirty mug.", "action": "clean mug 1 with sinkbasin 1"},
            {"observation": "You clean mug 1 using sinkbasin 1.", "action": "go to cabinet 1"},
        ],
        "step_index": 4,
    },
]


def run_smoke(model_path, device, output, threads=4):
    import torch
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new smoke output; previous diagnostics must not be overwritten or rerun")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    # Initializing a CUDA context (if requested) is not part of the RNG test.
    cuda_devices = [torch.device(device).index] if device.startswith("cuda:") else []
    for index in cuda_devices:
        torch.cuda.get_rng_state(index)
    rng_before = {
        "torch_cpu": torch.get_rng_state().clone(), "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "cuda": [torch.cuda.get_rng_state(i).clone() for i in cuda_devices],
    }
    memory, router = create_embedding_skillnet37_runtime(model_path=model_path, device=device,
        cache_path=output / "router.sqlite3", max_local_calls=len(SYNTHETIC_STATES))
    started = time.monotonic()
    verification = {}
    try:
        def no_network(*args, **kwargs):
            raise AssertionError("Local embedding smoke forbids network connections")
        decisions = []
        with patch("socket.socket.connect", no_network), patch("socket.create_connection", no_network):
            for state in SYNTHETIC_STATES:
                print(f"Encoding synthetic state {len(decisions) + 1}/{len(SYNTHETIC_STATES)} on {device}", flush=True)
                result = router.route(memory.retrieve(""), **state)
                if result["skill_router_api"]["cache_hit"] or len(result["skill_router_scores"]) != 37:
                    raise AssertionError("Smoke must use a new complete 37-candidate decision")
                replay = router.route(memory.retrieve(""), **state)
                if (not replay["skill_router_api"]["cache_hit"] or replay["skill_router_api"]["local_calls_this_step"] != 0
                        or replay["skill_router_scores"] != result["skill_router_scores"]):
                    raise AssertionError("Cache replay differs from the frozen decision")
                decisions.append({"state": state, "selected_skill_id": result["selected_skill_id"],
                    "scores": result["skill_router_scores"], "accounting": result["skill_router_api"],
                    "payload_sha256": result["payload_sha256"], "replay_accounting": replay["skill_router_api"]})
        model = router._encoder.model
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rng_numpy = np.random.get_state()
        rng_checks = {
            "torch_cpu": torch.equal(rng_before["torch_cpu"], torch.get_rng_state()),
            "python": rng_before["python"] == random.getstate(),
            "numpy": rng_before["numpy"][0] == rng_numpy[0] and np.array_equal(rng_before["numpy"][1], rng_numpy[1])
                     and rng_before["numpy"][2:] == rng_numpy[2:],
            "cuda": all(torch.equal(before, torch.cuda.get_rng_state(i)) for i, before in zip(cuda_devices, rng_before["cuda"])),
        }
        verification = {"trainable_parameters": trainable, "model_training": model.training, "rng_preserved": rng_checks}
        if trainable != 0 or model.training or not all(rng_checks.values()):
            raise AssertionError("Router is not frozen/eval/RNG-independent")
        report = {
            "status": "passed", "kind": "synthetic_embedding_router_component_smoke", "created_at_utc": utc_now(),
            "benchmark_experiment": False, "training_started": False, "external_api_calls": 0,
            "profile_sha256": file_hash(DEFAULT_EMBEDDING_ROUTER_PROFILE), "protocol_hash": router.protocol_hash,
            "protocol": router.protocol, "device": device, "torch_threads": threads,
            "trainable_parameters": trainable, "rng_preserved": rng_checks,
            "states": decisions, "stats": router.stats(), "elapsed_seconds": round(time.monotonic() - started, 3),
            "performance_claim": "No ALFWorld policy success/coverage/8-GPU coexistence claim; these are synthetic states.",
        }
        write_new_json(output / "report.json", report)
        return report
    except BaseException as error:
        write_new_json(output / "failure.json", {"status": "failed", "exception_type": type(error).__name__,
                       "stats": router.stats(), "verification": verification,
                       "training_started": False, "external_api_calls": 0})
        raise
    finally:
        router.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = run_smoke(args.model, args.device, args.output, args.threads)
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json"),
        "stats": report["stats"], "selected": [s["selected_skill_id"] for s in report["states"]],
        "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
