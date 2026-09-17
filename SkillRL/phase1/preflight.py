#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from phase1.archive import sha256_file


def check(condition: bool, message: str, *, warning: bool = False) -> dict:
    return {"status": "warning" if warning and not condition else ("pass" if condition else "fail"), "message": message}


def check_cuda_devices(available: bool, count: int, minimum: int = 2) -> dict:
    if not 1 <= minimum <= 8:
        raise ValueError("Expected a CUDA minimum between 1 and 8")
    return check(available and count >= minimum,
                 f"PyTorch sees {count} CUDA devices; this launch requires at least {minimum}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/wangyifan/model/Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument(
        "--fallback-model",
        type=Path,
        default=Path("/home/wangyifan/model/Qwen2.5-1.5B-Instruct"),
    )
    parser.add_argument("--skill-bank", type=Path, default=Path("phase1/config/frozen_alfworld_skills.json"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--step-routing", action="store_true")
    parser.add_argument("--min-free-gb", type=int, default=2000)
    parser.add_argument("--min-cuda-devices", type=int, choices=range(1,9), default=2)
    parser.add_argument("--output", type=Path, default=Path("artifacts/preflight.json"))
    args = parser.parse_args()

    results = {}
    results["conda_env"] = check(os.environ.get("CONDA_DEFAULT_ENV") == "skill-RL", "Conda environment must be skill-RL")
    results["model"] = check(args.model.exists(), f"Model exists at {args.model}")
    results["fallback_model"] = check(
        args.fallback_model.exists(),
        f"Fallback model exists at {args.fallback_model}",
    )
    data_root = Path(os.environ.get("ALFWORLD_DATA", ""))
    results["alfworld_data"] = check(bool(str(data_root)) and (data_root / "json_2.1.1").exists(), "ALFWORLD_DATA contains json_2.1.1")
    bank = json.loads(args.skill_bank.read_text(encoding="utf-8"))
    ids = [item["skill_id"] for item in bank["general_skills"]]
    ids += [item["skill_id"] for items in bank["task_specific_skills"].values() for item in items]
    mistake_ids = [item["mistake_id"] for item in bank.get("common_mistakes", [])]
    if args.step_routing:
        bank_valid = (
            len(ids) == 44 and len(mistake_ids) == 11
            and len(ids + mistake_ids) == len(set(ids + mistake_ids))
        )
        bank_message = (
            "Full upstream Skill Bank has 44 Skill IDs and 11 mistake IDs; "
            f"sha256={sha256_file(args.skill_bank)}"
        )
    else:
        bank_valid = (
            len(ids) == 7 and len(ids) == len(set(ids))
            and bank.get("metadata", {}).get("frozen") is True
        )
        bank_message = (
            "Frozen smoke Skill Bank has 7 unique IDs; "
            f"sha256={sha256_file(args.skill_bank)}"
        )
    results["skill_bank"] = check(bank_valid, bank_message)
    if args.protocol:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        protocol_valid = (
            protocol.get("schema_version") == "phase1.protocol.v2"
            and protocol.get("step_routing", {}).get("enabled") is True
            and protocol.get("skill_bank_policy", {}).get("frozen_during_rl") is True
        ) if args.step_routing else bool(protocol.get("schema_version"))
        results["protocol"] = check(
            protocol_valid,
            f"Protocol {protocol.get('schema_version')}; sha256={sha256_file(args.protocol)}",
        )
    if args.step_routing:
        try:
            from agent_system.memory import FrozenStepSkillRouter, SkillsOnlyMemory

            memory = SkillsOnlyMemory(str(args.skill_bank), task_specific_top_k=None)
            bundle = memory.retrieve("put a clean apple in the fridge", top_k=12)
            routed = FrozenStepSkillRouter().route(
                bundle,
                task_description="put a clean apple in the fridge",
                current_observation="On countertop 1, you see an apple 1.",
                admissible_actions=["take apple 1 from countertop 1"],
                history=[],
                step_index=1,
            )
            results["step_router"] = check(
                routed.get("selected_skill_id") == "gen_002"
                and routed.get("injected_skill_ids") == ["gen_002"]
                and len(routed.get("candidate_skill_ids", [])) == 18,
                f"Frozen router selected {routed.get('selected_skill_id')} from "
                f"{len(routed.get('candidate_skill_ids', []))} enabled candidates",
            )
        except Exception as error:
            results["step_router"] = {"status": "fail", "message": repr(error)}
    free_gb = shutil.disk_usage(Path.cwd()).free // (1024 ** 3)
    results["storage"] = check(free_gb >= args.min_free_gb, f"Free storage {free_gb}GB; formal target is {args.min_free_gb}GB", warning=True)
    try:
        import torch
        results["cuda"] = check_cuda_devices(torch.cuda.is_available(), torch.cuda.device_count(), args.min_cuda_devices)
        results["torch_cuda"] = {"status": "pass", "message": f"torch={torch.__version__}, runtime={torch.version.cuda}"}
    except Exception as error:
        results["cuda"] = {"status": "fail", "message": repr(error)}
    for module in ("vllm", "alfworld", "textworld", "ray", "transformers", "pandas", "pyarrow"):
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "unknown")
            results[f"import:{module}"] = {"status": "pass", "message": str(version)}
        except Exception as error:
            results[f"import:{module}"] = {"status": "fail", "message": repr(error)}
    results["nvidia_smi"] = {
        "status": "pass" if shutil.which("nvidia-smi") else "fail",
        "message": subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failures = [name for name, result in results.items() if result["status"] == "fail"]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(f"Preflight failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
