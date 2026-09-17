#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from phase1.archive import sha256_file, write_run_manifest


def output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as error:
        return f"unavailable: {error!r}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--skill-bank", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--rl-seed", type=int)
    parser.add_argument("--update-type", default="real_rl")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--command", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).parents[1]
    model_config = args.model / "config.json"
    config = {
        "stage": args.stage,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(model_config) if model_config.exists() else None,
        "skill_bank_path": str(args.skill_bank.resolve()),
        "protocol_path": str(args.protocol.resolve()) if args.protocol else None,
        "protocol_sha256": sha256_file(args.protocol) if args.protocol else None,
        "rl_seed": args.rl_seed,
        "update_type": args.update_type,
        "command": args.command,
    }
    extra = {
        "python": sys.version,
        "platform": platform.platform(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_list": json.loads(output(["conda", "list", "--json"])),
        "pip_freeze": output([sys.executable, "-m", "pip", "freeze"]).splitlines(),
        "gpu_inventory": output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]).splitlines(),
    }
    path = write_run_manifest(
        args.output_dir,
        args.run_id,
        config,
        repo_root,
        skill_bank_path=args.skill_bank,
        extra=extra,
    )
    print(path)


if __name__ == "__main__":
    main()
