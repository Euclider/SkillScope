"""Archive source and environment used by the standalone Phase2 pipeline."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import zipfile

from phase1.archive import atomic_write_json, sha256_file, stable_hash, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    files = sorted(set(repo.joinpath("phase2").rglob("*.py")) |
                   set(repo.joinpath("phase2/config").glob("*.json")) |
                   set(repo.joinpath("tests/phase2").glob("*.py")))
    files += [repo / name for name in (
        "phase2/run_training_update.sh", "phase2/config/semantic_direction_fast_v1.json",
        "phase2/config/elastic_resource_policy_v1.json",
        "phase2/config/signed_analysis_amendment_v2.json",
        "phase2/config/resource_watch_amendment_v2.json",
        "phase1/preflight.py", "examples/grpo_trainer/run_alfworld_phase1_step_router.sh",
        "agent_system/environments/env_manager.py",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "verl/trainer/ppo/ray_trainer.py", "verl/workers/actor/dp_actor.py",
        "verl/workers/fsdp_workers.py", "verl/utils/checkpoint/fsdp_checkpoint_manager.py",
        "scripts/model_merger.py", "memory_data/alfworld/claude_style_skills.json",
    )]
    files=sorted(set(files))
    hashes = {str(path.relative_to(repo)): sha256_file(path) for path in files}
    identity = stable_hash(hashes)
    directory = args.root.resolve()/"source_snapshots"/identity[:16]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"manifest.json").exists():
        print(directory)
        return
    with zipfile.ZipFile(directory/"source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(repo)))
    packages = {}
    for name in ("torch", "transformers", "numpy", "scipy", "pandas", "scikit-learn",
                 "tabulate", "matplotlib", "safetensors", "ray", "flash-linear-attention"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    atomic_write_json(directory/"manifest.json", {
        "created_at": utc_now(), "source_hash": identity, "files": hashes,
        "packages": packages, "protocol_sha256": sha256_file(args.root/"protocol.json"),
        "note": "Snapshot of source files; run-stage logs determine which modules executed.",
    })
    print(directory)


if __name__ == "__main__":
    main()
