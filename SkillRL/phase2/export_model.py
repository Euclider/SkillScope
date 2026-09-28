from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from safetensors import safe_open

from phase1.archive import atomic_write_json, utc_now
from phase1.watch_qwen35_checkpoints import validate_full_checkpoint


def merger_command(checkpoint: Path, temporary: Path):
    # A script pathname makes sys.path[0] the scripts/ directory, not cwd.
    # Module execution resolves the repository-local verl package without an
    # editable install or a caller's ambient PYTHONPATH.
    return [sys.executable, '-B', '-m', 'scripts.model_merger', 'merge',
            '--backend', 'fsdp', '--local_dir', str(checkpoint/'actor'),
            '--target_dir', str(temporary), '--output_dtype', 'float32']


def export(checkpoint: Path, target: Path, *, temporary: Path | None = None):
    checkpoint, target = Path(checkpoint).resolve(), Path(target).resolve()
    if (target/"phase2_export.json").exists():
        return
    if target.exists():
        raise FileExistsError('Never overwrite an unverified export target')
    metadata=validate_full_checkpoint(checkpoint)
    temporary = Path(temporary).resolve() if temporary is not None else target.with_name(target.name+'.partial')
    if temporary == target or temporary == checkpoint or temporary.is_relative_to(checkpoint):
        raise ValueError('Export staging must not overlap the checkpoint or target')
    temporary.mkdir(parents=True, exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    subprocess.run(merger_command(checkpoint, temporary), check=True, cwd=repo)
    tensor_count=0
    for file in temporary.glob("*.safetensors"):
        with safe_open(file,framework="pt") as handle:
            for key in handle.keys():
                if handle.get_slice(key).get_dtype() != "F32":
                    raise ValueError(f"Lossy or unexpected dtype for {key}")
                tensor_count+=1
    if tensor_count < 400:
        raise ValueError("Incomplete exported model")
    from .export_verify import verify
    parity = verify(checkpoint, temporary)
    atomic_write_json(temporary/"phase2_export.json",{
        "created_at":utc_now(),"parent":str(checkpoint),"dtype":"float32",
        "tensors":tensor_count,"source_validation":metadata,"native_weight_parity":parity,
        "model_bytes":sum(x.stat().st_size for x in temporary.glob("*.safetensors"))})
    temporary.rename(target)
    print(json.dumps({"exported":str(target),"dtype":"float32","tensors":tensor_count}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    a=p.parse_args()
    a.target.parent.mkdir(parents=True,exist_ok=True)
    export(a.checkpoint,a.target)


if __name__=="__main__":main()
