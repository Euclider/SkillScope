from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from safetensors import safe_open

from phase1.archive import atomic_write_json, utc_now
from phase1.watch_qwen35_checkpoints import validate_full_checkpoint


def export(checkpoint: Path, target: Path):
    if (target/"phase2_export.json").exists():
        return
    metadata=validate_full_checkpoint(checkpoint)
    temporary=target.with_name(target.name+".partial")
    temporary.mkdir(parents=True, exist_ok=True)
    repo=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(repo/"scripts/model_merger.py"),"merge",
                    "--backend","fsdp","--local_dir",str(checkpoint/"actor"),
                    "--target_dir",str(temporary),"--output_dtype","float32"],check=True,cwd=repo)
    tensor_count=0
    for file in temporary.glob("*.safetensors"):
        with safe_open(file,framework="pt") as handle:
            for key in handle.keys():
                if handle.get_slice(key).get_dtype() != "F32":
                    raise ValueError(f"Lossy or unexpected dtype for {key}")
                tensor_count+=1
    if tensor_count < 400:
        raise ValueError("Incomplete exported model")
    atomic_write_json(temporary/"phase2_export.json",{
        "created_at":utc_now(),"parent":str(checkpoint),"dtype":"float32",
        "tensors":tensor_count,"source_validation":metadata,
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
