"""Explicit FP32 snapshots for the B0 and five-update readout endpoints. No RL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import file_hash, load_preparation, read_json, require_authorization, write_new_json


def export_base(preparation, target, authorization):
    require_authorization(authorization, preparation, "exports")
    load_preparation(preparation)
    spec = read_json(Path(preparation).parent / "spec.json")
    target = Path(target).resolve()
    temporary = target.with_name(target.name + ".exporting")
    if target.exists() or temporary.exists():
        raise FileExistsError("Never overwrite a B0 snapshot")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Canonical text-only names and FP32 values must match later FSDP exports.
    # This is a CPU conversion only, not SFT/warm-up or an optimizer step.
    model = AutoModelForCausalLM.from_pretrained(spec["model_path"], dtype=torch.float32,
                                                attn_implementation="sdpa", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(spec["model_path"], local_files_only=True)
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    write_new_json(temporary / "phase2_export.json",
                   {"dtype": "float32", "source_model_manifest_sha256": file_hash(Path(preparation).parent / "model.json"),
                    "initialization": "unchanged post-trained text-only B0", "optimizer_steps": 0})
    temporary.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"operation": "fp32_export", "execute": False, "target": str(args.target),
                          "source": str(args.native_checkpoint) if args.native_checkpoint else "registered B0"}))
        return
    if args.native_checkpoint is None:
        export_base(args.preparation, args.target, args.authorization)
    else:
        permit = require_authorization(args.authorization, args.preparation, "exports")
        recovery = permit.get('post_training_recovery')
        temporary = None
        if recovery:
            from .post_training_recovery import verify_binding
            verify_binding(recovery, Path(permit['run_root']), args.preparation)
            if (args.target.resolve() != Path(permit['run_root'])/'models/u0005'
                    or args.native_checkpoint.resolve() != Path(permit['run_root'])/'checkpoints/global_step_5'):
                raise PermissionError('Post-training recovery only exports the retained U5 checkpoint')
            temporary = Path(recovery['export_staging'])
        if args.target.exists() or (temporary if temporary is not None else args.target.with_name(args.target.name + ".partial")).exists():
            raise FileExistsError("Never overwrite an endpoint or an interrupted export")
        from phase2.export_model import export
        export(args.native_checkpoint, args.target, temporary=temporary)


if __name__ == "__main__":
    main()
