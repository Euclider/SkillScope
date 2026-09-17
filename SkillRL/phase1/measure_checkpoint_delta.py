#!/usr/bin/env python3
"""Measure an exact L2 parameter delta between two saved checkpoints."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from phase1.archive import atomic_write_json, stable_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-checkpoint", required=True)
    parser.add_argument("--post-checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pre = AutoModelForCausalLM.from_pretrained(
        args.pre_checkpoint, torch_dtype=torch.float32, device_map="cpu"
    )
    post = AutoModelForCausalLM.from_pretrained(
        args.post_checkpoint, torch_dtype=torch.float32, device_map="cpu"
    )
    post_parameters = dict(post.named_parameters())
    delta_squared = 0.0
    parameter_count = 0
    for name, parameter in pre.named_parameters():
        if name not in post_parameters or parameter.shape != post_parameters[name].shape:
            raise ValueError(f"Checkpoint parameter mismatch: {name}")
        delta = post_parameters[name].data - parameter.data
        delta_squared += float(torch.sum(delta * delta).item())
        parameter_count += parameter.numel()
    atomic_write_json(args.output, {
        "schema_version": "phase1.checkpoint_delta.v1",
        "pre_checkpoint": args.pre_checkpoint,
        "post_checkpoint": args.post_checkpoint,
        "parameter_count": parameter_count,
        "parameter_delta_l2_norm": math.sqrt(delta_squared),
        "config_hash": stable_hash(vars(args)),
    })


if __name__ == "__main__":
    main()
