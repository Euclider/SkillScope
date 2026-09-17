#!/usr/bin/env python3
"""Create a random-direction checkpoint with norm matched to a real update."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase1.archive import atomic_write_json, stable_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-checkpoint", required=True)
    parser.add_argument("--post-checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    pre = AutoModelForCausalLM.from_pretrained(args.pre_checkpoint, torch_dtype=torch.float32, device_map="cpu")
    post = AutoModelForCausalLM.from_pretrained(args.post_checkpoint, torch_dtype=torch.float32, device_map="cpu")
    post_parameters = dict(post.named_parameters())
    delta_sq = 0.0
    for name, parameter in pre.named_parameters():
        delta = post_parameters[name].data - parameter.data
        delta_sq += float(torch.sum(delta * delta).item())
    delta_norm = math.sqrt(delta_sq)
    del post

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    random_sq = 0.0
    for parameter in pre.parameters():
        noise = torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
        random_sq += float(torch.sum(noise * noise).item())
    random_norm = math.sqrt(random_sq)
    scale = delta_norm / random_norm

    generator.manual_seed(args.seed)
    with torch.no_grad():
        for parameter in pre.parameters():
            noise = torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
            parameter.add_(noise, alpha=scale)
    args.output.mkdir(parents=True, exist_ok=True)
    pre.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.pre_checkpoint).save_pretrained(args.output)
    atomic_write_json(args.output / "phase1_random_parameter_manifest.json", {
        "control_type": "random_parameter",
        "pre_checkpoint": args.pre_checkpoint,
        "post_checkpoint": args.post_checkpoint,
        "seed": args.seed,
        "real_delta_l2_norm": delta_norm,
        "raw_random_l2_norm": random_norm,
        "random_scale": scale,
        "config_hash": stable_hash(vars(args)),
    })


if __name__ == "__main__":
    main()

