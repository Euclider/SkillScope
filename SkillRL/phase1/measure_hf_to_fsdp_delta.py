#!/usr/bin/env python3
"""Measure an exact parameter delta from a Hugging Face model to a verl FSDP checkpoint."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from phase1.archive import atomic_write_json, stable_hash

_SHARD_PATTERN = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")


def _checkpoint_shards(checkpoint_dir: Path) -> tuple[int, list[Path]]:
    shards: dict[int, Path] = {}
    world_sizes: set[int] = set()
    for path in checkpoint_dir.glob("model_world_size_*_rank_*.pt"):
        match = _SHARD_PATTERN.match(path.name)
        if match is None:
            continue
        world_size, rank = map(int, match.groups())
        world_sizes.add(world_size)
        shards[rank] = path
    if len(world_sizes) != 1:
        raise ValueError(f"Expected one FSDP world size in {checkpoint_dir}, got {world_sizes}")
    world_size = world_sizes.pop()
    if sorted(shards) != list(range(world_size)):
        raise ValueError(f"Incomplete FSDP shards in {checkpoint_dir}: {sorted(shards)}")
    return world_size, [shards[rank] for rank in range(world_size)]


def _local_tensor_and_dim(value: torch.Tensor) -> tuple[torch.Tensor, int]:
    if not hasattr(value, "_local_tensor") or not hasattr(value, "placements"):
        raise TypeError(f"Expected a DTensor, got {type(value).__name__}")
    placements = value.placements
    if len(placements) != 1 or not placements[0].is_shard():
        raise ValueError(f"Only one-dimensional FSDP shards are supported, got {placements}")
    return value._local_tensor, placements[0].dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", required=True, type=Path)
    parser.add_argument("--fsdp-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    world_size, shard_paths = _checkpoint_shards(args.fsdp_checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.hf_checkpoint,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    base_state = base_model.state_dict()

    delta_squared = 0.0
    reference_squared = 0.0
    parameter_count = 0
    changed_parameter_count = 0
    max_absolute_delta = 0.0

    for rank, shard_path in enumerate(shard_paths):
        shard_state = torch.load(shard_path, map_location="cpu", weights_only=False)
        if set(shard_state) != set(base_state):
            raise ValueError(f"Parameter keys differ between {args.hf_checkpoint} and {shard_path}")
        for name, sharded_value in shard_state.items():
            local_value, shard_dim = _local_tensor_and_dim(sharded_value)
            base_value = base_state[name]
            chunk_size = math.ceil(base_value.shape[shard_dim] / world_size)
            start = rank * chunk_size
            stop = min(start + chunk_size, base_value.shape[shard_dim])
            local_base = base_value.narrow(shard_dim, start, stop - start)
            if local_base.shape != local_value.shape:
                raise ValueError(
                    f"Local shape mismatch for {name} rank {rank}: "
                    f"base slice {tuple(local_base.shape)} versus checkpoint {tuple(local_value.shape)}"
                )
            delta = local_value.float() - local_base.float()
            delta_squared += float(torch.sum(delta * delta, dtype=torch.float64).item())
            reference_squared += float(
                torch.sum(local_base.float() * local_base.float(), dtype=torch.float64).item()
            )
            parameter_count += delta.numel()
            changed_parameter_count += int(torch.count_nonzero(delta).item())
            max_absolute_delta = max(max_absolute_delta, float(torch.max(torch.abs(delta)).item()))
        del shard_state

    delta_l2 = math.sqrt(delta_squared)
    reference_l2 = math.sqrt(reference_squared)
    atomic_write_json(
        args.output,
        {
            "schema_version": "phase1.hf_to_fsdp_checkpoint_delta.v1",
            "hf_checkpoint": str(args.hf_checkpoint.resolve()),
            "fsdp_checkpoint": str(args.fsdp_checkpoint.resolve()),
            "fsdp_world_size": world_size,
            "parameter_count": parameter_count,
            "changed_parameter_count": changed_parameter_count,
            "parameter_delta_l2_norm": delta_l2,
            "reference_parameter_l2_norm": reference_l2,
            "relative_l2_norm": delta_l2 / reference_l2,
            "max_absolute_delta": max_absolute_delta,
            "all_finite": all(
                math.isfinite(value)
                for value in (delta_l2, reference_l2, max_absolute_delta)
            ),
            "config_hash": stable_hash(vars(args)),
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
