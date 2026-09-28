"""CPU-only, bitwise FP32 parity gate between native DTensor shards and HF export."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path


def verify(checkpoint, target):
    import torch
    from torch.distributed.tensor import DTensor
    from safetensors import safe_open
    from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
    checkpoint, target = Path(checkpoint), Path(target)
    world = validate_full_checkpoint(checkpoint)['world_size']
    shards = [torch.load(checkpoint/'actor'/f'model_world_size_{world}_rank_{rank}.pt',
                         map_location='cpu', weights_only=False) for rank in range(world)]
    keys = set(shards[0])
    if any(set(state) != keys for state in shards):
        raise ValueError('Native model ranks disagree on parameter keys')
    parameters = 0
    with ExitStack() as stack:
        handles = [stack.enter_context(safe_open(path, framework='pt', device='cpu'))
                   for path in sorted(target.glob('*.safetensors'))]
        locations = {}
        for handle in handles:
            for name in handle.keys():
                if name in locations:
                    raise ValueError('Duplicate exported parameter: '+name)
                locations[name] = handle
        if set(locations) != keys:
            raise ValueError('Exported/native parameter names do not match exactly')
        for name in sorted(keys):
            values = [state.pop(name) for state in shards]
            if not all(isinstance(value, DTensor) and value.dtype == torch.float32 for value in values):
                raise ValueError('This parity gate requires the registered native FP32 DTensor checkpoint')
            placement = values[0].placements
            if len(placement) != 1 or any(value.placements != placement for value in values):
                raise ValueError('Unexpected or inconsistent native placement')
            local = [value.to_local() for value in values]
            if placement[0].is_shard():
                expected = torch.cat(local, dim=placement[0].dim)
            elif placement[0].is_replicate():
                expected = local[0]
                if any(not torch.equal(expected.contiguous().view(torch.int32), value.contiguous().view(torch.int32))
                       for value in local[1:]):
                    raise ValueError('Replicated native parameters disagree')
            else:
                raise ValueError('Unsupported partial placement')
            actual = locations[name].get_tensor(name)
            if (actual.dtype != torch.float32 or tuple(actual.shape) != tuple(values[0].shape)
                    or not torch.isfinite(expected).all() or not torch.isfinite(actual).all()
                    or not torch.equal(actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32))):
                raise ValueError('FP32 export differs from native checkpoint: '+name)
            parameters += actual.numel()
    return {'status': 'PASS', 'native_ranks': world, 'tensors': len(keys), 'parameters': parameters,
            'dtype': 'float32', 'all_parameters_bitwise_equal': True, 'all_parameters_finite': True,
            'environment_rollouts': 0, 'optimizer_updates': 0}
