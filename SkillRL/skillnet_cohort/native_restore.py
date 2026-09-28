"""Same-world native FSDP restore for root-flat and layer-wrapped models.

PyTorch's default single-flat SHARDED_STATE_DICT loader simultaneously gathers
all individual parameters AND a second full flat parameter on GPU. This opt-in
adapter retains the native files/optimizer/RNG but bounds the gather peak. It
does not reshard to a different world or reseed RNG like elastic recovery.
Layer-wrapped models use PyTorch's native strict sharded loader: its gather
scope is one FSDP unit, not the entire model as in the old single-root layout.
Their GPU memory bound must still be checked by the real restore preflight.
"""
from __future__ import annotations

import math


def restore_parameters(model, state, rank, world):
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
    from torch.distributed.tensor import DTensor, Shard
    handles = [module._handle for module in model.modules() if getattr(module, '_handle', None) is not None]
    handles = list({id(handle): handle for handle in handles}.values())
    if not handles:
        raise ValueError('Native restore requires FSDP parameter handles')
    if any(handle.rank != rank or handle.world_size != world for handle in handles):
        raise ValueError('Native live FSDP topology differs')
    if any(handle.flat_param.dtype != torch.float32 for handle in handles):
        raise ValueError('Native restore requires FP32 training parameters')
    if len(handles) != 1 or getattr(model, '_handle', None) is not handles[0]:
        if not isinstance(model, FSDP):
            raise ValueError('Layer-wrapped native restore requires an FSDP root')
        for param in state.values():
            if param.dtype != torch.float32:
                raise ValueError('Native parameter dtype mismatch')
            if isinstance(param, DTensor):
                if (param.device_mesh.size() != world or len(param.placements) != 1
                        or param.placements[0] != Shard(0)):
                    raise ValueError('Native DTensor topology differs')
            elif not hasattr(param, 'local_shards'):
                raise ValueError('Expected native distributed checkpoint tensors')
        print(f'[rank-{rank}]: Native strict layer-wrapped restore ({len(handles)} FSDP units)', flush=True)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                  ShardedStateDictConfig(offload_to_cpu=True)):
            model.load_state_dict(state, strict=True)
        return
    handle = handles[0]
    flat = handle.flat_param
    names, shapes = list(flat._fqns), list(flat._shapes)
    total = sum(math.prod(shape) for shape in shapes)
    width = math.ceil(total/world)
    if flat.dtype != torch.float32 or flat.numel() != width:
        raise ValueError('Native FP32 flat layout or world size changed')
    if set(state) - set(names) not in (set(), {'lm_head.weight'}):
        raise ValueError('Unexpected native state parameter inventory')
    if not all(name in state for name in names):
        raise ValueError('Native checkpoint misses live parameters')
    if 'lm_head.weight' not in names and 'lm_head.weight' in state:
        if not getattr(model._fsdp_wrapped_module.config, 'tie_word_embeddings', False):
            raise ValueError('Cannot discard an untied language head')
    local = torch.zeros(width, dtype=torch.float32)
    offset, start = 0, rank * width
    for name, shape in zip(names, shapes):
        param = state[name]
        if tuple(param.shape) != tuple(shape) or param.dtype != torch.float32:
            raise ValueError('Native parameter shape/dtype mismatch')
        count = math.prod(shape)
        if isinstance(param, DTensor):
            if param.device_mesh.size() != world or len(param.placements) != 1 or param.placements[0] != Shard(0):
                raise ValueError('Native DTensor topology differs')
            full = param.to(flat.device).full_tensor().reshape(-1)
        elif hasattr(param, 'local_shards'):
            shards = param.local_shards()
            if len(shards) > 1:
                raise ValueError('Unsupported native ShardedTensor layout')
            chunk = math.ceil(shape[0]/world) * (count//shape[0])
            source = torch.zeros(chunk, dtype=torch.float32, device=flat.device)
            if shards:
                value = shards[0].tensor.reshape(-1).to(flat.device)
                source[:value.numel()].copy_(value)
                del value
            result = torch.empty(chunk*world, dtype=torch.float32, device=flat.device)
            torch.distributed.all_gather_into_tensor(result, source)
            full = result[:count]
            del source
        else:
            raise ValueError('Expected native distributed checkpoint tensors')
        left, right = max(start, offset), min(start+width, offset+count)
        if right > left:
            local[left-start:right-start].copy_(full[left-offset:right-offset].cpu())
        offset += count
        del full
        if 'result' in locals():
            del result
    with torch.no_grad():
        flat.copy_(local.to(flat.device))
    if not torch.equal(flat.detach().cpu(), local):
        raise ValueError('Native local parameter copy failed byte-exact verification')
