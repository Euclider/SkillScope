"""Opt-in CPU flattening guard; numerical values and FSDP topology are unchanged."""
import hashlib


def verify_identical_cpu_model(model):
    import torch
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if tensor.device.type != 'cpu' or tensor.is_meta:
            raise ValueError('CPU shard initialization requires fully materialized CPU weights/buffers')
        digest.update(name.encode())
        # View bytes rather than converting BF16/F32 values; bound extra RAM.
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    values = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(values, digest.hexdigest())
    if len(set(values)) != 1:
        raise ValueError('Rank initial parameters differ; refusing unsynchronized CPU shard initialization')
