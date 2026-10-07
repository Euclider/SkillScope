"""Sampling replica; current FSDP policy state is copied before every request."""
import copy
import torch


def synchronize_replica(policy,replica,state,*,factory=None,dtype=None):
    parameter=next(policy.parameters())
    target_dtype=dtype or parameter.dtype
    if replica is None:
        devices=[parameter.device.index] if parameter.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            if factory is None:
                from transformers import AutoModelForCausalLM
                replica=AutoModelForCausalLM.from_config(copy.deepcopy(policy.config),dtype=target_dtype)
            else:
                replica=factory(copy.deepcopy(policy.config))
            replica.to(device=parameter.device,dtype=target_dtype)
        replica.requires_grad_(False)
    replica.load_state_dict(state,strict=True)
    replica.eval()
    return replica
