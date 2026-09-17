"""Resource adaptation without changing the global 32-decision optimizer batch."""
import math

import torch


def validate_elastic_dispatch(config):
    """Limit uneven-rank dispatch to the audited Phase2 HF/GRPO path.

    The environment loop pads/unpads *inference* requests independently of the
    32 real environment rollouts. adjust_batch pads collected decisions for
    equal-rank dispatch; odd_world_minibatches supplies optimizer sync slots.
    """
    actor=config.actor_rollout_ref.actor
    rollout=config.actor_rollout_ref.rollout
    checks={
        "Phase2 capture":config.get("phase2",{}).get("enabled",False),
        "HF rollout":rollout.name=="hf",
        "GRPO":config.algorithm.adv_estimator=="grpo",
        "environment rollout batch":(config.data.train_batch_size>0 and config.env.rollout.n>1)
            if config.get("phase2",{}).get("allow_expanded_rollout_batch",False)
            else config.data.train_batch_size*config.env.rollout.n==32,
        "32 optimizer decisions":actor.ppo_mini_batch_size*rollout.n==32,
        "actor microbatch one":actor.ppo_micro_batch_size_per_gpu==1,
        "fixed batch size":not actor.use_dynamic_bsz,
        "text only":config.actor_rollout_ref.model.get("load_text_only",False),
    }
    failed=[name for name,passed in checks.items() if not passed]
    if failed:raise ValueError(f"Unsupported elastic training configuration: {failed}")


def minibatch_indices(total, world, rank, global_minibatch=32):
    width=math.ceil(global_minibatch/world)
    batches=[]
    for begin in range(0,total,global_minibatch):
        indices=[];weights=[]
        for local in range(width):
            offset=rank+local*world
            real=offset<global_minibatch and begin+offset<total
            indices.append(begin+offset if real else 0)
            weights.append(float(real))
        batches.append((indices,weights))
    return batches


def odd_world_minibatches(batch):
    """Gather small decision tensors, then distribute exact 32-row minibatches.

    Extra synchronization slots have zero optimizer contribution, including KL.
    They are not added to rollout data or reward-directed signal support.
    """
    from tensordict import TensorDict
    world=torch.distributed.get_world_size()
    rank=torch.distributed.get_rank()
    combined={}
    for name,value in batch.items():
        gathered=[torch.empty_like(value) for _ in range(world)]
        torch.distributed.all_gather(gathered,value.contiguous())
        combined[name]=torch.cat(gathered)
    full=TensorDict(combined,batch_size=[len(batch)*world])
    result=[]
    for indices,weights in minibatch_indices(len(full),world,rank):
        index=torch.tensor(indices,device=batch.device,dtype=torch.long)
        part=full[index].clone()
        part["phase2_optimizer_weight"]=torch.tensor(weights,device=part["responses"].device)
        part["phase2_row_index"][part["phase2_optimizer_weight"]==0]=-1
        result.append(part)
    return result


def cpu_adam_step(optimizer):
    """Run the same AdamW optimizer on CPU copies, then copy parameters back.

    FSDP parameter/gradient storage stays on its original GPU. Only optimizer
    arithmetic and moments move to CPU; native state_dict keys remain unchanged.
    """
    original_groups=[list(group["params"]) for group in optimizer.param_groups]
    pairs=[]
    try:
        for group,originals in zip(optimizer.param_groups,original_groups):
            replacement=[]
            for original in originals:
                cpu=torch.nn.Parameter(original.detach().to("cpu",copy=True))
                if original.grad is not None:cpu.grad=original.grad.detach().to("cpu",copy=True)
                state=optimizer.state.pop(original,{})
                state={key:value.cpu() if torch.is_tensor(value) else value for key,value in state.items()}
                optimizer.state[cpu]=state
                pairs.append((original,cpu))
                replacement.append(cpu)
            group["params"]=replacement
        optimizer.step()
        with torch.no_grad():
            for original,cpu in pairs:original.copy_(cpu.to(original.device))
    finally:
        for group,originals in zip(optimizer.param_groups,original_groups):group["params"]=originals
        for original,cpu in pairs:optimizer.state[original]=optimizer.state.pop(cpu)
