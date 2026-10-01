"""Four-rank offline probe with real Qwen weights; never writes policy checkpoints."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision
from transformers import AutoModelForCausalLM
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.utils.fsdp_utils import get_fsdp_wrap_policy


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--actor-model',required=True)
    p.add_argument('--reference-model',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rows-per-rank',type=int,default=8)
    p.add_argument('--skip-actor-baseline',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(4)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    source=torch.load(args.batch,map_location='cpu',weights_only=False)['tensors']
    n=args.rows_per_rank
    indices=torch.arange(rank*n,(rank+1)*n)*7 % len(source['responses'])
    mask=torch.cat([source['prompt_mask'],source['actual_loss_mask']],-1)[indices].long().cuda()
    data={'input_ids':torch.cat([source['prompts'],source['responses']],-1)[indices].cuda(),
          'responses':source['responses'][indices].cuda(),'attention_mask':mask,
          'position_ids':(mask.cumsum(-1)-1)*mask,'old_log_probs':source['old_log_probs'][indices].cuda(),
          'advantages':source['advantages'][indices].cuda()}
    report={'world_size':dist.get_world_size(),'rows_per_rank':n,'results':{}}
    def record(name,values):
        gathered=[None]*dist.get_world_size();dist.all_gather_object(gathered,values)
        report['results'][name]=gathered
        if rank==0:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(name,json.dumps(gathered),flush=True)
    base_cfg=dict(use_torch_compile=False,ulysses_sequence_parallel_size=1,
        ppo_mini_batch_size=n,ppo_micro_batch_size_per_gpu=1,ppo_epochs=1,use_dynamic_bsz=False,
        clip_ratio=.2,clip_ratio_low=.2,clip_ratio_high=.2,entropy_coeff=.001,loss_agg_mode='token-mean',
        use_kl_loss=True,kl_loss_type='low_var_kl',kl_loss_coef=.01,policy_loss={'loss_mode':'vanilla'},grad_clip=1.)
    mp=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32)
    def build(path,dtype,offload,reference,root_wrap=False):
        model=AutoModelForCausalLM.from_pretrained(path,dtype=dtype,attn_implementation='sdpa',local_files_only=True)
        model.to(dtype=dtype)
        if not reference: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        policy=get_fsdp_wrap_policy(model,config={'min_num_params':0}) if reference and not root_wrap else None
        return FSDP(model,device_id=rank,auto_wrap_policy=policy,
            cpu_offload=CPUOffload(offload_params=True) if offload else None,
            mixed_precision=mp,use_orig_params=False,sync_module_states=True)
    reference_scores=None
    for label,offload,suffix,root_wrap in [('reference_cpu',True,False,False),
        ('reference_gpu',False,True,False),('reference_gpu_root',False,True,True)]:
        model=build(args.reference_model,torch.bfloat16,offload,True,root_wrap)
        worker=DataParallelPPOActor(OmegaConf.create({**base_cfg,'response_logits_only':suffix}),model)
        proto=DataProto.from_dict(tensors={k:v[:8] for k,v in data.items()},meta_info={'micro_batch_size':1,'temperature':1.,'use_dynamic_bsz':False})
        worker.compute_log_prob(proto)
        dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        scores,_=worker.compute_log_prob(proto)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        valid=mask[:8,-data['responses'].shape[-1]:].bool()
        error=0. if reference_scores is None else (scores-reference_scores)[valid].abs().max().item()
        reference_scores=scores.detach().clone()
        record(label,{'seconds':elapsed,'max_logprob_error':error,'peak_gib':torch.cuda.max_memory_allocated()/2**30})
        if not root_wrap:
            del worker,model;gc.collect();torch.cuda.empty_cache()
    if n>8:
        reference_scores,_=worker.compute_log_prob(DataProto.from_dict(tensors=data,meta_info=proto.meta_info))
    model._handle.reshard(True)  # Match the production reference-worker boundary.
    # Keep the GPU reference alive throughout actor tests, as in the proposed runtime.
    data['ref_log_prob']=reference_scores
    actor_model=build(args.actor_model,torch.float32,False,False)
    optim=torch.optim.AdamW(actor_model.parameters(),lr=1e-6,weight_decay=.01)
    # Allocate real Adam state before timing; no checkpoint is modified.
    for parameter in actor_model.parameters(): parameter.grad=torch.zeros_like(parameter)
    optim.step();optim.zero_grad(set_to_none=True)
    actor=DataParallelPPOActor(OmegaConf.create(base_cfg),actor_model,optim)
    proto=DataProto.from_dict(tensors=data,meta_info={'temperature':1.})
    captured=[]
    def capture_step():
        captured.append([parameter.grad.detach().cpu().clone() for parameter in actor_model.parameters()])
        norm=actor_model.clip_grad_norm_(1.)
        return norm  # Deliberately no parameter update; compare exactly the same model.
    actor._optimizer_step=capture_step
    # Warm up backward kernels before comparing wall time.
    actor.config.ppo_mini_batch_size=1
    actor.update_policy(proto[:1]);captured.clear()
    actor.config.ppo_mini_batch_size=n
    initial=None
    modes=[('actor_baseline',False,False),('actor_suffix',True,False),('actor_suffix_no_sync',True,True)]
    if args.skip_actor_baseline: modes=modes[1:]
    for label,suffix,nosync in modes:
        actor.response_logits_only=suffix;actor.config.accumulate_no_sync=nosync
        dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        metrics=actor.update_policy(proto)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        gradients=captured.pop()
        row={'seconds':elapsed,'peak_gib':torch.cuda.max_memory_allocated()/2**30,
             'optimizer_boundaries':len(metrics['actor/grad_norm']),
             'unclipped_grad_norm':metrics['actor/grad_norm'][0]}
        if initial is None: initial=gradients
        else:
            s0=s1=dot=diff=0.
            for a,b in zip(initial,gradients):
                # Chunk the root flat gradient to bound double-precision workspace.
                for aa,bb in zip(a.flatten().split(1000000),b.flatten().split(1000000)):
                    aa,bb=aa.double(),bb.double()
                    s0+=aa.square().sum().item();s1+=bb.square().sum().item()
                    dot+=(aa*bb).sum().item();diff+=(aa-bb).square().sum().item()
            row.update(grad_relative_l2=(diff/max(s0,1e-30))**.5,grad_cosine=dot/max((s0*s1)**.5,1e-30))
        record(label,row)
    dist.destroy_process_group()


if __name__=='__main__': main()
