"""Check HF generation memory with root-FSDP GPU reference and allocated Adam."""
import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from transformers import AutoModelForCausalLM, GenerationConfig


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(4)
    dist.init_process_group('nccl')
    mp=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32)
    def load(path,dtype):
        model=AutoModelForCausalLM.from_pretrained(path,dtype=dtype,attn_implementation='sdpa',local_files_only=True).to(dtype=dtype)
        return FSDP(model,device_id=rank,mixed_precision=mp,use_orig_params=False,sync_module_states=True)
    ref=load('/home/wangyifan/model/Qwen3.5-4B',torch.bfloat16)
    actor=load(args.root/'models/u0005',torch.float32)
    optim=torch.optim.AdamW(actor.parameters(),lr=1e-6)
    for param in actor.parameters(): param.grad=torch.zeros_like(param)
    optim.step();optim.zero_grad(set_to_none=True)
    source=torch.load(args.root/'direction_batches/u0001.pt',map_location='cpu',weights_only=False)['tensors']
    ids=source['prompts'][rank*2:rank*2+2].cuda();mask=source['prompt_mask'][rank*2:rank*2+2].cuda()
    actor.eval();dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.no_grad(),FSDP.summon_full_params(actor,writeback=False,recurse=False),torch.autocast('cuda',dtype=torch.bfloat16):
        output=actor.generate(input_ids=ids,attention_mask=mask,generation_config=GenerationConfig(
            do_sample=True,temperature=1.,top_p=1.,top_k=0,max_new_tokens=64,
            eos_token_id=248044,pad_token_id=248044,use_cache=True))
    torch.cuda.synchronize()
    values={'seconds':time.perf_counter()-start,'peak_gib':torch.cuda.max_memory_allocated()/2**30,
            'response_shape':list(output[:,4096:].shape),'reference_parameters':sum(p.numel() for p in ref.parameters())}
    rows=[None]*dist.get_world_size();dist.all_gather_object(rows,values)
    if rank==0:
        args.output.write_text(json.dumps({'world_size':dist.get_world_size(),'ranks':rows},indent=2)+'\n')
        print(json.dumps(rows),flush=True)
    dist.destroy_process_group()


if __name__=='__main__': main()
