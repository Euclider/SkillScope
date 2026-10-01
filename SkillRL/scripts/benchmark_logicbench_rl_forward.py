"""Offline real-model parity/latency probe. Never updates a training checkpoint."""
import argparse
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM
from verl.workers.actor.dp_actor import DataParallelPPOActor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rows', type=int, default=8)
    p.add_argument('--gradients', action='store_true')
    p.add_argument('--suffix-only', action='store_true')
    p.add_argument('--force-fp32', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(707)
    source = torch.load(args.batch, map_location='cpu', weights_only=False)['tensors']
    indices = torch.linspace(0, len(source['responses']) - 1, args.rows).long()
    mask = torch.cat([source['prompt_mask'], source['actual_loss_mask']], -1)[indices].long().cuda()
    data = {'input_ids':torch.cat([source['prompts'], source['responses']], -1)[indices].cuda(),
            'responses':source['responses'][indices].cuda(), 'attention_mask':mask,
            'position_ids':((mask.cumsum(-1) - 1) * mask)}
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
              attn_implementation='sdpa', local_files_only=True).cuda().eval()
    if args.force_fp32:
        model.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    config = OmegaConf.create({'use_torch_compile':False,'ulysses_sequence_parallel_size':1})
    actor = DataParallelPPOActor(config, model)
    results = {'model':args.model, 'row_indices':indices.tolist(), 'torch':torch.__version__,
               'parameter_dtypes':sorted({str(p.dtype) for p in model.parameters()}),
               'valid_lengths':mask.sum(-1).tolist(), 'variants':{}}
    print('dtypes',results['parameter_dtypes'],flush=True)
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2)+'\n')
    modes = [('baseline',False,False,1),('suffix',True,False,1),
             ('trim_mb1',True,True,1),('suffix_mb4',True,False,4),('trim_mb4',True,True,4),
             ('suffix_mb8',True,False,8),('trim_mb8',True,True,8)]
    if args.suffix_only:
        modes = modes[:2]
    baseline = None
    valid = mask[:, -data['responses'].shape[-1]:].bool()
    for name, suffix, trim, size in modes:
        actor.response_logits_only, actor.trim_common_padding = suffix, trim
        # One warm-up per variant; timing includes the same chosen-token and entropy work.
        with torch.no_grad():
            actor._forward_micro_batch({k:v[:size] for k,v in data.items()}, 1., True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter(); entropies, scores = [], []
        with torch.no_grad():
            for offset in range(0,args.rows,size):
                entropy, lp = actor._forward_micro_batch({k:v[offset:offset+size] for k,v in data.items()},1.,True)
                entropies.append(entropy); scores.append(lp)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        lp, ent = torch.cat(scores), torch.cat(entropies)
        if baseline is None: baseline = (lp.clone(),ent.clone())
        delta = (lp-baseline[0])[valid].abs()
        row = {'seconds':elapsed,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
               'max_logprob_error':delta.max().item(),'mean_logprob_error':delta.mean().item(),
               'max_entropy_error':(ent-baseline[1])[valid].abs().max().item()}
        results['variants'][name] = row
        print(name,json.dumps(row),flush=True);save()
        del lp,ent,entropies,scores
    if args.gradients:
        model.train()
        sample={k:v[:1] for k,v in data.items()}
        weights=sample['attention_mask'][:,-sample['responses'].shape[-1]:]
        grads = None
        for name,suffix in [('baseline',False),('suffix',True)]:
            actor.response_logits_only,actor.trim_common_padding=suffix,False
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            ent,lp=actor._forward_micro_batch(sample,1.,True)
            loss=((lp+.001*ent)*weights).sum()/weights.sum()
            loss.backward();torch.cuda.synchronize()
            row={'seconds':time.perf_counter()-start,'loss':loss.item(),
                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
            if grads is None:
                grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
            else:
                ss0=ss1=dot=diff=0.;maxabs=0.
                for n,p in model.named_parameters():
                    if p.grad is None: continue
                    a=grads.pop(n).to(p.device);b=p.grad
                    ss0+=a.double().square().sum().item();ss1+=b.double().square().sum().item()
                    dot+=(a.double()*b.double()).sum().item();diff+=(a.double()-b.double()).square().sum().item()
                    maxabs=max(maxabs,(a-b).abs().max().item())
                row.update(grad_relative_l2=(diff/max(ss0,1e-30))**.5,
                           grad_cosine=dot/max((ss0*ss1)**.5,1e-30),grad_max_abs=maxabs)
            results.setdefault('gradients',{})[name]=row
            print('grad',name,json.dumps(row),flush=True);save()


if __name__ == '__main__':
    main()
