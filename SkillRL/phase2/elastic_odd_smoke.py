"""Three-rank CPU/Gloo check of real TensorDict redistribution and gradients."""
import argparse
import os
from pathlib import Path

import torch
from tensordict import TensorDict

from phase1.archive import atomic_write_json
from phase2.elastic_training import odd_world_minibatches


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);args=parser.parse_args()
    torch.distributed.init_process_group("gloo")
    world=torch.distributed.get_world_size();rank=torch.distributed.get_rank()
    local=13
    rows=torch.arange(rank*local,(rank+1)*local)
    batch=TensorDict({"features":rows.float()/100,"responses":rows[:,None],"phase2_row_index":rows},batch_size=[local])
    parts=odd_world_minibatches(batch)
    maxima=[]
    for index,part in enumerate(parts):
        parameter=torch.tensor(1.,requires_grad=True)
        for feature,weight in zip(part["features"],part["phase2_optimizer_weight"]):
            (parameter*feature*weight*world/32).backward()
        gradient=parameter.grad
        torch.distributed.all_reduce(gradient);gradient/=world
        expected=torch.arange(index*32,min((index+1)*32,local*world)).float().sum()/100/32
        torch.testing.assert_close(gradient,expected,atol=1e-7,rtol=1e-6)
        maxima.append(float((gradient-expected).abs()))
    atomic_write_json(args.root/f"rank-{rank}.json",{"world_size":world,"all_gather_and_gradient_normalization_passed":True,
                                                    "max_abs_errors":maxima,"backend":"gloo_cpu"})
    torch.distributed.destroy_process_group()
    print(f"Odd-world exact-global-batch smoke passed rank {rank}/{world}",flush=True)


if __name__=="__main__":main()
