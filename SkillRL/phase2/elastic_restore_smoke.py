"""Disposable real-GPU FSDP restore/Adam audit; never trains experiment weights."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision

from phase1.archive import atomic_write_json
from phase2.capture import save_tensor_file
from phase2.elastic_checkpoint import restore,tensor_hash


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--cpu-adam",action="store_true")
    args=p.parse_args()
    rank=int(os.environ["LOCAL_RANK"]);world=int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank);torch.distributed.init_process_group("nccl")
    torch.set_num_threads(1)
    args.root.mkdir(parents=True,exist_ok=True)
    os.environ["PHASE2_ROOT"]=str(args.root.resolve())
    module=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Linear(4,2)).cuda()
    model=FSDP(module,device_id=rank,device_mesh=init_device_mesh("cuda",(world,)),
               mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32))
    flat=model._handle.flat_param
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    directory=args.root/"input";directory.mkdir(exist_ok=True)
    width=flat.numel()
    parameter=(torch.arange(width,dtype=torch.float32)+rank*width)/100
    state={"step":torch.tensor(519.),"exp_avg":torch.ones(width)*.125,"exp_avg_sq":torch.ones(width)*.25}
    save_tensor_file(directory/f"elastic_world_size_{world}_rank_{rank}.pt",{
        "parameter":parameter,"optimizer":{"param_groups":optimizer.state_dict()["param_groups"],"state":{0:state}}})
    record={"hashes":{"parameter":tensor_hash(parameter),**{key:tensor_hash(value) for key,value in state.items()}}}
    records=[None]*world;torch.distributed.all_gather_object(records,record)
    if rank==0:
        atomic_write_json(directory/"elastic_resume.json",{
            "source_checkpoint":"/smoke-only/global_step_31","source_world_size":8,
            "target_world_size":world,"optimizer_step":519,"parameter_names":list(flat._fqns),
            "parameter_shapes":[list(s) for s in flat._shapes],"flat_numel":flat._unpadded_unsharded_size.numel(),"ranks":records})
        save_tensor_file(directory/"scheduler.pt",scheduler.state_dict())
    torch.distributed.barrier()
    restore(SimpleNamespace(model=model,optimizer=optimizer,lr_scheduler=scheduler,rank=rank,world_size=world),directory)
    # Exercise the parity-forward -> HF rollout -> backward lifecycle. An
    # inference_mode forward here poisons FSDP buffers and fails at summon.
    model.eval()
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        probe=model(torch.ones(1,3,device=rank))
    del probe
    if world>1:model._handle.reshard(True)
    with torch.no_grad(),FSDP.summon_full_params(model,writeback=False):
        assert all(not parameter.is_inference() for parameter in model.parameters())
    model.train()
    with torch.autocast("cuda",dtype=torch.bfloat16):loss=model(torch.ones(1,3,device=rank)).float().square().mean()
    loss.backward()
    if args.cpu_adam:
        from phase2.elastic_training import cpu_adam_step
        cpu_adam_step(optimizer)
        assert optimizer.param_groups[0]["params"][0] is flat
        assert all(value.device.type=="cpu" for value in optimizer.state[flat].values() if torch.is_tensor(value))
    else:optimizer.step()
    assert optimizer.state[flat]["step"].item()==520
    atomic_write_json(args.root/f"smoke-rank-{rank}.json",{"restore_passed":True,"parity_forward_then_rollout_context_passed":True,"backward_and_adam_passed":True,"optimizer_backend":"cpu_adamw" if args.cpu_adam else "gpu_adamw","Adam_after":520,"loss":float(loss.detach())})
    torch.distributed.barrier();torch.distributed.destroy_process_group()
    print(f"Disposable FSDP restore + backward + Adam passed rank {rank}/{world}",flush=True)


if __name__=="__main__":main()
