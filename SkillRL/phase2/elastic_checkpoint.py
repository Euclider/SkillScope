"""Lossless, opt-in single-flat-parameter FSDP checkpoint redistribution.

The existing ALFWorld HF actor uses one FSDP FlatParameter. Preserve its exact
registration order, values, Adam moments/counter and scheduler across world sizes.
Native checkpoints are never modified. This is deliberately not a generic FSDP
converter: incompatible layouts abort instead of silently resetting optimizer state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from safetensors import safe_open

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
from phase2.capture import save_tensor_file


def tensor_hash(tensor):
    array=tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def redistribute_slice(shards, start, length, valid_total):
    """Copy a contiguous global interval; only terminal FSDP padding is zeroed."""
    output=torch.zeros(length,dtype=shards[0].dtype)
    offset=0
    for shard in shards:
        end=offset+shard.numel()
        left=max(start,offset)
        right=min(start+length,end,valid_total)
        if right>left:
            output[left-start:right-start].copy_(shard[left-offset:right-offset])
        offset=end
    if min(start+length,valid_total)>offset:
        raise ValueError("Missing source optimizer interval")
    return output


def prepare(checkpoint, model, target, world_size):
    checkpoint,model,target=(Path(p).resolve() for p in (checkpoint,model,target))
    if not 1<=world_size<=8:raise ValueError("Expected 1..8 target ranks")
    source=validate_full_checkpoint(checkpoint)
    completed=target/"actor/elastic_resume.json"
    if completed.exists():
        manifest=json.loads(completed.read_text())
        if manifest["source_checkpoint"]!=str(checkpoint) or manifest["target_world_size"]!=world_size:
            raise ValueError("Existing recovery conversion does not match request")
        return manifest
    target.mkdir(parents=True,exist_ok=True)
    actor=target/"actor";actor.mkdir(exist_ok=True)
    old_world=source["world_size"]
    metadata=torch.load(checkpoint/"actor"/f"model_world_size_{old_world}_rank_0.pt",
                        map_location="cpu",weights_only=False,mmap=True)
    configuration=json.loads((model/"config.json").read_text())
    if configuration["model_type"]!="qwen3_5_text" or not configuration.get("tie_word_embeddings"):
        raise ValueError("This converter requires the verified tied Qwen3.5 text actor")
    names=[key for key in metadata if key!="lm_head.weight"]
    shapes=[list(metadata[key].shape) for key in names]
    numels=[math.prod(shape) for shape in shapes]
    total=sum(numels)
    del metadata
    optimizer=[torch.load(checkpoint/"actor"/f"optim_world_size_{old_world}_rank_{rank}.pt",
                          map_location="cpu",weights_only=False,mmap=True) for rank in range(old_world)]
    for state in optimizer:
        if list(state["state"])!=[0] or len(state["param_groups"])!=1 or state["param_groups"][0]["params"]!=[0]:
            raise ValueError("Only one verified optimizer FlatParameter is supported")
        if state["param_groups"]!=optimizer[0]["param_groups"]:
            raise ValueError("Optimizer groups differ between source ranks")
        if set(state["state"][0])!={"step","exp_avg","exp_avg_sq"}:
            raise ValueError("Unsupported Adam state fields")
        if state["state"][0]["step"].item()!=optimizer[0]["state"][0]["step"].item():
            raise ValueError("Source Adam counters disagree")
        for key in ("exp_avg","exp_avg_sq"):
            value=state["state"][0][key]
            if value.shape!=(math.ceil(total/old_world),) or value.dtype!=torch.float32:
                raise ValueError("Adam layout does not match ordered model parameters")
    inventory={}
    for file in model.glob("*.safetensors"):
        with safe_open(file,framework="pt") as handle:
            inventory.update({key:file for key in handle.keys()})
    flat=torch.empty(total,dtype=torch.float32)
    offset=0
    for name,shape,size in zip(names,shapes,numels):
        with safe_open(inventory[name],framework="pt") as handle:
            value=handle.get_tensor(name)
            if value.dtype!=torch.float32 or list(value.shape)!=shape:
                raise ValueError(f"Lossy or incompatible source model: {name}")
            flat[offset:offset+size].copy_(value.flatten())
        offset+=size
    local_size=math.ceil(total/world_size)
    records=[]
    for rank in range(world_size):
        start=rank*local_size
        parameter=redistribute_slice([flat],start,local_size,total)
        local_state={"step":optimizer[0]["state"][0]["step"].clone()}
        for key in ("exp_avg","exp_avg_sq"):
            local_state[key]=redistribute_slice([x["state"][0][key] for x in optimizer],start,local_size,total)
        hashes={"parameter":tensor_hash(parameter),**{key:tensor_hash(value) for key,value in local_state.items()}}
        file=actor/f"elastic_world_size_{world_size}_rank_{rank}.pt"
        save_tensor_file(file,{"parameter":parameter,"optimizer":{"state":{0:local_state},"param_groups":optimizer[0]["param_groups"]}})
        reread=torch.load(file,map_location="cpu",weights_only=False,mmap=True)
        actual={"parameter":tensor_hash(reread["parameter"]),**{key:tensor_hash(value) for key,value in reread["optimizer"]["state"][0].items()}}
        if actual!=hashes:raise ValueError("Serialized conversion checksum mismatch")
        records.append({"rank":rank,"global_start":start,"local_numel":local_size,"hashes":hashes,"bytes":file.stat().st_size})
        print(f"Converted and verified rank {rank+1}/{world_size}",flush=True)
        del parameter,local_state,reread
    shutil.copy2(checkpoint/"data.pt",target/"data.pt")
    extra=torch.load(checkpoint/"actor"/f"extra_state_world_size_{old_world}_rank_0.pt",map_location="cpu",weights_only=False)
    save_tensor_file(actor/"scheduler.pt",extra["lr_scheduler"])
    manifest={"schema_version":"phase2.elastic_single_flat.v1","created_at":utc_now(),
              "source_checkpoint":str(checkpoint),"source_world_size":old_world,"source_model":str(model),
              "target_world_size":world_size,"flat_numel":total,"parameter_names":names,"parameter_shapes":shapes,
              "optimizer_step":int(optimizer[0]["state"][0]["step"].item()),"ranks":records,
              "source_export_sha256":sha256_file(model/"phase2_export.json"),
              "rng_policy":"world-size changes reseed each target rank deterministically; not bitwise RNG continuation"}
    atomic_write_json(completed,manifest)
    return manifest


def restore(manager, local_path):
    directory=Path(local_path)
    manifest=json.loads((directory/"elastic_resume.json").read_text())
    rank,world=manager.rank,manager.world_size
    if world!=manifest["target_world_size"]:raise ValueError("Elastic target world size mismatch")
    handles=[module._handle for module in manager.model.modules() if hasattr(module,"_handle") and module._handle is not None]
    if len({id(handle) for handle in handles})!=1:
        raise ValueError("Elastic restore only supports one root FSDP handle")
    flat=manager.model._handle.flat_param
    if list(flat._fqns)!=manifest["parameter_names"] or [list(shape) for shape in flat._shapes]!=manifest["parameter_shapes"]:
        raise ValueError("FSDP parameter registration order changed; refusing optimizer restore")
    if math.prod(flat._unpadded_unsharded_size)!=manifest["flat_numel"]:
        raise ValueError("FlatParameter size changed")
    payload=torch.load(directory/f"elastic_world_size_{world}_rank_{rank}.pt",map_location="cpu",weights_only=False,mmap=True)
    parameter=payload["parameter"]
    if flat.dtype!=torch.float32 or flat.shape!=parameter.shape:
        raise ValueError("Live FP32 FlatParameter shape does not match redistribution")
    with torch.no_grad():flat.copy_(parameter.to(flat.device))
    manager.optimizer.load_state_dict(payload["optimizer"])
    hashes={"parameter":tensor_hash(flat),**{key:tensor_hash(value) for key,value in manager.optimizer.state[flat].items()}}
    if hashes!=manifest["ranks"][rank]["hashes"]:
        raise ValueError("Restored parameter or Adam moment checksum differs")
    manager.lr_scheduler.load_state_dict(torch.load(directory/"scheduler.pt",map_location="cpu",weights_only=False))
    parent_update=int(Path(manifest["source_checkpoint"]).name.removeprefix("global_step_"))
    parity=None
    root=Path(os.environ["PHASE2_ROOT"])
    batchfile=root/"batches"/f"u{parent_update:04d}"/"training_batch.pt"
    if batchfile.exists():
        batch=torch.load(batchfile,map_location="cpu",weights_only=False)["tensors"]
        witness=torch.load(root/"new_logprobs"/f"u{parent_update:04d}"/"row-000000.pt",map_location="cpu",weights_only=False)
        manager.model.eval()
        # FSDP reuses unsharded buffers in the subsequent training/rollout path.
        # InferenceMode would create inference tensors here, making later
        # summon_full_params() fail when it constructs trainable Parameter views.
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            result=manager.model(input_ids=batch["input_ids"][:1].to(flat.device),
                                 attention_mask=batch["attention_mask"][:1].to(flat.device),
                                 position_ids=batch["position_ids"][:1].to(flat.device),use_cache=False)
        length=batch["responses"].shape[-1]
        # The parent actor uses temperature=1.0; read it explicitly for validation.
        meta=json.loads((batchfile.parent/"alignment_audit.json").read_text())
        logits=result.logits[0,-length-1:-1]/meta["temperature"]
        replay=logits[witness["token_positions"].to(flat.device)].float().log_softmax(-1).cpu()
        parity=float((replay-witness["log_probs"]).abs().max())
        if parity>1e-5:raise ValueError(f"Restored policy full-vocabulary parity failed: {parity}")
        del batch,result,logits,replay,witness
        if world>1:manager.model._handle.reshard(True)
    seed=20260910+1009*parent_update+rank
    torch.manual_seed(seed);torch.cuda.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    output=root/"elastic_restore_audits"/f"u{parent_update+1:04d}"/f"rank-{rank}.json"
    atomic_write_json(output,{"created_at":utc_now(),"source_checkpoint":manifest["source_checkpoint"],
                             "source_world_size":manifest["source_world_size"],"world_size":world,"rank":rank,
                             "optimizer_step":manifest["optimizer_step"],"checksums_match":True,
                             "full_vocab_parity_max_abs":parity,"new_rank_rng_seed":seed,"hashes":hashes})
    torch.distributed.barrier()
    print(f"[rank-{rank}] elastic restore verified: Adam {manifest['optimizer_step']}, parity {parity}",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--target",type=Path,required=True)
    parser.add_argument("--world-size",type=int,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    prepare(args.checkpoint,args.model,args.target,args.world_size)


if __name__=="__main__":main()
