"""Exact full-vocabulary endpoint measurements on the actual update batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase2.capture import save_tensor_file
from phase2.direction import token_signals
from phase2.protocol import controls_by_skill, signal_directory, validate_windows


def phase(step):
    return "initial" if step == 0 else "early" if step < 5 else "middle" if step < 15 else "late"


def load_model(path):
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                               attn_implementation="sdpa").to("cuda:0").eval()


@torch.inference_mode()
def forward(model, ids, mask, positions, response_length, valid_positions, temperature):
    pos = positions.unsqueeze(0).cuda()
    if pos.ndim == 3:
        pos = pos.transpose(0, 1)
    out = model(input_ids=ids.unsqueeze(0).cuda(), attention_mask=mask.unsqueeze(0).cuda(),
                position_ids=pos, use_cache=False, output_hidden_states=True,
                logits_to_keep=response_length+1, return_dict=True)
    logits = out.logits[0, :-1].float()/temperature
    lp = logits.index_select(0, valid_positions.cuda()).log_softmax(-1)
    prompt_width = ids.numel()-response_length
    pred_pos = (prompt_width-1+valid_positions).cuda()
    hidden = {layer:out.hidden_states[layer][0].index_select(0,pred_pos).float().cpu()
              for layer in (8,16,24,32)}
    return lp, hidden


def counter_input(tokenizer, metadata, tensors, row, arm, placebo):
    info=metadata["info"]
    original=info["prompt_text"]
    payload=info["phase2_payload_text"]
    if not payload or original.count(payload) != 1:
        raise ValueError(f"Ambiguous original payload at {metadata['decision_id']}")
    response_length=tensors["responses"].shape[-1]
    width=tensors["input_ids"].shape[-1]-response_length
    original_rendered=tokenizer.apply_chat_template([{"role":"user","content":original}],
                                                   tokenize=False,add_generation_prompt=True,enable_thinking=False)
    original_ids=tokenizer.encode(original_rendered,add_special_tokens=False)
    original_mask=tensors["attention_mask"][row,:width].bool()
    if original_ids != tensors["input_ids"][row,:width][original_mask].tolist():
        raise ValueError(f"Live prompt/tokenizer mismatch at {metadata['decision_id']}")
    replacement=placebo if arm=="placebo" else ""
    changed=original.replace(payload,replacement,1)
    rendered=tokenizer.apply_chat_template([{"role":"user","content":changed}],
                                         tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prompt=tokenizer.encode(rendered,add_special_tokens=False)
    if len(prompt)>width:
        raise ValueError("Counterfactual prompt exceeds actual training prompt width")
    ids=tensors["input_ids"][row].clone()
    mask=tensors["attention_mask"][row].clone()
    ids[:width]=tokenizer.pad_token_id
    ids[width-len(prompt):width]=torch.tensor(prompt,dtype=ids.dtype)
    mask[:width]=0
    mask[width-len(prompt):width]=1
    positions=(mask.cumsum(-1)-1).clamp_min(0)
    return ids,mask,positions


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--update",type=int,required=True)
    p.add_argument("--start-update",type=int,help="Explicit wider window; direction is the first update batch's old-policy advantage")
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=8)
    p.add_argument("--max-decisions",type=int)
    a=p.parse_args()
    torch.set_num_threads(1)
    config=json.loads((a.root/"protocol.json").read_text())
    start=a.update-1 if a.start_update is None else a.start_update
    batch_update=start+1
    if a.start_update is not None:
        validate_windows(config["windows"])
        if not any(w["start"]==start and w["end"]==a.update for w in config["windows"]):
            raise ValueError("Unregistered endpoint window")
        if config.get("window_direction")!="start_batch_endpoint_projection":
            raise ValueError("Window reward direction was not explicitly registered")
        if list((a.root/"evaluations"/f"u{a.update:04d}").glob("shard-*.jsonl")):
            raise ValueError("Target gold already open; cannot create a prospective window measurement")
    output=signal_directory(a.root,a.update,a.start_update)
    output.mkdir(parents=True,exist_ok=True)
    commit=output/f"shard-{a.shard}.json"
    if commit.exists():
        print(f"Already committed: {commit}",flush=True)
        return
    batch=torch.load(a.root/"batches"/f"u{batch_update:04d}"/"training_batch.pt",map_location="cpu",weights_only=False)
    b=batch["tensors"]
    old_path=a.root/"models"/f"u{start:04d}"
    new_path=a.root/"models"/f"u{a.update:04d}"
    tokenizer=AutoTokenizer.from_pretrained(old_path)
    old_model,new_model=load_model(old_path),load_model(new_path)
    repo=Path(__file__).resolve().parents[1]
    controls=controls_by_skill(config,repo)
    temperature=float(batch["meta_info"].get("temperature",1.0))
    rows=[]
    seen=set()
    for row,meta in enumerate(batch["metadata"]):
        if meta["decision_id"] in seen:
            continue
        seen.add(meta["decision_id"])
        if meta["info"].get("selected_skill_id") in controls:
            rows.append((row,meta))
    rows=rows[a.shard::a.shards]
    if a.max_decisions:
        rows=rows[:a.max_decisions]
    token_rows,decision_rows,noise=[],[],[]
    for i,(row,meta) in enumerate(rows):
        skill=meta["info"]["selected_skill_id"]
        old=torch.load(a.root/"old_logprobs"/f"u{batch_update:04d}"/f"row-{row:06d}.pt",map_location="cpu",weights_only=False)
        new=torch.load(a.root/"new_logprobs"/f"u{a.update:04d}"/f"row-{row:06d}.pt",map_location="cpu",weights_only=False) if a.start_update is None else None
        if new is not None and (not torch.equal(old["token_ids"],new["token_ids"]) or not torch.equal(old["token_positions"],new["token_positions"])):
            raise ValueError("Live old/new token alignment failed")
        use=b["phase2_actual_loss_mask"][row,old["token_positions"]].bool()
        positions=old["token_positions"][use]
        ids=old["token_ids"][use]
        if not torch.equal(ids,b["responses"][row,positions]):
            raise ValueError("Actual optimizer response token mismatch")
        advantage=b["advantages"][row,positions]
        old_o=old["log_probs"][use].cuda()
        response_length=b["responses"].shape[-1]
        offline_old,ho=forward(old_model,b["input_ids"][row],b["attention_mask"][row],b["position_ids"][row],response_length,positions,temperature)
        offline_new,hn=forward(new_model,b["input_ids"][row],b["attention_mask"][row],b["position_ids"][row],response_length,positions,temperature)
        # The end update's captured batch is NOT the start batch. For a window,
        # evaluate the end policy on the identical start-batch teacher-forced tokens.
        new_o=new["log_probs"][use].cuda() if new is not None else offline_new
        parity={"old_live_offline_max_abs":float((old_o-offline_old).abs().max()),
                "new_live_offline_max_abs":float((new_o-offline_new).abs().max()) if new is not None else float("nan"),
                "old_chosen_max_abs":float((old_o-offline_old).gather(1,ids[:,None].cuda()).abs().max()),
                "new_chosen_max_abs":float((new_o-offline_new).gather(1,ids[:,None].cuda()).abs().max()) if new is not None else float("nan")}
        base={"global_update":a.update,"row_index":row,"decision_id":meta["decision_id"],
              "trajectory_id":meta["trajectory_id"],"group_id":meta["group_id"],
              "game_id":meta["info"]["extra.gamefile"],"skill_id":skill,
              "environment_step":meta["environment_step"],"phase":phase(meta["environment_step"])}
        if config.get("evaluation",{}).get("anchor_sets"):
            context=meta["info"].get("skill_task_type")
            if not context:raise ValueError("Missing state context on actual training decision")
            base["context_id"]=context
        if a.start_update is not None:
            base.update(start_update=start,window_horizon=a.update-start,direction_batch_update=batch_update)
        witness={"old_original_live":old_o.cpu(),"new_original_live" if new is not None else "end_original_replay":new_o.cpu()}
        for arm in ("placebo","null"):
            cids,cmask,cpos=counter_input(tokenizer,meta,b,row,arm,controls[skill]["text"])
            old_c,hco=forward(old_model,cids,cmask,cpos,response_length,positions,temperature)
            new_c,hcn=forward(new_model,cids,cmask,cpos,response_length,positions,temperature)
            if i<2:
                repeated,_=forward(old_model,cids,cmask,cpos,response_length,positions,temperature)
                noise.extend((repeated-old_c).norm(dim=-1).cpu().tolist())
            signals=token_signals(old_o,new_o,old_c,new_c,ids.cuda(),advantage.cuda())
            # Matched-backend sensitivity quantifies potential live/replay backend drift.
            same_backend=token_signals(offline_old,offline_new,old_c,new_c,ids.cuda(),advantage.cuda())
            signals["P_int_matched_backend"]=same_backend["P_int"]
            signals["D_matched_backend"]=same_backend["D_contribution"]
            signals["C_upd_matched_backend"]=same_backend["C_upd"]
            signals["delta_norm_matched_backend"]=same_backend["delta_norm"]
            for k,pos in enumerate(positions.tolist()):
                token_rows.append({**base,"control":arm,"response_token_offset":pos,
                                   "action_token_id":int(ids[k]),
                                   **{name:value[k].item() for name,value in signals.items()}})
            activations={f"activation_l{layer}_norm":float(((hn[layer]-ho[layer])-(hcn[layer]-hco[layer])).norm(dim=-1).mean())
                         for layer in ho}
            decision_rows.append({**base,"control":arm,"token_count":len(positions),
                                  "advantage_mean":float(advantage.mean()),
                                  **parity,**activations})
            if i==0:
                witness[f"old_{arm}"]=old_c.cpu()
                witness[f"new_{arm}"]=new_c.cpu()
        if i==0:
            save_tensor_file(output/f"witness-shard-{a.shard}.pt",{"metadata":base,"positions":positions,"token_ids":ids,"advantage":advantage,**witness})
        print(f"u{a.update} shard{a.shard} {i+1}/{len(rows)} {skill} s{meta['environment_step']}",flush=True)
    token_frame=pd.DataFrame(token_rows)
    decision_frame=pd.DataFrame(decision_rows)
    if token_frame.empty:
        # A logical shard may have no eligible decisions in a small/contextual
        # batch. Preserve a schema so aggregation can mark support as absent.
        from phase2.aggregate import MEAN_SIGNALS
        common=["decision_id","control","skill_id","context_id","phase","game_id","trajectory_id"]
        for name in set(common+list(MEAN_SIGNALS)+["response_token_offset"]):
            token_frame[name]=pd.Series(dtype="float64" if name in MEAN_SIGNALS else "object")
        for name in ("direction_valid","fidelity_valid"):token_frame[name]=pd.Series(dtype="bool")
        for name in common:decision_frame[name]=pd.Series(dtype="object")
    token_frame.to_parquet(output/f"tokens-shard-{a.shard}.parquet",index=False)
    decision_frame.to_parquet(output/f"decisions-shard-{a.shard}.parquet",index=False)
    atomic_write_json(commit,{"created_at":utc_now(),"update":a.update,"shard":a.shard,
                            "decisions":len(rows),"tokens_with_controls":len(token_rows),
                            "repeat_forward_noise":noise,"max_decisions":a.max_decisions,
                            "tokens_sha256":sha256_file(output/f"tokens-shard-{a.shard}.parquet"),
                            "start_update":start,"direction_batch_update":batch_update,
                            "readout_kind":"single_update" if a.start_update is None else "start_batch_endpoint_projection"})


if __name__=="__main__":main()
