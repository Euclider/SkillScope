"""Commit label-free signals after alignment audits and parameter measurement."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase2.protocol import signal_directory, parent_update


MEAN_SIGNALS = ("P_int","D_contribution","D_ungated_contribution","C_upd","C_upd_centered",
                "delta_norm","delta_centered_norm","u_control_norm","u_original_norm",
                "forward_kl_original","js_original","advantage","P_int_matched_backend","D_matched_backend")


def unsupported_row(update, control, skill, state_phase, tau, decision_columns):
    result = {"global_update":update,"control":control,"skill_id":skill,"phase":state_phase,
              "supported":False,"tau_delta":tau,"unsupported_reason":"no_on_batch_decision"}
    for name in ("token_count","decision_count","nonzero_advantage_decisions","training_games",
                 "nonzero_advantage_games","nonzero_advantage_trajectories"):
        result[name] = 0
    for name in MEAN_SIGNALS+("direction_coverage","gate_coverage","D_decision_equal"):
        result[name] = np.nan
    for name in decision_columns:
        if name.startswith("activation_") or "max_abs" in name:
            result[name] = np.nan
    return result


def parameters(old_path,new_path):
    def inventory(path):
        mapping={}
        for f in path.glob("*.safetensors"):
            with safe_open(f,framework="pt") as h:
                mapping.update({k:f for k in h.keys()})
        return mapping
    old,new=inventory(old_path),inventory(new_path)
    if old.keys()!=new.keys():
        raise ValueError("Parameter names changed")
    norms=[]
    for key in sorted(old):
        # Qwen3.5 ties the language head to input embeddings.
        if key=="lm_head.weight":
            continue
        with safe_open(old[key],framework="pt") as oh,safe_open(new[key],framework="pt") as nh:
            a,b=oh.get_tensor(key),nh.get_tensor(key)
            if a.dtype!=torch.float32 or b.dtype!=torch.float32 or a.shape!=b.shape:
                raise ValueError(f"FP32 parameter mismatch: {key}")
            diff=b-a
            ds=float(diff.square().sum(dtype=torch.float64))
            os=float(a.square().sum(dtype=torch.float64))
            norms.append({"parameter":key,"elements":a.numel(),"delta_squared":ds,"old_squared":os})
    ds=sum(x["delta_squared"] for x in norms)
    os=sum(x["old_squared"] for x in norms)
    return {"delta_l2":math.sqrt(ds),"relative_delta_l2":math.sqrt(ds/os),
            "unique_parameters":sum(x["elements"] for x in norms),"tensors":norms}


def aggregate_features(tokens, decisions, config, update, tau):
    """Keep natural training support separate by Skill, context and state phase."""
    tokens=tokens.copy()
    settings=config.get("signals",{})
    tokens["gate"]=(tokens.direction_valid & tokens.fidelity_valid & (tokens.C_upd>=settings.get("tau_C",0.)) & (tokens.delta_norm>=tau))
    tokens["D_contribution"]=tokens.gate*np.maximum(-tokens.P_int,0)
    ev=config["evaluation"]
    pairs=[(s["skill_id"],s["context_id"]) for s in ev["anchor_sets"]] if ev.get("anchor_sets") else [(s,None) for s in ev["skills"]]
    rows=[]
    for control in ("placebo","null"):
        for skill,context in pairs:
            for state_phase in ("all","initial","early","middle","late"):
                t=tokens[(tokens.control==control)&(tokens.skill_id==skill)]
                d=decisions[(decisions.control==control)&(decisions.skill_id==skill)]
                if context is not None:
                    t=t[t.context_id==context];d=d[d.context_id==context]
                if state_phase!="all":
                    t=t[t.phase==state_phase];d=d[d.phase==state_phase]
                if t.empty:
                    row=unsupported_row(update,control,skill,state_phase,tau,decisions.columns)
                else:
                    active=t[t.direction_valid].drop_duplicates("decision_id")
                    nactive=len(active)
                    supported=(nactive>=settings.get("minimum_nonzero_advantage_decisions",20)
                               and active.game_id.nunique()>=settings.get("minimum_training_games",4)
                               and active.trajectory_id.nunique()>=settings.get("minimum_training_trajectories",8))
                    row={"global_update":update,"control":control,"skill_id":skill,"phase":state_phase,
                         "token_count":len(t),"decision_count":t.decision_id.nunique(),
                         "nonzero_advantage_decisions":nactive,"training_games":t.game_id.nunique(),
                         "nonzero_advantage_games":active.game_id.nunique(),
                         "nonzero_advantage_trajectories":active.trajectory_id.nunique(),
                         "supported":bool(supported),"unsupported_reason":None if supported else "insufficient_nonzero_advantage_support",
                         "direction_coverage":float(t.direction_valid.mean()),
                         "gate_coverage":float(t.gate.mean()),"tau_delta":tau}
                    for col in MEAN_SIGNALS:row[col]=float(t[col].mean())
                    row["D_decision_equal"]=float(t.groupby("decision_id").D_contribution.mean().mean())
                    for col in d.columns:
                        if col.startswith("activation_") or "max_abs" in col:row[col]=float(d[col].mean())
                if context is not None:row["context_id"]=context
                rows.append(row)
    return pd.DataFrame(rows),tokens


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--update",type=int,required=True)
    p.add_argument("--start-update",type=int)
    p.add_argument("--shards",type=int,default=8)
    a=p.parse_args()
    torch.set_num_threads(1)
    config=json.loads((a.root/"protocol.json").read_text())
    start=a.update-1 if a.start_update is None else a.start_update
    batch_update=start+1
    out=signal_directory(a.root,a.update,a.start_update)
    if (out/"committed.json").exists():return
    from phase2.audit import audit
    if a.start_update is not None and list((a.root/"evaluations"/f"u{a.update:04d}").glob("shard-*.jsonl")):
        raise ValueError("Target gold already open before window signal commitment")
    audit(a.root,batch_update)
    manifests=[json.loads((out/f"shard-{s}.json").read_text()) for s in range(a.shards)]
    if any(x["max_decisions"] is not None for x in manifests):
        raise ValueError("A smoke shard cannot be committed as a full measurement")
    tokens=pd.concat([pd.read_parquet(out/f"tokens-shard-{s}.parquet") for s in range(a.shards)],ignore_index=True)
    decisions=pd.concat([pd.read_parquet(out/f"decisions-shard-{s}.parquet") for s in range(a.shards)],ignore_index=True)
    if tokens.duplicated(["decision_id","control","response_token_offset"]).any():
        raise ValueError("Duplicate decision-token measurement")
    if any(x.get("start_update",a.update-1)!=start or x.get("direction_batch_update",a.update)!=batch_update for x in manifests):
        raise ValueError("Mixed endpoint/direction-batch window shards")
    bm=json.loads((a.root/"batches"/f"u{batch_update:04d}"/"manifest.json").read_text())
    if a.start_update is not None and config.get('window_direction') != 'start_batch_endpoint_projection':
        raise ValueError('Endpoint replay requires the registered window direction')
    for stage in (("old", "new") if a.start_update is None else ("old",)):
        paths=list((a.root/f"{stage}_logprobs"/f"u{batch_update:04d}").glob("row-*.pt"))
        if len(paths)!=bm["row_count"]:
            raise ValueError(f"Missing {stage} live logit rows: {len(paths)}/{bm['row_count']}")
    window_evidence = None
    if config.get('capture_scope') == 'window_start_old_only_v1':
        if a.start_update is None:
            raise ValueError('OLD-only capture cannot supply single-update live NEW signals')
        from phase2.window_evidence import validate_shards
        window_evidence = validate_shards(a.root, start, a.update, a.shards)
    calibration=a.root/"signals"/"calibration.json"
    if not calibration.exists():
        if batch_update!=parent_update(config)+1:raise ValueError("First-update calibration is required")
        noise=[z for x in manifests for z in x["repeat_forward_noise"]]
        if not noise:raise ValueError("No same-checkpoint noise calibration")
        threshold=max(1e-8,10*float(np.quantile(noise,.95)))
        calibration_record={"created_at":utc_now(),"source_update":batch_update,
                                     "noise_p95":float(np.quantile(noise,.95)),
                                     "noise_max":max(noise),"samples":len(noise),"tau_delta":threshold,
                                     "gold_read":False}
    else:
        calibration_record=json.loads(calibration.read_text())
    tau=calibration_record['tau_delta']
    features,tokens=aggregate_features(tokens,decisions,config,a.update,tau)
    if a.start_update is not None:
        features["start_update"]=start
        features["window_horizon"]=a.update-start
        features["direction_batch_update"]=batch_update
    if window_evidence is not None and start == 0:
        from skillnet_cohort.parameter_delta import registered_initial_delta
        raw=registered_initial_delta(config, a.root/'models'/f'u{a.update:04d}')
    else:
        raw=parameters(a.root/"models"/f"u{start:04d}",a.root/"models"/f"u{a.update:04d}")
    opt=[]
    for u in range(batch_update,a.update+1):
        f=a.root/"optimizer_steps"/f"u{u:04d}-rank0.jsonl"
        if not f.exists():raise ValueError(f"Missing intermediate optimizer record: {u}")
        opt.extend(json.loads(x) for x in f.read_text().splitlines() if x.strip())
    if not opt:raise ValueError("Optimizer-step provenance is missing")
    # Finish validation before publishing outputs, and never replace a partial
    # or historical aggregate on an implicitly retried invocation.
    from skillnet_cohort.common import write_new_bytes, write_new_json
    if not calibration.exists():
        write_new_json(calibration, calibration_record)
    write_new_bytes(out/'skill_context_features.parquet', features.to_parquet(index=False))
    write_new_bytes(out/'token_signals.parquet', tokens.to_parquet(index=False))
    write_new_json(out/'parameter_delta.json', raw)
    if window_evidence is not None:
        write_new_json(out/'endpoint_replay_audit.json', window_evidence)
    commitment={
        "created_at":utc_now(),"global_update":a.update,"gold_read":False,
        "features_sha256":sha256_file(out/"skill_context_features.parquet"),
        "token_signals_sha256":sha256_file(out/"token_signals.parquet"),
        "live_old_and_new_rows":bm["row_count"],"unique_decisions":decisions.decision_id.nunique(),
        "optimizer_steps":len(opt),"adam_step_before":min(x["adam_step_before"] for x in opt),
        "adam_step_after":max(x["adam_step_after"] for x in opt),
        "raw_parameter_delta_l2":raw["delta_l2"],"tau_delta":tau,
        "start_update":start,"direction_batch_update":batch_update,
        "readout_kind":"single_update" if a.start_update is None else "start_batch_endpoint_projection"}
    if a.start_update is not None:
        commitment["live_start_batch_rows"]=commitment.pop("live_old_and_new_rows")
        commitment["end_original_source"]="FP32-exported endpoint replay on the start batch; not end-update live batch"
    write_new_json(out/"committed.json",commitment)
    print(features.query("phase=='all'")[["control","skill_id","supported","P_int","D_contribution","gate_coverage"]].to_string(index=False),flush=True)


if __name__=="__main__":main()
