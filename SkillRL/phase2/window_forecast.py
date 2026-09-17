"""Lock small, matched-complexity window probes before any target gold rollout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase2.protocol import signal_directory, validate_extended
from phase2.utilities import window_units


KEYS = ["skill_id", "context_id", "phase"]


def features(root, config, window, margins):
    directory=signal_directory(root,window["end"],window["start"])
    q=pd.read_parquet(directory/"skill_context_features.parquet")
    phases=config.get("prediction",{}).get("phases",["all"])
    q=q[(q.control=="placebo") & q.phase.isin(phases)].copy()
    evidence=margins[(margins["update"]==window["start"]) & (margins.purpose=="evidence")]
    evidence_rows=[]
    for identity, group in evidence.groupby(KEYS[:2]):
        for phase in phases:
            z=group if phase=="all" else group[group.phase==phase]
            game=z.groupby("game_id").M_placebo.mean()
            evidence_rows.append({"skill_id":identity[0],"context_id":identity[1],"phase":phase,
                                  "old_margin":game.mean(),"old_margin_se":game.std()/np.sqrt(len(game)) if len(game)>1 else np.nan})
    q=q.merge(pd.DataFrame(evidence_rows,columns=KEYS+["old_margin","old_margin_se"]),on=KEYS,how="left",validate="one_to_one")
    q["start_update"]=window["start"]
    q["raw_parameter_delta_l2"]=json.loads((directory/"parameter_delta.json").read_text())["delta_l2"]
    return q


def fit_predict(train, current, numeric, alpha=1.):
    """All candidate additions use one scalar and the same categorical controls."""
    transform=ColumnTransformer([
        ("numeric",StandardScaler(),numeric),
        ("identity",OneHotEncoder(handle_unknown="ignore",sparse_output=False),KEYS),
    ])
    model=make_pipeline(transform,Ridge(alpha=alpha))
    model.fit(train[numeric+KEYS],train.delta_utility)
    return model.predict(current[numeric+KEYS])


def lock_prediction(root, start, end):
    root=Path(root)
    config=json.loads((root/"protocol.json").read_text())
    validate_extended(config,Path(__file__).resolve().parents[1])
    window=next(w for w in config["windows"] if w["start"]==start and w["end"]==end)
    directory=signal_directory(root,end,start)
    output=directory/"prediction.json"
    if output.exists():return output
    if list((root/"evaluations"/f"u{end:04d}").glob("shard-*.jsonl")):
        raise ValueError("Target endpoint already opened; cannot claim a prospective prediction")
    commit=json.loads((directory/"committed.json").read_text())
    if commit["features_sha256"]!=sha256_file(directory/"skill_context_features.parquet"):
        raise ValueError("Window features changed after commitment")
    labels,_,margins=window_units(root,max_update=end-1)
    current=features(root,config,window,margins)
    settings=config["prediction"]
    base=settings["base_features"]
    candidates=settings["candidate_features"]
    train=[]
    for old in config["windows"]:
        if labels.empty:break
        if old["role"]!="development" or old["end"]>=start:continue
        y=labels[(labels.start_update==old["start"]) & (labels.global_update==old["end"]) & (labels.control=="placebo")]
        if not y.empty:
            f=features(root,config,old,margins)
            train.append(f.merge(y[KEYS+["delta_utility"]],on=KEYS,validate="one_to_one"))
    train=pd.concat(train,ignore_index=True) if train else pd.DataFrame()
    rows=[]
    for _, row in current.iterrows():
        rows.append({**{k:row[k] for k in KEYS},"supported":bool(row.supported),
                     "P_int":None if pd.isna(row.P_int) else float(row.P_int),
                     "D":None if pd.isna(row.D_contribution) else float(row.D_contribution),
                     "predicted_delta_zero":0.})
    models={}
    if not train.empty and settings.get("enable_fitted_models",True):
        # Use an identical complete-case set across all additions.
        train=train[train.supported].dropna(subset=base+candidates)
        for phase in settings.get("phases",["all"]):
            phase_train=train[train.phase==phase]
            nwindow=phase_train[["start_update","global_update"]].drop_duplicates().shape[0]
            if window["role"]!="test" or nwindow<settings["minimum_development_windows"] or len(phase_train)<settings["minimum_training_units"]:
                continue
            # 'all' and its component phases share outcomes. Fit separate probes,
            # not duplicated observations masquerading as extra training support.
            valid=current.supported & (current.phase==phase) & current[base+candidates].notna().all(axis=1)
            for name, extra in [("baseline",[])] + [(x,[x]) for x in candidates]:
                numeric=base+extra
                if valid.any():
                    predictions=fit_predict(phase_train,current.loc[valid],numeric,settings.get("ridge_alpha",1.))
                    for i,value in zip(np.flatnonzero(valid.to_numpy()),predictions):
                        rows[i][f"predicted_delta_{name}"]=float(value)
                models[f"{phase}/{name}"]={"numeric":numeric,"categorical":KEYS,"training_rows":len(phase_train),"training_windows":nwindow}
    ranking={}
    if config.get("ranking"):
        from phase2.ranking import score_snapshot
        ranking={"ranking_plan":config["ranking"],"ranking_scores":score_snapshot(current,config["ranking"]),
                 "ranking_primary":True,"fitted_regressor_required":False}
    atomic_write_json(output,{"created_at":utc_now(),"start_update":start,"global_update":end,
        "window_role":window["role"],"rl_path_id":config["rl_path_id"],"target_gold_read":False,
        "protocol_sha256":sha256_file(root/"protocol.json"),"features_sha256":commit["features_sha256"],
        "predictions":rows,"models":models,**ranking,"status":"learned_temporal_prediction" if models else "direct_scores_only_or_insufficient_development_support"})
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--start-update",type=int,required=True)
    p.add_argument("--update",type=int,required=True)
    a=p.parse_args()
    print(lock_prediction(a.root,a.start_update,a.update))


if __name__=="__main__":main()
