from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_evaluations(root: Path):
    rows=[]
    for directory in sorted((root/"evaluations").glob("u*")):
        markers=list(directory.glob("shard-*-complete.json"))
        if len(markers)!=8:
            continue
        if any(json.loads(x.read_text())["max_jobs"] is not None for x in markers):
            continue
        part=[]
        for file in sorted(directory.glob("shard-*.jsonl")):
            part.extend(json.loads(x) for x in file.read_text().splitlines() if x.strip())
        if len(part)!=1200:
            raise ValueError(f"Incomplete evaluation endpoint {directory}: {len(part)}/1200")
        rows.extend(part)
    return pd.DataFrame(rows)


def margins(evaluations):
    if evaluations.empty:return pd.DataFrame()
    keys=["update","purpose","skill_id","anchor_id","game_id","phase","trigger_step"]
    if evaluations.duplicated(keys+["arm"]).any():
        raise ValueError("Duplicate evaluation branch")
    p=evaluations.pivot(index=keys,columns="arm",values="success").astype(float).reset_index()
    for control in ("placebo","null"):
        p[f"M_{control}"]=p.original-p[control]
    return p


def mean_game(values, column):
    return float(values.groupby("game_id")[column].mean().mean())


def units(root:Path, max_update:int|None=None):
    m=margins(read_evaluations(root))
    if m.empty:return pd.DataFrame(),pd.DataFrame(),m
    all_games,all_units=[],[]
    present=set(m["update"].unique())
    rng=np.random.default_rng(20260909)
    for update in sorted(present):
        if update-1 not in present or (max_update is not None and update>max_update):continue
        old=m[(m["update"]==update-1)&(m.purpose=="gold")]
        new=m[(m["update"]==update)&(m.purpose=="gold")]
        keys=["skill_id","anchor_id","game_id","phase","trigger_step"]
        paired=old.merge(new,on=keys,suffixes=("_old","_new"),validate="one_to_one")
        for control in ("placebo","null"):
            paired["delta"]=paired[f"M_{control}_new"]-paired[f"M_{control}_old"]
            for skill in sorted(paired.skill_id.unique()):
                for phase in ("all","initial","early","middle","late"):
                    q=paired[paired.skill_id==skill]
                    if phase!="all":q=q[q.phase==phase]
                    if q.empty:continue
                    game=q.groupby("game_id").agg(delta=("delta","mean"),old=(f"M_{control}_old","mean"),new=(f"M_{control}_new","mean")).reset_index()
                    vals=game.delta.to_numpy()
                    reps=vals[rng.integers(0,len(vals),size=(10000,len(vals)))].mean(1)
                    low,high=np.quantile(reps,[.025,.975])
                    sign="positive" if low>.05 else "negative" if high<-.05 else "stable" if low>=-.05 and high<=.05 else "uncertain"
                    record={"global_update":int(update),"control":control,"skill_id":skill,"phase":phase,
                            "anchor_count":len(q),"game_count":len(game),"delta_utility":float(vals.mean()),
                            "ci_low":float(low),"ci_high":float(high),"direction":sign,
                            "utility_old":float(game.old.mean()),"utility_new":float(game.new.mean()),
                            "negative_point_label":bool(vals.mean()<-.05)}
                    for endpoint in ("old", "new"):
                        record[f"original_{endpoint}"] = mean_game(q, f"original_{endpoint}")
                        record[f"control_{endpoint}"] = mean_game(q, f"{control}_{endpoint}")
                    record["delta_original"] = record["original_new"]-record["original_old"]
                    record["delta_control"] = record["control_new"]-record["control_old"]
                    all_units.append(record)
                    for row in game.to_dict("records"):
                        all_games.append({"global_update":int(update),"control":control,"skill_id":skill,"phase":phase,**row})
    return pd.DataFrame(all_units),pd.DataFrame(all_games),m


def feature_frame(root:Path, update:int, m:pd.DataFrame):
    path=root/"signals"/f"u{update:04d}"
    f=pd.read_parquet(path/"skill_context_features.parquet")
    f=f[(f.phase=="all")&(f.control=="placebo")].copy()
    if m.empty:
        f["old_margin"]=np.nan
        f["old_margin_se"]=np.nan
    else:
        previous=m[(m["update"]==update-1)&(m.purpose=="evidence")]
        means={s:mean_game(g,"M_placebo") for s,g in previous.groupby("skill_id")}
        f["old_margin"]=f.skill_id.map(means)
        ses={s:float(g.groupby("game_id").M_placebo.mean().std()/np.sqrt(g.game_id.nunique())) for s,g in previous.groupby("skill_id")}
        f["old_margin_se"]=f.skill_id.map(ses)
    raw=json.loads((path/"parameter_delta.json").read_text())
    f["raw_parameter_delta_l2"]=raw["delta_l2"]
    train=Path(__file__).resolve().parents[1]/"artifacts/training_steps"/f"phase2-s303-fast-u{update}"/f"step-{update:06d}.json"
    f["train_success"]=json.loads(train.read_text())["metrics"]["episode/success_rate"]
    return f
