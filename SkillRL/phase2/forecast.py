"""Write predictions before revealing the target endpoint's gold labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase2.utilities import feature_frame, units


BASE=["old_margin","old_margin_se","u_control_norm","delta_norm","raw_parameter_delta_l2","train_success","advantage"]
MODELS={"old_margin":["old_margin","old_margin_se"],"unsigned":BASE,
        "activation":BASE+[f"activation_l{layer}_norm" for layer in (8,16,24,32)],
        "signed":BASE+["C_upd","P_int"],"opposition":BASE+["C_upd","D_contribution"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--update",type=int,required=True)
    a=p.parse_args()
    output=a.root/"predictions"/f"u{a.update:04d}.json"
    if output.exists():return
    if list((a.root/"evaluations"/f"u{a.update:04d}").glob("shard-*.jsonl")):
        raise ValueError("Target endpoint gold is already open; cannot claim prospective prediction")
    config=json.loads((a.root/"protocol.json").read_text())
    utility,_,m=units(a.root,max_update=a.update-1)
    current=feature_frame(a.root,a.update,m)
    training=[]
    if not utility.empty:
        for u in config["development_updates"]:
            if u>=a.update:continue
            labels=utility[(utility.global_update==u)&(utility.phase=="all")&(utility.control=="placebo")]
            if labels.empty:continue
            f=feature_frame(a.root,u,m)
            training.append(f.merge(labels[["skill_id","delta_utility","negative_point_label"]],on="skill_id",validate="one_to_one"))
    train=pd.concat(training,ignore_index=True) if training else pd.DataFrame()
    if not train.empty:
        train=train[train.supported].dropna(subset=BASE)
    predictions=[]
    nonzero=train.loc[train.delta_utility.abs()>1e-12,"delta_utility"] if not train.empty else pd.Series(dtype=float)
    majority=int(np.sign(np.sign(nonzero).sum())) if len(nonzero) else 0
    for _,row in current.iterrows():
        predictions.append({"skill_id":row.skill_id,"supported":bool(row.supported),
                            "P_int":None if pd.isna(row.P_int) else float(row.P_int),
                            "D_t":None if pd.isna(row.D_contribution) else float(row.D_contribution),
                            "old_margin":None if pd.isna(row.old_margin) else float(row.old_margin),
                            "predicted_delta_zero":0.,
                            "predicted_delta_dev_mean":float(train.delta_utility.mean()) if len(train) else None,
                            "predicted_direction_dev_majority":majority})
    learned={}
    if len(train)>=6 and a.update in config["test_updates"]:
        for name,cols in MODELS.items():
            model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
            model.fit(train[cols],train.delta_utility)
            valid=current[cols].notna().all(axis=1)
            if valid.any():
                for index,value in zip(current.index[valid],model.predict(current.loc[valid,cols])):
                    predictions[list(current.index).index(index)][f"predicted_delta_{name}"]=float(value)
            learned[name]={"features":cols,"coefficients_standardized":model[-1].coef_.tolist(),
                           "intercept":float(model[-1].intercept_),"training_rows":len(train)}
            if train.negative_point_label.nunique()==2:
                classifier=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000,random_state=20260909))
                classifier.fit(train[cols],train.negative_point_label)
                if valid.any():
                    for index,value in zip(current.index[valid],classifier.predict_proba(current.loc[valid,cols])[:,1]):
                        predictions[list(current.index).index(index)][f"negative_probability_{name}"]=float(value)
    atomic_write_json(output,{"created_at":utc_now(),"global_update":a.update,
                              "target_gold_read":False,"primary_control":"placebo",
                              "features_sha256":sha256_file(a.root/"signals"/f"u{a.update:04d}"/"skill_context_features.parquet"),
                              "development_updates":config["development_updates"],
                              "analysis_amendment_sha256":sha256_file(a.root/"analysis_amendment_v2.json") if (a.root/"analysis_amendment_v2.json").exists() else None,
                              "simple_baseline_training_rows":len(train),
                              "predictions":predictions,"fitted_models":learned,
                              "status":"learned_temporal_prediction" if learned else "direct_scores_only"})
    print(f"Locked predictions at {output}",flush=True)


if __name__=="__main__":main()
