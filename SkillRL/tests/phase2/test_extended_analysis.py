import json

import numpy as np
import pandas as pd
import pytest

from phase1.archive import sha256_file
from phase2.aggregate import MEAN_SIGNALS, aggregate_features
from phase2.protocol import make_windows, signal_directory


def test_context_support_and_gate_denominator_are_not_pooled():
    tokens=[]
    for i,(context,valid,projection) in enumerate([("clean",True,-2.),("clean",False,0.),("heat",True,1.)]):
        row={name:0. for name in MEAN_SIGNALS}
        row.update(context_id=context,skill_id="gen_002",control="placebo",phase="early",decision_id=str(i),
                   game_id=context,trajectory_id=str(i),direction_valid=valid,fidelity_valid=valid,
                   P_int=projection,C_upd=1.,delta_norm=1.)
        tokens.append(row)
    token_frame=pd.DataFrame(tokens)
    decisions=token_frame.copy()
    config={"evaluation":{"anchor_sets":[{"skill_id":"gen_002","context_id":context} for context in ["clean","heat","cool"]]},
            "signals":{"minimum_nonzero_advantage_decisions":1,"minimum_training_games":1,"minimum_training_trajectories":1}}
    features,_=aggregate_features(token_frame,decisions,config,40,1e-8)
    q=features[(features.control=="placebo")&(features.phase=="all")].set_index("context_id")
    assert q.loc["clean","D_contribution"]==1.  # 2 / 2, not 2 / 1 active token
    assert q.loc["clean","gate_coverage"]==.5
    assert q.loc["heat","D_contribution"]==0.
    assert q.loc["heat","supported"]
    assert not q.loc["cool","supported"]
    assert pd.isna(q.loc["cool","D_contribution"])


def test_window_forecast_fits_phases_separately_and_only_past_dev(tmp_path,monkeypatch):
    import phase2.window_forecast as wf
    config={"windows":make_windows(0,5,4,1,1),"rl_path_id":"test",
            "prediction":{"base_features":["old_margin"],"candidate_features":["P_int","D_contribution"],
                          "phases":["all","early"],"minimum_development_windows":4,"minimum_training_units":4}}
    (tmp_path/"protocol.json").write_text(json.dumps(config))
    directory=signal_directory(tmp_path,30,25);directory.mkdir(parents=True)
    (directory/"skill_context_features.parquet").write_text("test-hash-only")
    (directory/"committed.json").write_text(json.dumps({"features_sha256":sha256_file(directory/"skill_context_features.parquet")}))
    monkeypatch.setattr(wf,"validate_extended",lambda *a:None)
    labels=[]
    for start in [0,5,10,15,25]:
        for phase in ["all","early"]:
            labels.append({"start_update":start,"global_update":start+5,"control":"placebo","skill_id":"s",
                           "context_id":"clean","phase":phase,"delta_utility":.1 if start<25 else 999.})
    monkeypatch.setattr(wf,"window_units",lambda *a,**k:(pd.DataFrame(labels),pd.DataFrame(),pd.DataFrame()))
    feature_calls=[]
    def feature(*args):
        window=args[2];feature_calls.append(window["end"])
        return pd.DataFrame([{"global_update":window["end"],"start_update":window["start"],"skill_id":"s","context_id":"clean", "phase":phase,
                              "supported":True,"old_margin":.1,"P_int":.2,"D_contribution":.3} for phase in ["all","early"]])
    monkeypatch.setattr(wf,"features",feature)
    fit_calls=[]
    def fit(train,current,*args):
        fit_calls.append(train.copy())
        return np.zeros(len(current))
    monkeypatch.setattr(wf,"fit_predict",fit)
    output=wf.lock_prediction(tmp_path,25,30)
    payload=json.loads(output.read_text())
    assert len(payload["models"])==6
    assert all(len(frame)==4 and frame.phase.nunique()==1 and frame.global_update.max()==20 for frame in fit_calls)
    assert all(frame.delta_utility.max()<999 for frame in fit_calls)
    assert set(feature_calls)=={5,10,15,20,30}
    assert payload["target_gold_read"] is False


def test_window_forecast_refuses_open_target_gold(tmp_path,monkeypatch):
    import phase2.window_forecast as wf
    config={"windows":[{"start":10,"end":15,"role":"test"}]}
    (tmp_path/"protocol.json").write_text(json.dumps(config))
    directory=tmp_path/"evaluations/u0015";directory.mkdir(parents=True)
    (directory/"shard-0.jsonl").write_text("{}\n")
    monkeypatch.setattr(wf,"validate_extended",lambda *a:None)
    with pytest.raises(ValueError,match="already opened"):wf.lock_prediction(tmp_path,10,15)


def test_coverage_uses_natural_occurrences_and_keeps_unsupported(tmp_path):
    from phase2.coverage_audit import audit_coverages
    source=tmp_path/"source";source.mkdir()
    (source/"all").mkdir()
    (source/"coverage.json").write_text(json.dumps({"checkpoint_id":"pre-update","context_id":"clean",
        "coverage":[{"skill_id":"cle_002"},{"skill_id":"cle_005"}]}))
    rows=[{"anchor_id":str(i),"game_id":f"g{i}","source_eval_seed":1,"trigger_step":i+1,
           "context_id":"clean","skill_id":"cle_002","source_checkpoint_id":"pre-update"} for i in range(3)]
    (source/"all/cle_002.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    records,assets=audit_coverages([source/"coverage.json"],tmp_path/"new",tmp_path/"controls","pre-update",2,2,3)
    assert records[0]["supported"] and records[0]["selected_games"]==3
    assert not records[1]["supported"]
    assert len(assets)==1 and "placebo_sha256" not in assets[0]
    assert not records[0]["placebo_ready"]
    with pytest.raises(FileExistsError):audit_coverages([source/"coverage.json"],tmp_path/"new",tmp_path/"controls","pre-update",2,2,3)
