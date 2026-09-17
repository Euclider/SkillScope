import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase1.archive import sha256_file, stable_hash
from phase2.protocol import (anchor_sets, evaluation_jobs, evaluation_identity, expected_trajectory_ids,
                             make_windows, seed_streams, validate_extended, validate_windows)
from phase2.utilities import _units_from_margins, margins, read_evaluations


def protocol(tmp_path):
    sets=[]
    for context in ("clean","heat"):
        anchors=[{"anchor_id":f"{context}-{i}","skill_id":"gen_002","context_id":context,
                  "game_id":f"{context}-game-{i}","phase":"middle","trigger_step":7} for i in range(2)]
        path=tmp_path/f"{context}.jsonl"
        path.write_text("\n".join(json.dumps(a) for a in anchors)+"\n")
        placebo=tmp_path/"placebo.json"
        placebo.write_text(json.dumps({"text":"neutral", "original_text":"skill"}))
        sets.append({"skill_id":"gen_002","context_id":context,"anchors_path":str(path),"anchors_sha256":sha256_file(path),
                     "placebo_path":str(placebo),"placebo_sha256":sha256_file(placebo)})
    return {"schema_version":"phase2.extended.direction.v2","status":"frozen","run_id":"unit-test-v2",
            "root":str(tmp_path),"rl_path_id":"independent-test-path","parent_update":30,
            "post_updates":list(range(31,46)),"windows":make_windows(30,5,1,1,1),
            "window_direction":"start_batch_endpoint_projection","parent_checkpoint":str(tmp_path/"parent"),
            "training":{"run_id_prefix":"phase2-test-only","sampler_seed_base":12300,"games_per_update":16,
                        "rollouts_per_game":4,"reference":"/model/local","max_prompt_tokens":2048,"max_response_tokens":64,
                        "max_environment_steps":30,"lr":1e-6,"kl_coef":.01,"task_types":[2,3]},
            "evaluation":{"anchor_sets":sets,"anchor_count":4,"shards":2,"arms":["original","placebo","null"],
                          "old_evidence_seeds":[1,2],"gold_seeds":[3,4,5,6]}}


def test_dynamic_jobs_preserve_contexts_and_seed_identity(tmp_path):
    config=protocol(tmp_path)
    validate_extended(config,tmp_path)
    jobs=evaluation_jobs(config,tmp_path)
    assert len(jobs)==4*3*6
    ids=expected_trajectory_ids(config,tmp_path,35)
    assert len(set(ids))==len(jobs)
    assert {job[1]["context_id"] for job in jobs}=={"clean","heat"}
    assert ids[0]==stable_hash(evaluation_identity(config,35,jobs[0]))[:24]


@pytest.mark.parametrize("change",["duplicate_seed","shared_seed","wrong_arm"])
def test_invalid_seed_or_arm_protocol_is_rejected(tmp_path,change):
    config=protocol(tmp_path)
    if change=="duplicate_seed":config["evaluation"]["gold_seeds"]=[3,3]
    if change=="shared_seed":config["evaluation"]["gold_seeds"]=[1,3]
    if change=="wrong_arm":config["evaluation"]["arms"]=["original","placebo"]
    with pytest.raises(ValueError):seed_streams(config)


def test_frozen_asset_mutation_is_rejected(tmp_path):
    config=protocol(tmp_path)
    Path(config["evaluation"]["anchor_sets"][0]["anchors_path"]).write_text("{}\n")
    with pytest.raises(ValueError,match="Frozen asset changed"):anchor_sets(config,tmp_path)


def test_draft_cannot_launch(tmp_path):
    config=protocol(tmp_path);config["status"]="draft"
    with pytest.raises(ValueError,match="Draft protocol cannot launch"):validate_extended(config,tmp_path)


def test_window_split_has_real_endpoint_isolation():
    windows=make_windows(35,5,4,1,4)
    assert len(windows)==9
    assert max(w["end"] for w in windows if w["role"]=="development")==55
    assert min(w["start"] for w in windows if w["role"]=="test")==60
    with pytest.raises(ValueError,match="share an endpoint"):
        validate_windows([{"start":30,"end":35,"role":"development"},{"start":35,"end":40,"role":"test"}])
    with pytest.raises(ValueError,match="overlap"):
        validate_windows([{"start":30,"end":35,"role":"development"},{"start":34,"end":39,"role":"development"}])


def test_loader_checks_dynamic_full_job_set_not_1200(tmp_path):
    config=protocol(tmp_path)
    (tmp_path/"protocol.json").write_text(json.dumps(config))
    directory=tmp_path/"evaluations/u0035";directory.mkdir(parents=True)
    jobs=evaluation_jobs(config,tmp_path)
    for shard in range(2):
        rows=[{**evaluation_identity(config,35,job),"trajectory_id":stable_hash(evaluation_identity(config,35,job))[:24]} for job in jobs[shard::2]]
        (directory/f"shard-{shard}.jsonl").write_text("\n".join(json.dumps(row) for row in rows)+"\n")
        (directory/f"shard-{shard}-complete.json").write_text(json.dumps({"max_jobs":None,"jobs":len(rows)}))
    assert len(read_evaluations(tmp_path))==72
    index=directory/"shard-0.jsonl"
    lines=index.read_text().splitlines()
    index.write_text("\n".join(lines[:-1])+"\n")
    with pytest.raises(ValueError,match="Incomplete/changed"):read_evaluations(tmp_path)


def observations():
    rows=[]
    for context in ("clean","heat"):
        for update in (30,35):
            for seed in (101,102,103,104):
                for arm in ("original","placebo","null"):
                    success=update==35 and context=="clean" and arm=="original" and seed in (101,102)
                    rows.append(dict(update=update,purpose="gold",skill_id="gen_002",context_id=context,
                                     anchor_id=context,game_id=context,phase="middle",trigger_step=7,
                                     continuation_seed=seed,arm=arm,success=success))
    return pd.DataFrame(rows)


def test_repeats_remain_paired_and_contexts_never_pool():
    m=margins(observations())
    u,_,_=_units_from_margins(m,[{"start":30,"end":35,"role":"test"}])
    q=u[(u.control=="placebo")&(u.phase=="all")].set_index("context_id")
    assert q.loc["clean","delta_utility"]==.5
    assert q.loc["heat","delta_utility"]==0.
    assert q.loc["clean","anchor_count"]==1
    assert q.loc["clean","paired_anchor_repeats"]==4
    assert q.loc["clean","continuation_repeats"]==4
    # One game with nonidentical continuations still has seed uncertainty.
    assert q.loc["clean","ci_low"]==0
    assert q.loc["clean","ci_high"]==1


def test_missing_matched_seed_fails_instead_of_inner_join_dropping():
    m=margins(observations())
    m=m[~((m["update"]==35)&(m.continuation_seed==101))]
    with pytest.raises(ValueError,match="exactly matched"):
        _units_from_margins(m,[{"start":30,"end":35}])


def test_duplicate_repeat_is_rejected():
    data=observations()
    with pytest.raises(ValueError,match="Duplicate"):
        margins(pd.concat([data,data.iloc[:1]],ignore_index=True))


def test_launch_spec_uses_env_rollout_group_and_unique_artifact_names(tmp_path):
    from phase2.launch_training import launch_spec
    config=protocol(tmp_path)
    (tmp_path/"protocol.json").write_text(json.dumps(config))
    command,env=launch_spec(config,31,tmp_path,{"PHASE2_GPU_IDS":"3,6"})
    assert "env.rollout.n=4" in command
    assert not any(s.startswith("actor_rollout_ref.rollout.n=") for s in command)
    assert "env.alfworld.task_types=[2,3]" in command
    assert env["PHASE1_RUN_ID"]=="phase2-test-only-u31"
    assert env["PHASE1_TRAIN_DATA_SIZE"]=="16"
    assert env["PHASE2_CPU_ADAM"]=="1"
    assert env["PHASE1_RL_SEED"]=="12331"


def test_expanded_rollout_opt_in_does_not_change_optimizer_batch():
    from omegaconf import OmegaConf
    from phase2.elastic_training import validate_elastic_dispatch
    config=OmegaConf.create({"phase2":{"enabled":True,"allow_expanded_rollout_batch":True},"algorithm":{"adv_estimator":"grpo"},
        "data":{"train_batch_size":16},"env":{"rollout":{"n":4}},
        "actor_rollout_ref":{"model":{"load_text_only":True},"rollout":{"name":"hf","n":1},
        "actor":{"ppo_mini_batch_size":32,"ppo_micro_batch_size_per_gpu":1,"use_dynamic_bsz":False}}})
    validate_elastic_dispatch(config)
    config.actor_rollout_ref.actor.ppo_mini_batch_size=64
    with pytest.raises(ValueError):validate_elastic_dispatch(config)


def test_window_report_archives_dynamic_repeated_cohort(tmp_path,monkeypatch):
    import sys
    from phase2.window_report import main
    config=protocol(tmp_path)
    config["evaluation"]["bootstrap_repetitions"]=100
    (tmp_path/"protocol.json").write_text(json.dumps(config))
    jobs=evaluation_jobs(config,tmp_path)
    for update in (30,35):
        directory=tmp_path/f"evaluations/u{update:04d}";directory.mkdir(parents=True)
        for shard in range(2):
            rows=[]
            for job in jobs[shard::2]:
                _,anchor,_,seed,arm,_=job
                identity=evaluation_identity(config,update,job)
                rows.append({**identity,"trajectory_id":stable_hash(identity)[:24],
                             **{k:anchor[k] for k in ("game_id","context_id","phase","trigger_step")},
                             "success":update==35 and arm=="original" and seed%2==0})
            (directory/f"shard-{shard}.jsonl").write_text("\n".join(json.dumps(x) for x in rows)+"\n")
            (directory/f"shard-{shard}-complete.json").write_text(json.dumps({"max_jobs":None,"jobs":len(rows)}))
    monkeypatch.setattr(sys,"argv",["window_report","--root",str(tmp_path)])
    main()
    units=pd.read_parquet(tmp_path/"window_metrics/utility_units.parquet")
    assert units.anchor_count.eq(2).all()
    assert units.continuation_repeats.eq(4).all()
    assert units.delta_utility.eq(.5).all()
    assert (tmp_path/"reports/window-results.md").exists()


def test_extended_runner_locks_window_before_gold_without_evaluating_intermediate(tmp_path,monkeypatch):
    import phase2.run_extended as runner
    from types import SimpleNamespace
    p=runner.ExtendedPipeline.__new__(runner.ExtendedPipeline)
    p.extended=True;p.root=tmp_path;p.repo=tmp_path
    p.config={"parent_update":30,"parent_checkpoint":str(tmp_path/"parent"),"post_updates":[31,32],
              "evaluation":{"shards":2},"windows":[{"start":30,"end":32,"role":"development"}]}
    (tmp_path/"models/u0030").mkdir(parents=True)
    sequence=[]
    p.parallel=lambda module,update,start_update=None:sequence.append((module,update,start_update))
    p.command=lambda command,name:sequence.append((name,))
    p.train=lambda update:sequence.append(("train",update))
    p.status=lambda *a,**k:None;p.refresh_report=lambda:None
    monkeypatch.setattr(runner,"read_committed_step",lambda _:0)
    monkeypatch.setattr(runner,"validate_full_checkpoint",lambda _:None)
    monkeypatch.setattr(runner.shutil,"disk_usage",lambda _:SimpleNamespace(free=1000*2**30))
    p.run()
    assert ("phase2.evaluate",31,None) not in sequence
    assert ("phase2.measure",31,None) in sequence
    assert ("phase2.measure",32,None) in sequence
    assert sequence.index(("phase2.measure",32,30)) < sequence.index(("aggregate-window-30-32",))
    assert sequence.index(("forecast-window-30-32",)) < sequence.index(("phase2.evaluate",32,None))
