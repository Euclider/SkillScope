"""Resumable first-invocation evaluation with independent evidence/gold seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from agent_system.memory import FrozenStepSkillRouter, SkillsOnlyMemory
from phase1.archive import append_jsonl_idempotent, atomic_write_json, stable_hash, utc_now, sha256_file
from phase1.eval_first_invocation_utility import run_branch, TransformersPolicy
from phase1.first_invocation import PayloadArm
from phase2.measure import phase
from phase2.protocol import evaluation_jobs, evaluation_identity, parent_update, signal_directory, validate_extended


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--update",type=int,required=True)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=8)
    p.add_argument("--max-jobs",type=int)
    a=p.parse_args()
    torch.set_num_threads(1)
    config=json.loads((a.root/"protocol.json").read_text())
    repo=Path(__file__).resolve().parents[1]
    extended = config.get("schema_version", "").startswith("phase2.extended.")
    preupdate_only = config.get("schema_version") == "phase2.preupdate_baseline.v1"
    if preupdate_only:
        from phase2.queued_preparation import validate_baseline
        validate_baseline(config,a.root,a.update,repo)
    if extended:
        validate_extended(config,repo)
        windows = [w for w in config["windows"] if w["end"] == a.update]
        allowed = {u for w in config["windows"] for u in (w["start"],w["end"])}
        if a.update not in allowed:
            raise ValueError("Endpoint is not in the frozen evaluation schedule")
        for window in windows:
            directory=signal_directory(a.root,a.update,window["start"])
            if not (directory/"committed.json").exists() or not (directory/"prediction.json").exists():
                raise RuntimeError("Window signals AND prediction must be locked before endpoint gold")
    skillnet = config.get("runtime", {}).get("kind") == "skillnet37"
    if skillnet:
        from skillnet_cohort.common import require_authorization
        require_authorization(config["runtime"].get("authorization_path"),
                              config["runtime"]["preparation"], "evaluation")
        if a.update > parent_update(config):
            from skillnet_cohort.evaluate import verify_prediction
            verify_prediction(signal_directory(a.root,a.update,parent_update(config))/"prediction.json",
                              config["runtime"]["preparation"], a.update,
                              a.root/"models"/f"u{a.update:04d}")
    if not skillnet and a.update>parent_update(config) and not (a.root/"signals"/f"u{a.update:04d}"/"committed.json").exists():
        raise RuntimeError("Post-update gold is locked until signals are committed")
    ev=config["evaluation"]
    if a.shards != ev.get("shards",8) or not 0 <= a.shard < a.shards:
        raise ValueError("Shard layout differs from the frozen evaluation protocol")
    jobs=evaluation_jobs(config,repo)
    jobs=jobs[a.shard::a.shards]
    if a.max_jobs:
        jobs=jobs[:a.max_jobs]
    out=a.root/"evaluations"/f"u{a.update:04d}"
    out.mkdir(parents=True,exist_ok=True)
    index=out/f"shard-{a.shard}.jsonl"
    completed={x["trajectory_id"] for x in [json.loads(l) for l in index.read_text().splitlines()]} if index.exists() else set()
    if skillnet:
        from skillnet_cohort.runtime import make_runtime
        from skillnet_cohort.inference import make_policy
        memory, router = make_runtime(config["runtime"])
        policy=make_policy(a.root/"models"/f"u{a.update:04d}", config['runtime'],
                           ev.get("max_prompt_tokens", 4096))
        import os
        os.environ["ALFWORLD_DATA"] = config["runtime"]["data_root"]
    else:
        policy=TransformersPolicy(str(a.root/"models"/f"u{a.update:04d}"))
        memory=SkillsOnlyMemory(str(repo/"memory_data/alfworld/claude_style_skills.json"),retrieval_mode="template",task_specific_top_k=None)
        router=FrozenStepSkillRouter(include_common_mistakes=False)
    for i,(skill,anchor,purpose,seed,arm,placebo) in enumerate(jobs):
        identity=evaluation_identity(config,a.update,(skill,anchor,purpose,seed,arm,placebo))
        tid=stable_hash(identity)[:24]
        if tid in completed:
            continue
        replay_anchor={**anchor,"source_eval_seed":seed}
        result=run_branch(policy=policy,anchor=replay_anchor,memory=memory,router=router,
                          arm=PayloadArm(arm),target_skill_id=skill,placebo_text=placebo["text"],
                          temperature=ev["temperature"],top_p=ev["top_p"],max_new_tokens=ev["max_new_tokens"],
                          history_length=ev["history_length"],router_general_top_k=37 if skillnet else 12)
        path=out/"trajectories"/skill/f"{tid}.json"
        row={**identity,"trajectory_id":tid,"game_id":anchor["game_id"],"state_id":anchor["state_id"],
             "trigger_step":anchor["trigger_step"],"phase":phase(anchor["trigger_step"]),
             "trajectory_path":str(path),"created_at":utc_now(),
             **{k:v for k,v in result.items() if k!="steps"}}
        if extended or preupdate_only:
            row.update(context_id=anchor["context_id"],rl_path_id=config["rl_path_id"])
        atomic_write_json(path,{**row,"original_anchor":anchor,"actual_continuation_seed":seed,"steps":result["steps"]})
        append_jsonl_idempotent(index,[row],unique_fields=("trajectory_id",))
        print(f"u{a.update} shard{a.shard} {i+1}/{len(jobs)} {skill} {purpose} {arm} success={result['success']}",flush=True)
    atomic_write_json(out/f"shard-{a.shard}-complete.json",{"created_at":utc_now(),"jobs":len(jobs),"shard":a.shard,"max_jobs":a.max_jobs,
        "protocol_sha256":sha256_file(a.root/"protocol.json"),"shards":a.shards})


if __name__=="__main__":main()
