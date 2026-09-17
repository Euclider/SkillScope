"""Build a NEW frozen-support proposal using natural pre-update invocations only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase1.build_all_first_invocation_anchors import phase_name, round_robin_games, write_jsonl


def audit_coverages(paths, output, controls, checkpoint_id, minimum_occurrences=30, minimum_games=15, maximum_anchors=50):
    output=Path(output)
    if output.exists():raise FileExistsError("Use a new cohort directory; never overwrite old anchors")
    if maximum_anchors<minimum_games:raise ValueError("Anchor cap cannot support the minimum number of games")
    results,assets,sources=[],[],{}
    identities=set()
    for path in map(Path,paths):
        coverage=json.loads(path.read_text())
        if coverage["checkpoint_id"]!=checkpoint_id:
            raise ValueError("Coverage from a different pre-update policy cannot be silently pooled")
        sources[str(path.resolve())]=sha256_file(path)
        context=coverage["context_id"]
        for row in coverage["coverage"]:
            skill=row["skill_id"]
            identity=(context,skill)
            if identity in identities:raise ValueError("Duplicate context/Skill coverage source")
            identities.add(identity)
            source=path.parent/"all"/f"{skill}.jsonl"
            anchors=[json.loads(line) for line in source.read_text().splitlines() if line.strip()] if source.exists() else []
            if source.exists():sources[str(source.resolve())]=sha256_file(source)
            if any(a["context_id"]!=context or a["skill_id"]!=skill or a["source_checkpoint_id"]!=checkpoint_id for a in anchors):
                raise ValueError("Anchor identity/source checkpoint mismatch")
            if len({a["anchor_id"] for a in anchors})!=len(anchors):raise ValueError("Duplicate natural anchor occurrence")
            games=len({a["game_id"] for a in anchors})
            supported=len(anchors)>=minimum_occurrences and games>=minimum_games
            selected=round_robin_games(anchors,maximum_anchors) if supported else []
            record={"context_id":context,"skill_id":skill,"natural_occurrences":len(anchors),"games":games,
                    "supported":supported,"selected_anchors":len(selected),"selected_games":len({a["game_id"] for a in selected}),
                    **{phase:sum(phase_name(a["trigger_step"])==phase for a in selected) for phase in ("initial","early","middle","late")}}
            control=Path(controls)/f"{skill}.json"
            record["placebo_ready"]=control.exists()
            results.append(record)
            if selected:
                destination=output/"selected"/context/f"{skill}.jsonl"
                write_jsonl(destination,selected)
                spec={"skill_id":skill,"context_id":context,"anchors_path":str(destination.resolve()),
                      "anchors_sha256":sha256_file(destination),"placebo_path":str(control.resolve())}
                if control.exists():spec["placebo_sha256"]=sha256_file(control)
                assets.append(spec)
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(output/"coverage.csv",index=False)
    atomic_write_json(output/"anchor_sets.proposal.json",{"created_at":utc_now(),"status":"support_proposal_not_a_frozen_training_protocol",
        "source_checkpoint_id":checkpoint_id,"selection_uses_post_update_utility":False,
        "minimum_occurrences":minimum_occurrences,"minimum_games":minimum_games,"maximum_anchors":maximum_anchors,
        "coverage":results,"anchor_sets":assets,"anchor_count":sum(r["selected_anchors"] for r in results),"source_sha256":sources})
    return results,assets


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--coverage",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--controls",type=Path,required=True)
    p.add_argument("--checkpoint-id",required=True)
    p.add_argument("--minimum-occurrences",type=int,default=30)
    p.add_argument("--minimum-games",type=int,default=15)
    p.add_argument("--maximum-anchors",type=int,default=50)
    a=p.parse_args()
    rows,_=audit_coverages(a.coverage,a.output,a.controls,a.checkpoint_id,a.minimum_occurrences,a.minimum_games,a.maximum_anchors)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__":main()
