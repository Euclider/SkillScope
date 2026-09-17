from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.protocol import expected_trajectory_ids, seed_streams, validate_windows
from phase1.archive import sha256_file


def read_evaluations(root: Path):
    root = Path(root)
    protocol_path = root / "protocol.json"
    config = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    repo = Path(__file__).resolve().parents[1]
    shards = config.get("evaluation", {}).get("shards", 8) if config else 8
    rows=[]
    for directory in sorted((root/"evaluations").glob("u*")):
        markers=list(directory.glob("shard-*-complete.json"))
        if len(markers)!=shards or {p.name for p in markers} != {f"shard-{s}-complete.json" for s in range(shards)}:
            continue
        if any(json.loads(x.read_text())["max_jobs"] is not None for x in markers):
            continue
        if config:
            digest=sha256_file(protocol_path)
            if any(json.loads(x.read_text()).get("protocol_sha256",digest)!=digest for x in markers):
                raise ValueError("Evaluation protocol changed after a shard was committed")
        part=[]
        for file in sorted(directory.glob("shard-*.jsonl")):
            part.extend(json.loads(x) for x in file.read_text().splitlines() if x.strip())
        if config:
            expected = expected_trajectory_ids(config, repo, int(directory.name[1:]))
            observed = [row["trajectory_id"] for row in part]
            if len(observed) != len(expected) or set(observed) != set(expected):
                raise ValueError(f"Incomplete/changed evaluation endpoint {directory}: {len(observed)}/{len(expected)}")
        elif len(part)!=1200:
            raise ValueError(f"Incomplete evaluation endpoint {directory}: {len(part)}/1200")
        rows.extend(part)
    return pd.DataFrame(rows)


def margins(evaluations):
    if evaluations.empty:return pd.DataFrame()
    keys=["update","purpose","skill_id","anchor_id","game_id","phase","trigger_step"]
    keys += [key for key in ("context_id", "continuation_seed") if key in evaluations]
    if evaluations.duplicated(keys+["arm"]).any():
        raise ValueError("Duplicate evaluation branch")
    p=evaluations.pivot(index=keys,columns="arm",values="success").astype(float).reset_index()
    if any(c not in p for c in ("original","placebo","null")) or p[["original","placebo","null"]].isna().any().any():
        raise ValueError("Every matched repeat requires all three arms")
    for control in ("placebo","null"):
        p[f"M_{control}"]=p.original-p[control]
    return p


def mean_game(values, column):
    return float(values.groupby("game_id")[column].mean().mean())


def paired_bootstrap(q, rng, repetitions=10000):
    """Paired game bootstrap; with repeats, resample paired seeds inside anchors.

    All arm/endpoint contrasts are formed before resampling. Preserve the old
    one-repeat random stream and results exactly for the historical cohort.
    """
    game = q.groupby("game_id").delta.mean()
    if "continuation_seed" not in q or q.continuation_seed.nunique() == 1:
        values = game.to_numpy()
        return values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(1)
    distributions = []
    for _, rows in q.groupby("game_id"):
        values = rows.pivot(index="anchor_id", columns="continuation_seed", values="delta").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("Missing paired continuation repeats inside an anchor")
        nanchors, nseeds = values.shape
        draws = rng.integers(0, nseeds, size=(repetitions, nanchors, nseeds))
        distributions.append(values[np.arange(nanchors)[None, :, None], draws].mean((1, 2)))
    pool = np.stack(distributions)
    game_draws = rng.integers(0, len(pool), size=(repetitions, len(pool)))
    repeat_draws = rng.integers(0, repetitions, size=game_draws.shape)
    return pool[game_draws, repeat_draws].mean(1)


def _units_from_margins(m, windows, tau=.05, repetitions=10000):
    if m.empty:return pd.DataFrame(),pd.DataFrame(),m
    all_games,all_units=[],[]
    present=set(m["update"].unique())
    rng=np.random.default_rng(20260909)
    for window in windows:
        start, update = window["start"], window["end"]
        if start not in present or update not in present:continue
        old=m[(m["update"]==start)&(m.purpose=="gold")]
        new=m[(m["update"]==update)&(m.purpose=="gold")]
        keys=["skill_id","anchor_id","game_id","phase","trigger_step"]
        keys += [key for key in ("context_id", "continuation_seed") if key in m]
        paired=old.merge(new,on=keys,suffixes=("_old","_new"),validate="one_to_one",how="outer",indicator=True)
        if not paired._merge.eq("both").all():
            raise ValueError("Old/new endpoints must have exactly matched anchors and continuation seeds")
        for control in ("placebo","null"):
            paired["delta"]=paired[f"M_{control}_new"]-paired[f"M_{control}_old"]
            groups = ["skill_id"] + (["context_id"] if "context_id" in paired else [])
            for identity, skill_rows in paired.groupby(groups):
                identity = identity if isinstance(identity, tuple) else (identity,)
                group_meta = dict(zip(groups, identity))
                skill = group_meta["skill_id"]
                for phase in ("all","initial","early","middle","late"):
                    q=skill_rows
                    if phase!="all":q=q[q.phase==phase]
                    if q.empty:continue
                    game=q.groupby("game_id").agg(delta=("delta","mean"),old=(f"M_{control}_old","mean"),new=(f"M_{control}_new","mean")).reset_index()
                    vals=game.delta.to_numpy()
                    reps=paired_bootstrap(q,rng,repetitions)
                    low,high=np.quantile(reps,[.025,.975])
                    sign="positive" if low>tau else "negative" if high<-tau else "stable" if low>=-tau and high<=tau else "uncertain"
                    record={"global_update":int(update),"control":control,"skill_id":skill,"phase":phase,
                            "anchor_count":q.anchor_id.nunique(),"game_count":len(game),"delta_utility":float(vals.mean()),
                            "ci_low":float(low),"ci_high":float(high),"direction":sign,
                            "utility_old":float(game.old.mean()),"utility_new":float(game.new.mean()),
                            "negative_point_label":bool(vals.mean()<-tau)}
                    if "context_id" in group_meta:record["context_id"] = group_meta["context_id"]
                    if "role" in window:
                        record.update(start_update=start, window_horizon=update-start, window_role=window["role"],
                                      continuation_repeats=q.continuation_seed.nunique() if "continuation_seed" in q else 1,
                                      paired_anchor_repeats=len(q))
                    for endpoint in ("old", "new"):
                        record[f"original_{endpoint}"] = mean_game(q, f"original_{endpoint}")
                        record[f"control_{endpoint}"] = mean_game(q, f"{control}_{endpoint}")
                    record["delta_original"] = record["original_new"]-record["original_old"]
                    record["delta_control"] = record["control_new"]-record["control_old"]
                    all_units.append(record)
                    for row in game.to_dict("records"):
                        meta={key:record[key] for key in ("context_id", "start_update", "window_horizon", "window_role") if key in record}
                        all_games.append({"global_update":int(update),"control":control,"skill_id":skill,"phase":phase,**meta,**row})
    return pd.DataFrame(all_units),pd.DataFrame(all_games),m


def units(root:Path, max_update:int|None=None):
    m=margins(read_evaluations(root))
    if m.empty:return pd.DataFrame(),pd.DataFrame(),m
    present=set(m["update"].unique())
    windows=[{"start":u-1,"end":u} for u in sorted(present) if u-1 in present and (max_update is None or u<=max_update)]
    return _units_from_margins(m,windows)


def window_units(root:Path, max_update:int|None=None):
    config=json.loads((Path(root)/"protocol.json").read_text())
    seed_streams(config)
    windows=validate_windows(config["windows"])
    windows=[w for w in windows if max_update is None or w["end"]<=max_update]
    return _units_from_margins(margins(read_evaluations(root)),windows,
                               tau=config["evaluation"].get("tau_M",.05),
                               repetitions=config["evaluation"].get("bootstrap_repetitions",10000))


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
