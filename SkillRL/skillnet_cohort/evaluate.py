"""Exhaustive seen/unseen evaluation and natural-anchor collection; plan by default."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .common import digest, exclusive_writer, file_hash, load_preparation, read_json, require_authorization, write_new_json
from .runtime import authorized_runtime, make_runtime
from .inference import make_policy


def job_plan(preparation, checkpoint, update, split, purpose="performance"):
    preparation = Path(preparation).resolve()
    load_preparation(preparation)
    assets = preparation.parent
    spec, inventory = read_json(assets / "spec.json"), read_json(assets / "games.json")
    if split not in spec["evaluation"]["splits"] or purpose not in ("performance", "anchors"):
        raise ValueError("Only explicit seen/unseen performance or anchor collection is allowed")
    if update not in range(0, spec['training']['iterations'] + 1, 5):
        raise ValueError("Checkpoint is outside the fixed five-update schedule")
    if purpose == 'anchors' and split not in spec['evaluation'].get('utility_splits', spec['evaluation']['splits']):
        raise ValueError('This split is not registered for utility anchor collection')
    seeds = spec["evaluation"]["performance_seeds" if purpose == "performance" else "anchor_source_seeds"]
    identity = {"preparation_sha256": file_hash(preparation), "checkpoint": str(Path(checkpoint).resolve()),
                "update": update, "split": split, "purpose": purpose}
    jobs = []
    for game in inventory["splits"][split]["games"]:
        for seed in seeds:
            value = {**identity, **game, "eval_seed": seed,
                     "environment_seed": int(spec["seed"]) + 1000}
            jobs.append({**value, "job_id": digest(value)[:32]})
    return {"identity": identity, "jobs": jobs, "game_count": inventory["splits"][split]["count"],
            "expected_episodes": len(jobs), "execution_started": False}


def collect_results(plan, output):
    """A full-game claim requires exact set equality, not a count or marker alone."""
    output = Path(output)
    expected = {job["job_id"]: job for job in plan["jobs"]}
    if (not expected or len(expected) != len(plan["jobs"])
            or len(expected) != plan["expected_episodes"]
            or len({job["game_id"] for job in plan["jobs"]}) != plan["game_count"]):
        raise ValueError("Invalid/duplicate evaluation job plan")
    paths = list((output / "results").glob("*.json"))
    if plan.get('shard_count', 1) > 1:
        sharded_paths = list((output / 'shards').glob('*/results/*.json'))
        allowed = {f'{i:02d}' for i in range(plan['shard_count'])}
        if any(path.parent.parent.name not in allowed for path in sharded_paths):
            raise ValueError('Unexpected evaluator shard')
        paths.extend(sharded_paths)
    actual = {path.stem: path for path in paths}
    if len(actual) != len(paths):
        raise ValueError('Duplicate result identities across shards')
    if set(actual) - set(expected):
        raise ValueError("Unexpected/foreign evaluation results")
    rows = []
    for key, path in actual.items():
        row = read_json(path)
        if row.get("job") != expected[key] or row.get("status") != "complete":
            raise ValueError("Invalid result identity or unfinished episode")
        if row.get("result_sha256") != digest(row["result"]):
            raise ValueError("Trajectory/result content changed")
        rows.append(row)
    return rows, sorted(set(expected) - set(actual))


def execute_jobs(plan, output, runner):
    """Runner injection permits a complete CPU fake-game regression without API."""
    with exclusive_writer(output):
        return _execute_jobs(plan, output, runner)


def _execute_jobs(plan, output, runner):
    output = Path(output)
    write_new_json(output / "plan.json", plan)
    _, missing = collect_results(plan, output)
    needed = set(missing)
    for job in plan["jobs"]:
        if job["job_id"] not in needed:
            continue
        result = runner(job)
        if not isinstance(result.get("success"), bool) or not result.get("steps"):
            raise ValueError("Runner must return a completed, nonempty episode")
        write_new_json(output / "results" / f"{job['job_id']}.json",
                       {"job": job, "status": "complete", "result": result, "result_sha256": digest(result)})
    rows, missing = collect_results(plan, output)
    if missing:
        raise ValueError("Cannot mark incomplete traversal complete")
    return finish_summary(plan, output, rows)


def finish_summary(plan, output, rows):
    output = Path(output)
    task_rows = {}
    for row in rows:
        task_rows.setdefault(row["job"]["task_type"], []).append(row["result"]["success"])
    summary = {"status": "complete", "plan_sha256": digest(plan), "episodes": len(rows),
               "unique_games": len({row["job"]["game_id"] for row in rows}),
               "expected_games": plan["game_count"], "split": plan["identity"]["split"],
               "success_rate": sum(row["result"]["success"] for row in rows) / len(rows),
               "per_task": {task: {"episodes": len(values), "success_rate": sum(values)/len(values)}
                            for task, values in sorted(task_rows.items())}}
    write_new_json(output / "completion.json", summary)
    return summary


def partition_plan(plan, shard, shards):
    if type(shards) is not int or not 1 <= shards <= 8 or not 0 <= shard < shards:
        raise ValueError('Invalid evaluation shard topology')
    jobs = plan['jobs'][shard::shards]
    if not jobs:
        raise ValueError('Empty evaluator shard; reduce registered worker count')
    return {**plan, 'jobs': jobs, 'game_count': len({j['game_id'] for j in jobs}),
            'expected_episodes': len(jobs), 'partition': {'shard': shard, 'shards': shards}}


def finalize_shards(plan, output, shards):
    output = Path(output)
    if type(shards) is not int or not 1 <= shards <= 8:
        raise ValueError('Invalid evaluator shard count')
    with exclusive_writer(output):
        for shard in range(shards):
            sub = output / 'shards' / f'{shard:02d}'
            part = partition_plan(plan, shard, shards)
            if read_json(sub / 'plan.json') != part:
                raise ValueError('Shard plan differs from the registered full-game partition')
            rows, missing = collect_results(part, sub)
            done = read_json(sub / 'completion.json')
            if missing or done.get('plan_sha256') != digest(part) or done.get('episodes') != len(rows):
                raise ValueError('Incomplete or mismatched evaluator shard')
        full = {**plan, 'shard_count': shards}
        rows, missing = collect_results(full, output)
        if missing:
            raise ValueError('Cannot mark incomplete sharded traversal complete')
        write_new_json(output / 'plan.json', full)
        return finish_summary(full, output, rows)


def result_path(output, job_id):
    output = Path(output)
    paths = ([output / 'results' / f'{job_id}.json'] if (output / 'results' / f'{job_id}.json').is_file() else [])
    paths += list((output / 'shards').glob(f'*/results/{job_id}.json'))
    if len(paths) != 1:
        raise ValueError('Result path missing or duplicated')
    return paths[0]


def verify_prediction(prediction, preparation, update, checkpoint=None):
    """Bind the prospective lock to this preparation, window, features and model."""
    if prediction is None:
        raise PermissionError("Post-update episodes require a locked prediction first")
    prediction = Path(prediction).resolve()
    root = prediction.parents[2]
    config = read_json(root / "protocol.json")
    from phase2.protocol import signal_directory, validate_extended
    validate_extended(config, Path(__file__).resolve().parents[1])
    locked, commit = read_json(prediction), read_json(prediction.parent / "committed.json")
    if (config.get("runtime", {}).get("kind") != "skillnet37"
            or Path(config["runtime"]["preparation"]).resolve() != Path(preparation).resolve()
            or locked.get("global_update") != update
            or locked.get("start_update") != update - 5
            or locked.get("target_gold_read") is not False
            or locked.get("rl_path_id") != config["rl_path_id"]
            or locked.get("protocol_sha256") != file_hash(root / "protocol.json")
            or prediction != (signal_directory(root, update, update - 5) / "prediction.json").resolve()):
        raise PermissionError("Prediction lock does not match this cohort/window")
    features_hash = file_hash(prediction.parent / "skill_context_features.parquet")
    if (locked.get("features_sha256") != features_hash or commit.get("features_sha256") != features_hash
            or commit.get("global_update") != update or commit.get("start_update") != update - 5
            or commit.get("gold_read") is not False):
        raise ValueError("Prediction commitment/features changed")
    if checkpoint is not None and Path(checkpoint).resolve() != (root / "models" / f"u{update:04d}").resolve():
        raise ValueError("Evaluation checkpoint differs from the predicted endpoint")
    return {"path": str(prediction), "sha256": file_hash(prediction), "protocol_sha256": locked["protocol_sha256"]}


def execute(preparation, checkpoint, update, split, purpose, output, authorization, prediction=None,
            *, shard=None, shards=1, collect_shards=False):
    permit = require_authorization(authorization, preparation, "evaluation")
    assets = Path(preparation).resolve().parent
    spec = read_json(assets / "spec.json")
    expected_shards = spec['evaluation'].get('parallel_workers', 1)
    if shards != expected_shards or (shard is not None and collect_shards):
        raise ValueError('Evaluation shard execution differs from the preparation')
    if expected_shards > 1 and shard is None and not collect_shards:
        raise ValueError('Explicit shard or collection is required for parallel evaluation')
    # A later performance/anchor run must not expose target outcomes before the
    # previous five-update window prediction is committed.
    prediction_lock = verify_prediction(prediction, preparation, update, checkpoint) if update > 0 else None
    plan = job_plan(preparation, checkpoint, update, split, purpose)
    # Bind actual weights, not just a reusable directory name.
    from .assets import model_inventory
    plan["checkpoint_identity"] = model_inventory(checkpoint)
    plan["prediction_lock"] = prediction_lock
    if collect_shards:
        return finalize_shards(plan, output, shards)
    if shard is not None:
        plan = partition_plan(plan, shard, shards)
        output = Path(output) / 'shards' / f'{shard:02d}'
    os.environ["ALFWORLD_DATA"] = spec["data_root"]
    from phase1.eval_skill_margin import run_episode
    from phase1.conditions import SkillCondition
    settings = spec["evaluation"]
    # Construct expensive resources lazily: a complete resume performs no API,
    # model load or environment work, and plan conflicts fail first.
    resources = {}
    cache = permit["router_cache_path"]

    def runner(job):
        game = Path(spec["data_root"]) / job["game_id"]
        if file_hash(game) != job["game_sha256"]:
            raise ValueError("Game bytes changed after preparation")
        if not resources:
            memory, router = make_runtime(authorized_runtime(spec, cache, permit))
            resources.update(memory=memory, router=router,
                             policy=make_policy(checkpoint, spec, settings['max_prompt_tokens']))
        result = run_episode(policy=resources["policy"], game_file=game, memory=resources["memory"],
                             condition=SkillCondition.FULL_BANK, skill_id="",
                             environment_seed=job["environment_seed"], eval_seed=job["eval_seed"],
                             temperature=settings["temperature"], top_p=settings["top_p"],
                             max_steps=settings["max_steps"], max_new_tokens=settings["max_new_tokens"],
                             history_length=settings["history_length"], step_skill_router=resources["router"],
                             router_general_top_k=37)
        result.update(trajectory_id=job["job_id"], game_id=job["game_id"], context_id="all_alfworld",
                      environment_seed=job["environment_seed"], eval_seed=job["eval_seed"],
                      checkpoint_id=f"u{update:04d}", max_steps=settings["max_steps"])
        return result

    return execute_jobs(plan, output, runner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--split", choices=["valid_seen", "valid_unseen"], required=True)
    parser.add_argument("--purpose", choices=["performance", "anchors"], default="performance")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--collect-shards", action='store_true')
    args = parser.parse_args()
    if args.execute:
        if args.output is None:
            parser.error("--output is required for execution")
        result = execute(args.preparation, args.checkpoint, args.update, args.split, args.purpose,
                         args.output, args.authorization, args.prediction,
                         shard=args.shard, shards=args.shards, collect_shards=args.collect_shards)
    else:
        result = job_plan(args.preparation, args.checkpoint, args.update, args.split, args.purpose)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
