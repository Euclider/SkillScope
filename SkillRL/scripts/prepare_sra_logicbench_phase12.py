"""Freeze routed LogicBench Aug schedules and Eval routes before policy training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pandas as pd

from agent_system.memory.frozen_skill_bank import FrozenSkillBankMemory
from agent_system.memory.logicbench_embedding_router import LogicBenchEmbeddingRouter
from agent_system.memory.skillrl_embedding_batch_router import load_batch_profile
from agent_system.memory.sra_logicbench_bank import load_sra_logicbench19
from phase1.logicbench_single_step import LogicBenchQuestion, build_prompt, load_logicbench_eval

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/logicbench/sra19/aug_split_v2"
EXPECTED_TRAIN_SHA256 = "7ffdf4dcb003a8b0a4b5a917df2da25678d5378498073ec115b68a4c3d33aca3"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_questions(rows, *, seed: int, count: int) -> list[dict]:
    """Stratified by source pattern, with one question per Aug context."""
    rng = random.Random(seed)
    by_pattern = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_pattern[row["skill_pattern_id"]][row["context_id"]].append(row)
    queues = {}
    for pattern, contexts in by_pattern.items():
        chosen = [rng.choice(group) for group in contexts.values()]
        rng.shuffle(chosen)
        queues[pattern] = chosen
    selected, seen_contexts = [], set()
    patterns = sorted(queues)
    while len(selected) < count and any(queues.values()):
        for pattern in patterns:
            if not queues[pattern]:
                continue
            row = queues[pattern].pop()
            if row["context_id"] in seen_contexts:
                continue
            seen_contexts.add(row["context_id"])
            selected.append(row)
            if len(selected) == count:
                break
    if len(selected) != count or len({r["question_id"] for r in selected}) != count:
        raise ValueError("Insufficient distinct LogicBench training contexts")
    return selected


def _route(router, memory, rows):
    requests = [{"candidate_bundle": memory.retrieve(""),
                 "question": row.question if isinstance(row, LogicBenchQuestion) else row["question"]}
                for row in rows]
    return router.route_many(requests)


def _parquet_rows(rows, routed, bank):
    result = []
    for row, decision in zip(rows, routed):
        skill_id = decision["selected_skill_id"]
        if skill_id not in bank.skill_ids:
            raise ValueError("Router selected a skill outside the frozen bank")
        question = LogicBenchQuestion(row["question_id"], row["question"],
                                      row["task_type"], row["answer"], skill_id)
        prompt = build_prompt(question, bank.get(skill_id).payload)
        result.append({"data_source": "sra_logicbench_aug_v2",
                       "prompt": [{"role": "user", "content": prompt}],
                       "question_id": row["question_id"],
                       "context_id": row.get("context_id", ""),
                       "task_type": row["task_type"], "answer": row["answer"],
                       "selected_skill_id": skill_id,
                       "router_cache_key": decision["skill_router_api"]["cache_key"]})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--router-cache", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[404, 505])
    parser.add_argument("--questions-per-update", type=int, default=128)
    parser.add_argument("--updates", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Prepared output must be new and empty")
    train_path = SOURCE / "train.json"
    if _sha256(train_path) != EXPECTED_TRAIN_SHA256:
        raise ValueError("Aug train data differs from audited v2 split")
    train = json.loads(train_path.read_text())
    dev = json.loads((SOURCE / "dev.json").read_text())
    bank = load_sra_logicbench19()
    memory = FrozenSkillBankMemory(bank)
    config, model_files, execution = load_batch_profile(
        ROOT / "configs/skillnet37_router_qwen3_embedding_batch_v1.json")
    config = replace(config, torch_version="2.11.0+cpu")
    router = LogicBenchEmbeddingRouter(memory, config, model_files=model_files,
        model_path=ROOT.parent.parent / "model/Qwen3-Embedding-0.6B",
        device="cpu", cache_path=args.router_cache,
        max_local_calls=4000, execution=execution)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        manifest = {"schema_version": "skillscope.logicbench_phase12_schedule.v1",
                    "bank_manifest_sha256": bank.manifest_sha256,
                    "aug_train_sha256": _sha256(train_path),
                    "aug_dev_sha256": _sha256(SOURCE / "dev.json"),
                    "router_protocol_hash": router.protocol_hash,
                    "question_only_router": True,
                    "questions_per_update": args.questions_per_update,
                    "updates": args.updates, "repeats": 8,
                    "seeds": {}}
        for seed in args.seeds:
            selected = select_questions(train, seed=seed,
                                        count=args.questions_per_update * args.updates)
            routed = _route(router, memory, selected)
            records = _parquet_rows(selected, routed, bank)
            # The separate dev file only satisfies VERL's nonempty dataset
            # contract; all utility labels come from the untouched Eval set.
            dev_rows = dev[:32]
            dev_records = _parquet_rows(dev_rows, _route(router, memory, dev_rows), bank)
            seed_dir = args.output / f"seed{seed}"
            seed_dir.mkdir()
            train_out, dev_out = seed_dir / "train.parquet", seed_dir / "dev.parquet"
            pd.DataFrame(records).to_parquet(train_out, index=False)
            pd.DataFrame(dev_records).to_parquet(dev_out, index=False)
            manifest["seeds"][str(seed)] = {
                "train_parquet_sha256": _sha256(train_out),
                "dev_parquet_sha256": _sha256(dev_out),
                "train_question_count": len(records),
                "distinct_train_contexts": len({r["context_id"] for r in records}),
                "routed_skill_counts": dict(sorted(Counter(
                    r["selected_skill_id"] for r in records).items())),
                "question_ids": [r["question_id"] for r in records],
            }
        eval_rows = load_logicbench_eval()
        eval_routes = _route(router, memory, eval_rows)
        with (args.output / "eval_routes.jsonl").open("w") as file:
            for row, routed in zip(eval_rows, eval_routes):
                file.write(json.dumps({"instance_id": row.instance_id,
                    "selected_skill_id": routed["selected_skill_id"],
                    "router_cache_key": routed["skill_router_api"]["cache_key"]},
                    sort_keys=True) + "\n")
        manifest["eval_routes_sha256"] = _sha256(args.output / "eval_routes.jsonl")
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "prepared", "skill_counts":
                          {k: v["routed_skill_counts"] for k, v in manifest["seeds"].items()}},
                         sort_keys=True))
    finally:
        router.close()


if __name__ == "__main__":
    main()
