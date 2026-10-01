"""Freeze LogicBench(Aug) BQA records and a context-disjoint train/dev split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

UPSTREAM_COMMIT = "c014153303c98de4d5f09d41c3a235cd869be5c8"
PATTERN_TO_SKILL = {
    "modus_ponens": "logicbench_000",
    "modus_tollens": "logicbench_001",
    "disjunctive_syllogism": "logicbench_002",
    "hypothetical_syllogism": "logicbench_003",
    "constructive_dillema": "logicbench_004",
    "destructive_dillema": "logicbench_005",
    "bidirectional_dilemma": "logicbench_006",
    "commutation": "logicbench_007",
    "material_implication": "logicbench_008",
    "existential_instantiation": "logicbench_009",
    "universal_instantiation": "logicbench_010",
    "reasoning_about_exceptions_1": "logicbench_011",
    "reasoning_about_exceptions_2": "logicbench_012",
    "reasoning_about_exceptions_3": "logicbench_013",
    "reasoning_about_priority": "logicbench_014",
    "default_reasoning_default": "logicbench_015",
    "default_reasoning_irr": "logicbench_016",
    "default_reasoning_open": "logicbench_017",
    "default_reasoning_several": "logicbench_018",
}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def near_eval_contexts(aug_contexts: list[str], eval_contexts: list[str],
                       *, threshold: float) -> set[str]:
    """Conservative lexical screen, fitted without answers or utility labels."""
    if not 0 < threshold <= 1:
        raise ValueError("Invalid near-duplicate threshold")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    candidates = sorted({_normalized(value) for value in aug_contexts})
    references = sorted({_normalized(value) for value in eval_contexts})
    if not candidates or not references:
        return set()
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit(
        candidates + references)
    similarity = cosine_similarity(vectorizer.transform(candidates),
                                   vectorizer.transform(references),
                                   dense_output=False).tocsr()
    return {candidate for index, candidate in enumerate(candidates)
            if similarity.getrow(index).nnz
            and similarity.getrow(index).data.max() >= threshold}


def build_aug_records(data_root: Path) -> tuple[list[dict], dict]:
    root = Path(data_root)
    aug_paths = sorted((root / "LogicBench(Aug)").glob("*/*/data_instances.json"))
    eval_paths = sorted((root / "LogicBench(Eval)").glob("*/*/*/data_instances.json"))
    if not aug_paths or not eval_paths:
        raise ValueError("Both LogicBench(Aug) and LogicBench(Eval) are required")
    eval_contexts = {
        _normalized(sample["context"])
        for path in eval_paths
        for sample in json.loads(path.read_text())["samples"]
    }
    all_aug_contexts = [sample["context"] for path in aug_paths
                        for sample in json.loads(path.read_text())["data_samples"]]
    near_contexts = near_eval_contexts(all_aug_contexts, list(eval_contexts),
                                       threshold=0.7)
    records, source_hashes = [], {}
    excluded_contexts, excluded_questions = set(), 0
    near_only_contexts, near_only_questions = set(), 0
    for path in aug_paths:
        category, pattern = path.parts[-3:-1]
        if pattern not in PATTERN_TO_SKILL:
            raise ValueError(f"Unknown Aug pattern: {pattern}")
        content = path.read_bytes()
        source_hashes[str(path.relative_to(root))] = hashlib.sha256(content).hexdigest()
        samples = json.loads(content)["data_samples"]
        for sample_index, sample in enumerate(samples):
            context = sample["context"].strip()
            normalized = _normalized(context)
            if normalized in eval_contexts:
                excluded_contexts.add(normalized)
                excluded_questions += len(sample["qa_pairs"])
                continue
            if normalized in near_contexts:
                near_only_contexts.add(normalized)
                near_only_questions += len(sample["qa_pairs"])
                continue
            for question_index, qa in enumerate(sample["qa_pairs"]):
                answer = qa["answer"].strip().casefold()
                if answer not in {"yes", "no"}:
                    raise ValueError("Aug contains nonbinary answer")
                question = qa["question"].strip()
                if not question or not context:
                    raise ValueError("Empty LogicBench question or context")
                source_id = f"{category}/{pattern}:{sample_index}:{question_index}"
                records.append({
                    "question_id": f"aug-{_digest(source_id)[:20]}",
                    "context_id": f"context-{_digest(normalized)[:20]}",
                    "question": context + "\n\n" + question,
                    "answer": answer,
                    "task_type": "BQA",
                    "source_pattern": f"{category}/{pattern}",
                    "skill_pattern_id": PATTERN_TO_SKILL[pattern],
                })
    if len({row["question_id"] for row in records}) != len(records):
        raise ValueError("Duplicate source question IDs")
    return records, {
        "source_file_sha256": source_hashes,
        "aug_pattern_files": len(aug_paths),
        "eval_pattern_files": len(eval_paths),
        "exact_eval_overlap_contexts_removed": len(excluded_contexts),
        "exact_eval_overlap_questions_removed": excluded_questions,
        "near_eval_contexts_removed": len(near_only_contexts),
        "near_eval_questions_removed": near_only_questions,
        "near_duplicate_protocol": "char_tfidf_3to5_cosine_threshold_0.7_context_only_v1",
        "remaining_questions": len(records),
        "source_pattern_counts": dict(sorted(Counter(row["source_pattern"] for row in records).items())),
        "mapped_skill_counts": dict(sorted(Counter(row["skill_pattern_id"] for row in records).items())),
    }


def split_context_groups(records: list[dict], *, seed: int,
                         dev_fraction: float) -> tuple[list[dict], list[dict]]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must lie inside (0,1)")
    contexts = sorted({row["context_id"] for row in records},
                      key=lambda value: _digest(f"{seed}:{value}"))
    if not contexts:
        raise ValueError("No training contexts remain")
    n_dev = min(len(contexts) - 1, max(1, round(len(contexts) * dev_fraction)))
    dev_ids = set(contexts[:n_dev])
    train = [row for row in records if row["context_id"] not in dev_ids]
    dev = [row for row in records if row["context_id"] in dev_ids]
    return train, dev


def _write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=404)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    args = parser.parse_args()
    source = args.upstream.resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Unexpected LogicBench upstream commit: {commit}")
    records, audit = build_aug_records(source / "data")
    if audit["aug_pattern_files"] != 25 or audit["eval_pattern_files"] != 50:
        raise ValueError("Unexpected LogicBench source inventory")
    train, dev = split_context_groups(records, seed=args.split_seed,
                                      dev_fraction=args.dev_fraction)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be new and empty")
    _write_new(output / "train.json", train)
    _write_new(output / "dev.json", dev)
    manifest = {
        "schema_version": "skillscope.logicbench_aug_split.v1",
        "upstream_commit": commit,
        "split_seed": args.split_seed,
        "dev_fraction": args.dev_fraction,
        "train_questions": len(train),
        "dev_questions": len(dev),
        "train_contexts": len({row["context_id"] for row in train}),
        "dev_contexts": len({row["context_id"] for row in dev}),
        "near_duplicate_eval_audit_complete": True,
        **audit,
        "train_sha256": hashlib.sha256((output / "train.json").read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256((output / "dev.json").read_bytes()).hexdigest(),
    }
    _write_new(output / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
