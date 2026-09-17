#!/usr/bin/env python3
"""Create a tokenizer-length-matched, task-irrelevant Skill payload."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_system.memory import SkillsOnlyMemory
from phase1.archive import atomic_write_json, sha256_file, stable_hash
from phase1.first_invocation import make_length_matched_placebo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--skill-bank", type=Path, required=True)
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    memory = SkillsOnlyMemory(
        str(args.skill_bank), retrieval_mode="template", task_specific_top_k=None
    )
    retrieved = memory.retrieve(args.task_description, top_k=12)
    selected = None
    selected_kind = None
    for kind in ("general_skills", "task_specific_skills", "mistakes_to_avoid"):
        for item in retrieved.get(kind, []):
            if item.get("skill_id") == args.skill_id:
                selected = item
                selected_kind = kind
                break
        if selected is not None:
            break
    if selected is None or selected_kind is None:
        raise KeyError(f"Skill {args.skill_id!r} is absent from retrieved candidates")
    payload = {
        "general_skills": [selected] if selected_kind == "general_skills" else [],
        "task_specific_skills": (
            [selected] if selected_kind == "task_specific_skills" else []
        ),
        "mistakes_to_avoid": (
            [selected] if selected_kind == "mistakes_to_avoid" else []
        ),
        "task_type": retrieved["task_type"],
        "retrieval_mode": retrieved["retrieval_mode"],
    }
    original_text = memory.format_for_prompt(payload)
    placebo = make_length_matched_placebo(tokenizer, original_text)
    atomic_write_json(args.output, {
        **placebo,
        "skill_id": args.skill_id,
        "model_tokenizer": args.model,
        "skill_bank": str(args.skill_bank),
        "skill_bank_hash": sha256_file(args.skill_bank),
        "original_text": original_text,
        "original_text_hash": stable_hash(original_text),
    })


if __name__ == "__main__":
    main()
