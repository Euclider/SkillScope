"""One-question LogicBench evaluation with fixed skill/control interventions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

EVAL_PATH = Path(__file__).resolve().parents[1] / "data/logicbench/sra19/eval.json"
EVAL_SHA256 = "af5055caac041ea08cee47622f5d922b47b2ba0a8e3a60b87349599eeff1bdfe"
PROMPT_VERSION = "logicbench-direct-label-v2"
_LABELS = {"BQA": frozenset({"yes", "no"}),
           "MCQA": frozenset(f"choice_{i}" for i in range(1, 5))}
_FINAL_LINE = re.compile(r"(?im)^\s*final answer:\s*(\S+)\s*$")


@dataclass(frozen=True)
class LogicBenchQuestion:
    instance_id: str
    question: str
    task_type: str
    answer: str
    gold_skill_id: str


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    generated_tokens: int
    hit_length_cap: bool
    prompt_tokens: int | None = None


def load_logicbench_eval(path: Path = EVAL_PATH) -> tuple[LogicBenchQuestion, ...]:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != EVAL_SHA256:
        raise ValueError("LogicBench Eval file differs from pinned snapshot")
    data = json.loads(raw)
    if not isinstance(data, list) or len(data) != 760:
        raise ValueError("LogicBench Eval inventory mismatch")
    rows = []
    for item in data:
        task_type = item["eval_data"]["task_type"]
        answer = item["eval_data"]["answer"]
        annotations = item["skill_annotations"]
        if (task_type not in _LABELS or answer not in _LABELS[task_type]
                or len(annotations) != 1 or not isinstance(item["question"], str)
                or not item["question"].strip()):
            raise ValueError("Malformed LogicBench Eval instance")
        rows.append(LogicBenchQuestion(item["instance_id"], item["question"],
                                       task_type, answer, annotations[0]))
    return tuple(rows)


def build_prompt(row: LogicBenchQuestion, skill_payload: str | None) -> str:
    if row.task_type not in _LABELS:
        raise ValueError("Unsupported LogicBench task type")
    labels = ", ".join(sorted(_LABELS[row.task_type]))
    prompt = "Solve the following logic question silently.\n\n"
    if skill_payload is not None:
        if not skill_payload.strip():
            raise ValueError("Empty skill payload")
        prompt += f"Skill guidance:\n{skill_payload}\n\n"
    return (prompt + f"Question:\n{row.question}\n\n"
            f"Output exactly one label from this list: {labels}. "
            "No explanation, no preamble, and no punctuation.\n")


def extract_final_answer(output: str, task_type: str) -> str | None:
    if task_type not in _LABELS or not isinstance(output, str):
        raise ValueError("Unsupported output or task type")
    direct = output.strip().lower()
    if direct in _LABELS[task_type]:
        return direct
    matches = _FINAL_LINE.findall(output)
    if len(matches) != 1:
        return None
    label = matches[0].lower()
    return label if label in _LABELS[task_type] else None


def evaluate_single_step(
    row: LogicBenchQuestion,
    selected_skill_id: str,
    skill_payload: str,
    generate: Callable[[str, str, int], str | GenerationOutput],
    *,
    seeds: Iterable[int] = (0,),
) -> dict:
    """Call one policy continuation per old/new × skill/control × seed.

    Selection occurs before this function; Eval answers are read only for reward.
    """
    seeds = tuple(seeds)
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("At least one integer seed is required")
    if not selected_skill_id or not skill_payload:
        raise ValueError("A frozen selected skill and payload are required")
    records = []
    for checkpoint in ("old", "new"):
        for condition in ("skill", "control"):
            prompt = build_prompt(row, skill_payload if condition == "skill" else None)
            for seed in seeds:
                generated = generate(checkpoint, prompt, seed)
                output = generated.text if isinstance(generated, GenerationOutput) else generated
                parsed = extract_final_answer(output, row.task_type)
                records.append({"checkpoint": checkpoint, "condition": condition,
                                "seed": seed, "response_text": output, "parsed_answer": parsed,
                                "format_valid": parsed is not None,
                                "generated_tokens": (generated.generated_tokens
                                                     if isinstance(generated, GenerationOutput) else None),
                                "prompt_tokens": (generated.prompt_tokens
                                                  if isinstance(generated, GenerationOutput) else None),
                                "hit_length_cap": (generated.hit_length_cap
                                                   if isinstance(generated, GenerationOutput) else None),
                                "reward": int(parsed == row.answer)})
    means = {(checkpoint, condition): sum(r["reward"] for r in records
             if r["checkpoint"] == checkpoint and r["condition"] == condition) / len(seeds)
             for checkpoint in ("old", "new") for condition in ("skill", "control")}
    m_old = means["old", "skill"] - means["old", "control"]
    m_new = means["new", "skill"] - means["new", "control"]
    return {"instance_id": row.instance_id, "selected_skill_id": selected_skill_id,
            "m_old": m_old, "m_new": m_new, "delta_m": m_new - m_old,
            "records": records}
