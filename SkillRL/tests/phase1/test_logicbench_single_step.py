"""Single-decision LogicBench utility contracts."""

import pytest

from phase1.logicbench_single_step import (
    GenerationOutput,
    build_prompt,
    evaluate_single_step,
    extract_final_answer,
    load_logicbench_eval,
)


@pytest.mark.parametrize("task_type,output,expected", [
    ("BQA", "Reasoning.\nFinal answer: YES", "yes"),
    ("MCQA", "Reasoning.\nFinal answer: choice_3", "choice_3"),
    ("BQA", "Final answer: choice_3", None),
    ("MCQA", "Final answer: yes", None),
    ("BQA", "I think yes", None),
    ("MCQA", "Final answer: choice_5", None),
    ("BQA", "  NO  ", "no"),
    ("MCQA", "choice_2", "choice_2"),
])
def test_final_answer_parser(task_type, output, expected):
    assert extract_final_answer(output, task_type) == expected


def test_eval_load_verifies_pinned_file_and_schema():
    rows = load_logicbench_eval()
    assert len(rows) == 760
    assert rows[0].instance_id == "logicbench_00000"
    assert rows[0].task_type == "MCQA"
    assert rows[0].answer == "choice_4"


def test_prompt_demands_direct_single_label_without_reasoning():
    row = load_logicbench_eval()[0]
    prompt = build_prompt(row, None)
    assert "no explanation" in prompt.lower()
    assert "choice_1" in prompt and "choice_4" in prompt


def test_four_conditions_share_question_skill_and_seed_without_gold_in_prompt():
    row = load_logicbench_eval()[1]
    calls = []

    def generate(checkpoint, prompt, seed):
        calls.append((checkpoint, prompt, seed))
        # Old skill succeeds, old control fails; new skill fails, new control succeeds.
        correct = (checkpoint, "SECRET SKILL" in prompt) in (("old", True), ("new", False))
        return "Final answer: yes" if correct else "Final answer: no"

    result = evaluate_single_step(row, "logicbench_006", "SECRET SKILL", generate, seeds=(11,))
    assert len(calls) == 4
    assert {(x[0], x[2]) for x in calls} == {("old", 11), ("new", 11)}
    assert all(row.question in prompt for _, prompt, _ in calls)
    assert all("logicbench_006" not in prompt for _, prompt, _ in calls)
    assert all("Final answer: yes" not in prompt for _, prompt, _ in calls)
    assert sum("SECRET SKILL" in prompt for _, prompt, _ in calls) == 2
    assert result["selected_skill_id"] == "logicbench_006"
    assert result["m_old"] == 1.0
    assert result["m_new"] == -1.0
    assert result["delta_m"] == -2.0


def test_generation_cap_is_reported_separately_from_wrong_answer():
    row = load_logicbench_eval()[0]

    def generate(checkpoint, prompt, seed):
        del checkpoint, prompt, seed
        return GenerationOutput("Unfinished reasoning", 256, True)

    result = evaluate_single_step(row, "logicbench_001", "skill text", generate)
    assert all(record["reward"] == 0 for record in result["records"])
    assert all(record["format_valid"] is False for record in result["records"])
    assert all(record["hit_length_cap"] is True for record in result["records"])
