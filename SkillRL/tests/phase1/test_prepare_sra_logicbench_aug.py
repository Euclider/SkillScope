import json

from scripts.prepare_sra_logicbench_aug import (
    build_aug_records,
    near_eval_contexts,
    split_context_groups,
)


def test_aug_mapping_filters_eval_context_and_keeps_question_groups(tmp_path):
    source = tmp_path / "data"
    aug = source / "LogicBench(Aug)/first_order_logic/constructive_dillema"
    aug.mkdir(parents=True)
    (aug / "data_instances.json").write_text(json.dumps({
        "data_samples": [
            {"context": "Eval context", "qa_pairs": [{"question": "Q1", "answer": "yes"}]},
            {"context": "Fresh context", "qa_pairs": [
                {"question": "Q2", "answer": "yes"},
                {"question": "Q3", "answer": "no"},
            ]},
        ]
    }))
    evaluation = source / "LogicBench(Eval)/BQA/first_order_logic/constructive_dilemma"
    evaluation.mkdir(parents=True)
    (evaluation / "data_instances.json").write_text(json.dumps({
        "samples": [{"context": "Eval context", "qa_pairs": []}]
    }))
    records, audit = build_aug_records(source)
    assert [row["answer"] for row in records] == ["yes", "no"]
    assert {row["skill_pattern_id"] for row in records} == {"logicbench_004"}
    assert audit["exact_eval_overlap_questions_removed"] == 1
    train, dev = split_context_groups(records, seed=404, dev_fraction=0.5)
    assert bool(train) != bool(dev)


def test_context_split_is_deterministic_and_disjoint():
    records = [{"context_id": f"group-{i // 2}", "question_id": f"q-{i}"}
               for i in range(20)]
    first = split_context_groups(records, seed=404, dev_fraction=0.2)
    second = split_context_groups(records, seed=404, dev_fraction=0.2)
    assert first == second
    train, dev = first
    assert {row["context_id"] for row in train}.isdisjoint(
        {row["context_id"] for row in dev})


def test_near_eval_context_filter_catches_minor_rephrasing():
    aug = ["Cows are herbivores. Herbivores usually do not eat meat.",
           "The engine contains pistons and a cooling system."]
    evaluation = ["Cows are herbivores; herbivores usually do not eat meat."]
    matched = near_eval_contexts(aug, evaluation, threshold=0.7)
    assert len(matched) == 1
    assert aug[0].casefold() in matched
