import json

import pytest

from phase3.common import ProtocolError
from phase3.logicbench_run import validate_setting, summarize_test, main, prompt_validator
from phase3.logicbench import initial_bank, SkillPromptBudgetError
from phase1.logicbench_single_step import LogicBenchQuestion


def setting():
    with open("configs/phase3_logicbench_draft_v1.json") as f:
        return json.load(f)


def test_draft_refuses_execute_before_loading_models_or_api(tmp_path):
    with pytest.raises(ProtocolError, match="confirmed"):
        main(["--setting", "configs/phase3_logicbench_draft_v1.json", "--seed", "404",
              "--method", "D_signed_gate", "--output", str(tmp_path), "--execute"])
    assert not list(tmp_path.iterdir())


def test_invalid_generation_or_seed_configuration_rejected():
    config = setting()
    config["eval_seeds"] = [0, 0]
    with pytest.raises(ProtocolError):
        validate_setting(config)
    config = setting()
    config["max_new_tokens"] = 0
    with pytest.raises(ProtocolError):
        validate_setting(config)


def test_paired_test_report_has_type_specific_improvement_and_regressions():
    before = [{"game_id": "a", "task_type": "BQA", "eval_seed": 0, "success": False},
              {"game_id": "b", "task_type": "MCQA", "eval_seed": 0, "success": True}]
    after = [{**before[0], "success": True}, {**before[1], "success": False}]
    report = summarize_test(before, after)
    assert report["overall"]["delta_success_rate"] == 0
    assert report["BQA"]["delta_success_rate"] == 1
    assert report["MCQA"]["delta_success_rate"] == -1
    assert report["overall"]["repairs"] == report["overall"]["regressions"] == 1


def test_edit_length_validator_checks_complete_prompt_not_only_skill():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": text.split()}

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

    config = setting()
    config["max_prompt_tokens"] = 50
    config["max_edited_skill_tokens"] = 100000
    validate = prompt_validator(Tokenizer(), [LogicBenchQuestion("q", "word " * 60, "BQA", "", "")], config)
    with pytest.raises(SkillPromptBudgetError, match="prompt"):
        validate(initial_bank("test").skills[0])
