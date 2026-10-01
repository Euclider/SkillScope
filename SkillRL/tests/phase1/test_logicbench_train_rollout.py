import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from logicbench_phase12.rollout import single_step_rollout


class Tokenizer:
    def batch_decode(self, rows, skip_special_tokens=True):
        return ["yes" if int(row[0]) == 7 else "no" for row in rows]


class Actor:
    world_size = 2

    def generate_sequences(self, batch):
        assert len(batch) == 4
        prompts = batch.batch["input_ids"]
        response = torch.tensor([[7], [8], [7], [8]])
        return DataProto(TensorDict({"prompts": prompts,
            "responses": response,
            "input_ids": torch.cat((prompts, response), 1),
            "attention_mask": torch.ones(4, 3, dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2]] * 4)}, batch_size=[4]))


def test_single_step_rollout_groups_repeats_and_scores_exact_answers():
    batch = DataProto(TensorDict({"input_ids": torch.ones(2, 2, dtype=torch.long),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "position_ids": torch.tensor([[0, 1], [0, 1]])}, batch_size=[2]),
        non_tensor_batch={"question_id": np.array(["q1", "q2"], dtype=object),
            "selected_skill_id": np.array(["logicbench_001", "logicbench_002"], dtype=object),
            "answer": np.array(["yes", "no"], dtype=object),
            "task_type": np.array(["BQA", "BQA"], dtype=object),
            "data_source": np.array(["logicbench", "logicbench"], dtype=object)},
        meta_info={"eos_token_id": 9})
    result = single_step_rollout(batch, Actor(), Tokenizer(), repeats=2,
                                 run_id="test", update=1)
    assert len(result) == 4
    assert result.non_tensor_batch["episode_rewards"].tolist() == [1., 0., 0., 1.]
    assert result.non_tensor_batch["is_action_valid"].tolist() == [True] * 4
    assert result.non_tensor_batch["uid"][0] == result.non_tensor_batch["uid"][1]
    assert result.non_tensor_batch["uid"][1] != result.non_tensor_batch["uid"][2]
    assert len(set(result.non_tensor_batch["phase2_decision_id"])) == 4
    assert result.non_tensor_batch["phase2_metadata"][0]


def test_single_step_rollout_rejects_non_divisible_batch():
    batch = DataProto(TensorDict({"input_ids": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "position_ids": torch.tensor([[0, 1]])}, batch_size=[1]),
        non_tensor_batch={"question_id": np.array(["q"], dtype=object),
            "selected_skill_id": np.array(["logicbench_001"], dtype=object),
            "answer": np.array(["yes"], dtype=object),
            "task_type": np.array(["BQA"], dtype=object),
            "data_source": np.array(["logicbench"], dtype=object)})
    import pytest
    with pytest.raises(ValueError, match="divisible"):
        single_step_rollout(batch, Actor(), Tokenizer(), repeats=1,
                            run_id="test", update=1)
