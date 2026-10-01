import torch

from logicbench_phase12.fixed_state import score_recorded_tokens, summarize_token_signals


class ToyModel:
    def __call__(self, *, input_ids, attention_mask, use_cache, logits_to_keep):
        assert use_cache is False
        assert attention_mask.shape == input_ids.shape
        assert logits_to_keep == 2
        # Last two model positions predict the two recorded response tokens.
        logits = torch.tensor([[[0., 2., 0.], [0., 0., 2.]]])
        return type("Output", (), {"logits": logits})()


def test_recorded_token_scoring_uses_prompt_last_and_response_prefix():
    values = score_recorded_tokens(ToyModel(), [0, 1], [1, 2], device="cpu")
    assert values.shape == (2, 3)
    assert values.argmax(-1).tolist() == [1, 2]


def test_skill_summary_preserves_signed_and_magnitude_scores():
    tokens = [{"skill_id": "s", "question_id": "q1", "trajectory_id": "t1",
               "gate": True, "P_int": -2., "D_contribution": 2.,
               "delta_norm": 3., "delta_centered_norm": 2.,
               "advantage": 1., "chosen_delta": -1., "D_real": 0.5},
              {"skill_id": "s", "question_id": "q1", "trajectory_id": "t1",
               "gate": False, "P_int": 4., "D_contribution": 0.,
               "delta_norm": 1., "delta_centered_norm": 1.,
               "advantage": -1., "chosen_delta": 1., "D_real": -0.5}]
    result = summarize_token_signals(tokens)[0]
    assert result["D_signed_gate"] == 1.
    assert result["D_original"] == 1.
    assert result["M_delta_centered"] == 1.5
    assert result["n_questions"] == 1
    assert result["n_loss_tokens"] == 2
