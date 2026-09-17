import torch

from phase2.direction import token_signals


def test_signed_projection_matches_explicit_formula_and_keeps_zero_rows():
    old = torch.tensor([[.6, .3, .1], [.2, .5, .3]], dtype=torch.float64).log()
    new = torch.tensor([[.5, .4, .1], [.3, .4, .3]], dtype=torch.float64).log()
    a = torch.tensor([0, 1])
    advantage = torch.tensor([1., 0.], dtype=torch.float64)
    r = token_signals(old, new, old, old, a, advantage)
    d = torch.tensor([1., 0., 0.])-old[0].exp()
    expected = (d*(new[0]-old[0])).sum()/(d.norm()+1e-12)
    torch.testing.assert_close(r["P_int"][0], expected)
    assert r["P_int"][0] < 0
    assert not r["direction_valid"][1]
    assert r["D_contribution"][1] == 0
    assert r["P_int"].shape == (2,)


def test_common_policy_shift_has_zero_interaction():
    old = torch.tensor([[.7, .2, .1]], dtype=torch.float64).log()
    new = torch.tensor([[.6, .25, .15]], dtype=torch.float64).log()
    r = token_signals(old, new, old, new, torch.tensor([0]), torch.tensor([1.]))
    assert r["delta_norm"].item() == 0
    assert r["P_int"].item() == 0


def test_reward_opposition_with_positive_update_fidelity():
    old = torch.tensor([[.5, .5]], dtype=torch.float64).log()
    new_o = torch.tensor([[.6, .4]], dtype=torch.float64).log()
    new_p = torch.tensor([[.8, .2]], dtype=torch.float64).log()
    r = token_signals(old, new_o, old, new_p, torch.tensor([0]), torch.tensor([1.]))
    assert r["C_upd"].item() > 0
    assert r["P_int"].item() < 0
    assert r["D_contribution"].item() > 0


def test_capture_preserves_live_probabilities_and_input(tmp_path):
    from phase2.capture import capture_old_logits
    logits = torch.tensor([[[1., 2., 0.], [2., 1., 0.], [0., 0., 0.]]])
    before = logits.clone()
    responses = torch.tensor([[1, 0, 2]])
    chosen = logits.log_softmax(-1).gather(-1, responses.unsqueeze(-1)).squeeze(-1)
    batch = {"responses": responses, "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
             "phase2_row_index": torch.tensor([9])}
    capture_old_logits(logits, chosen, batch, {"root":str(tmp_path),"update":31})
    x = torch.load(tmp_path/"old_logprobs/u0031/row-000009.pt",weights_only=False)
    assert x["token_ids"].tolist() == [1, 0]
    torch.testing.assert_close(x["log_probs"], before[0,:2].log_softmax(-1))
    torch.testing.assert_close(logits, before)
