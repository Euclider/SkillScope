import pytest
import torch
from verl.trainer.ppo.core_algos import compute_policy_loss, kl_penalty


@pytest.mark.parametrize('advantage', [-2., 0., 2.])
def test_saturated_dual_clip_has_finite_exact_gradient(advantage):
    lp = torch.tensor([[0.]], requires_grad=True)
    loss, *_ = compute_policy_loss(torch.tensor([[-100.]]), lp,
        torch.tensor([[advantage]]), torch.ones(1, 1), cliprange=.2, clip_ratio_c=3.)
    loss.backward()
    expected = {-2.: 6., 0.: 0., 2.: -2.4}[advantage]
    assert loss.item() == pytest.approx(expected)
    assert torch.isfinite(lp.grad).all()
    assert lp.grad.item() == 0.


@pytest.mark.parametrize('negative_logprob', [-101., -1e10])
def test_saturated_k3_has_finite_zero_gradient(negative_logprob):
    lp = torch.tensor([[negative_logprob]], requires_grad=True)
    loss = kl_penalty(lp, torch.tensor([[-1.]]), 'low_var_kl')
    loss.sum().backward()
    assert loss.item() == 10.
    assert torch.isfinite(lp.grad).all()
    assert lp.grad.item() == 0.


def test_normal_range_loss_and_gradient_match_direct_equations_exactly():
    x = torch.tensor([[-2., -.3, 0., .15, 2., 15.]], requires_grad=True)
    ref = torch.zeros_like(x)
    expected = (torch.exp(-x) + x - 1).clamp(-10, 10)
    actual = kl_penalty(x, ref, 'low_var_kl')
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
                       torch.autograd.grad(expected.sum(), x)[0])


@pytest.mark.parametrize('sign', [-1., 0., 1.])
def test_dual_clip_preserves_finite_original_objective_and_gradient(sign):
    x = torch.tensor([[-70., -3., -.3, 0., .1, 1., 25., 69.]], requires_grad=True)
    a = torch.full_like(x, sign * 2)
    ratio = x.exp()
    clipped = torch.maximum(-a * ratio, -a * ratio.clamp(.8, 1.2))
    expected = torch.where(a < 0, torch.minimum(-a * 3, clipped), clipped).mean()
    actual, *_ = compute_policy_loss(torch.zeros_like(x), x, a, torch.ones_like(x),
                                     cliprange=.2, clip_ratio_c=3.)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, x, retain_graph=True)[0],
                       torch.autograd.grad(expected, x)[0])
