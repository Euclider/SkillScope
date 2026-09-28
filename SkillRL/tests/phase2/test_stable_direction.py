import pytest
import torch

from phase2.direction import token_signals as legacy
from phase2.stable_direction import token_signals


def example(dtype=torch.float64):
    old = torch.tensor([[.5, .3, .2], [.2, .5, .3], [.7, .2, .1]], dtype=dtype).log()
    new = torch.tensor([[.6, .2, .2], [.3, .4, .3], [.6, .3, .1]], dtype=dtype).log()
    control = torch.tensor([[.7, .2, .1], [.4, .3, .3], [.5, .3, .2]], dtype=dtype).log()
    return old, new, old.clone(), control, torch.tensor([0, 1, 2]), torch.tensor([1., -2., 0.], dtype=dtype)


def test_matches_equations_and_original_in_well_conditioned_fp64_case():
    args = example()
    before = [x.clone() for x in args]
    got = token_signals(*args, include_legacy=True)
    expected = legacy(*args)
    for name in ('P_int', 'C_upd', 'C_upd_centered', 'D_contribution',
                 'D_ungated_contribution', 'd_norm', 'delta_norm'):
        torch.testing.assert_close(got[name], expected[name], atol=2e-14, rtol=2e-14)
    for name, value in expected.items():
        assert torch.equal(got['legacy_'+name], value)
    for a, b in zip(args, before):
        assert torch.equal(a, b)
    assert not got['direction_valid'][2]
    assert got['D_contribution'][2] == got['D_centered_contribution'][2] == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_peaked_probabilities_keep_tail_mass_and_zero_sum(dtype, sign):
    old = torch.tensor([[0., -20., -22.]], dtype=dtype)
    new = old + torch.tensor([[10., 11., 8.]], dtype=dtype)
    args = (old, new, old, old, torch.tensor([0]), torch.tensor([sign], dtype=dtype))
    got = token_signals(*args)
    assert got['direction_valid'].item()
    assert abs(got['direction_sum_residual'].item()) < 1e-23
    assert got['P_centering_abs_error'].item() < 1e-12
    other = old.double().softmax(-1)[0, 1:]
    d = torch.cat([other.sum().reshape(1), -other]) * sign
    want = (d*(new-old).double()[0]).sum()/(d.norm()+1e-12)
    torch.testing.assert_close(got['P_int'][0], want, rtol=1e-12, atol=1e-12)


def test_projection_is_invariant_to_common_shift_and_centering():
    args = list(example())
    raw = token_signals(*args)
    args[1] = args[1] + 123.
    args[3] = args[3] - 75.
    shifted = token_signals(*args)
    torch.testing.assert_close(raw['P_int'], shifted['P_int'], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(shifted['P_int'], shifted['P_int_centered'], atol=1e-12, rtol=1e-12)


def test_zero_interaction_and_degenerate_direction_are_explicit():
    old, new, _, _, ids, adv = example()
    got = token_signals(old, new, old, new, ids, adv)
    assert torch.equal(got['P_int'], torch.zeros(3, dtype=torch.float64))
    assert not got['gate'].any()
    tiny = torch.tensor([[0., -100., -100.]], dtype=torch.float64)
    got = token_signals(tiny, tiny+1, tiny, tiny, torch.tensor([0]), torch.tensor([1.]))
    assert not got['direction_valid'].item()
    assert not got['gate_centered'].item()


def test_gates_and_denominators_are_not_silently_changed():
    args = example()
    normal = token_signals(*args)
    gated = token_signals(*args, tau_delta=1e8)
    assert not gated['gate'].any() and not gated['gate_centered'].any()
    torch.testing.assert_close(gated['P_int'], normal['P_int'])
    assert not token_signals(*args, tau_c=2.)['gate'].any()
    assert not token_signals(*args, epsilon=100.)['direction_valid'].any()


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_rejected(bad):
    args = list(example()); args[0][0, 0] = bad
    with pytest.raises(ValueError):
        token_signals(*args)


def test_kl_js_remain_original_baselines():
    args = example(torch.float32)
    got, old = token_signals(*args), legacy(*args)
    for name in ('forward_kl_original', 'js_original'):
        assert torch.equal(got[name], old[name])


def test_many_normalized_fp64_rows_obey_centering_identity():
    generator = torch.Generator().manual_seed(715)
    vectors = [torch.randn(20, 101, generator=generator, dtype=torch.float64).log_softmax(-1) for _ in range(4)]
    result = token_signals(*vectors, torch.arange(20), torch.linspace(-2, 2, 20))
    torch.testing.assert_close(result['P_int'], result['P_int_centered'], rtol=1e-12, atol=1e-12)
    assert result['normalized_probability_mass_error'].abs().max() < 1e-14
