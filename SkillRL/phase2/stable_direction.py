"""Numerically stable v1 of the unchanged full-vocabulary C/P/D equations.

The source-frozen ``direction.token_signals`` remains the legacy comparator.
Only readout arithmetic is FP64; neither model execution nor stored logits are
silently promoted to a different model. Thresholds and validity definitions
remain caller-controlled and identical to the original protocol.
"""
from __future__ import annotations

import math
import torch

from .direction import token_signals as legacy_token_signals

VERSION = 'fp64_zero_sum_readout_v1'


def _fp64_rows(old_o, new_o, old_control, new_control, actions, advantages,
               *, tau_delta, tau_c, epsilon):
    if actions.dtype not in (torch.int32, torch.int64):
        raise ValueError('Integer action IDs required')
    device = old_o.device
    action = actions.to(device=device, dtype=torch.int64)
    advantage = advantages.to(device=device, dtype=torch.float64)
    old, new, oc, nc = (x.to(device=device, dtype=torch.float64)
                        for x in (old_o, new_o, old_control, new_control))
    # Normalize the OLD probability used by the reward direction. Preserve the
    # four logged action-preference vectors themselves for u and delta.
    raw_mass = old.exp().sum(-1)
    p = torch.softmax(old, dim=-1)
    other = p.clone()
    other.scatter_(1, action[:, None], 0.)
    mass_other = other.sum(-1)
    d = -advantage[:, None] * other
    # 1-p[a] loses all tail mass when p[a] rounds to 1. Summing the other
    # probabilities is the same mathematical quantity without cancellation.
    d.scatter_(1, action[:, None], (advantage * mass_other)[:, None])
    dnorm = d.square().sum(-1).sqrt()
    u = new-old
    base_u = nc-oc
    delta = u-base_u
    uc = u-u.mean(-1, keepdim=True)
    dc = delta-delta.mean(-1, keepdim=True)

    def dot(vector):
        # Equivalent to d.T @ vector but cancels the common component before
        # multiplying. This is not centering across skills or across examples.
        relative = vector.gather(1, action[:, None])-vector
        return (other * relative).sum(-1) * advantage

    unorm = u.square().sum(-1).sqrt()
    ucn = uc.square().sum(-1).sqrt()
    delta_norm = delta.square().sum(-1).sqrt()
    dcn = dc.square().sum(-1).sqrt()
    valid = dnorm > epsilon
    fidelity_valid = valid & (unorm > epsilon)
    centered_valid = valid & (ucn > epsilon)
    numerator = dot(u)
    fidelity = torch.where(fidelity_valid,
        numerator/(dnorm*unorm).clamp_min(epsilon), torch.zeros_like(dnorm))
    centered_fidelity = torch.where(centered_valid,
        numerator/(dnorm*ucn).clamp_min(epsilon), torch.zeros_like(dnorm))
    projection = dot(delta)/(dnorm+epsilon)
    centered_projection = dot(dc)/(dnorm+epsilon)
    gate = fidelity_valid & (fidelity >= tau_c) & (delta_norm >= tau_delta)
    centered_gate = centered_valid & (centered_fidelity >= tau_c) & (dcn >= tau_delta)
    revised = {
        'd_norm': dnorm, 'u_original_norm': unorm,
        'u_control_norm': base_u.square().sum(-1).sqrt(),
        'delta_norm': delta_norm, 'delta_centered_norm': dcn,
        'C_upd': fidelity, 'C_upd_centered': centered_fidelity,
        'direction_valid': valid, 'fidelity_valid': fidelity_valid,
        'fidelity_valid_centered': centered_valid,
        'P_int': projection, 'P_int_centered': centered_projection,
        'gate': gate, 'gate_centered': centered_gate,
        'D_contribution': gate*torch.relu(-projection),
        # Use the same stable P for both gates; the independent centered-P
        # computation is retained as an identity check, not a new predictor.
        'D_centered_contribution': centered_gate*torch.relu(-projection),
        'D_ungated_contribution': valid*torch.relu(-projection),
        'chosen_u_original': u.gather(1, action[:, None]).squeeze(-1),
        'chosen_delta': delta.gather(1, action[:, None]).squeeze(-1),
        'advantage': advantage,
        'old_probability_mass_error': raw_mass-1.,
        'normalized_probability_mass_error': p.sum(-1)-1.,
        'direction_sum_residual': d.sum(-1),
        'P_centering_abs_error': (projection-centered_projection).abs(),
        'C_numerator_centering_abs_error': (numerator-dot(uc)).abs(),
        'P_direct_dot_abs_error': ((d*delta).sum(-1)/(dnorm+epsilon)-projection).abs(),
    }
    if not all(torch.isfinite(x).all() for x in revised.values()):
        raise ValueError('Non-finite stable readout; preserve evidence and stop')
    return {k: v.detach().cpu() for k, v in revised.items()}


def token_signals(old_o, new_o, old_control, new_control, actions, advantages,
                  *, tau_delta=1e-8, tau_c=0., epsilon=1e-12,
                  include_legacy=False):
    if not all(math.isfinite(x) for x in (tau_delta, tau_c, epsilon)) or epsilon <= 0 or tau_delta < 0:
        raise ValueError('Finite gates and positive epsilon required')
    # Keep the legacy call's original row shape and reduction implementation.
    original = legacy_token_signals(old_o, new_o, old_control, new_control,
                                   actions, advantages, tau_delta=tau_delta,
                                   tau_c=tau_c, epsilon=epsilon)
    result = dict(original)
    chunks = []
    # Pure per-token arithmetic can be streamed without reducing vocabulary
    # support. Bound FP64 scratch even for retained 512-token failed actions.
    for first in range(0, len(actions), 16):
        sl = slice(first, first+16)
        chunks.append(_fp64_rows(old_o[sl], new_o[sl], old_control[sl], new_control[sl],
            actions[sl], advantages[sl], tau_delta=tau_delta, tau_c=tau_c, epsilon=epsilon))
    if not chunks:
        raise ValueError('At least one actual loss token is required')
    result.update({k: torch.cat([r[k] for r in chunks]) for k in chunks[0]})
    if include_legacy:
        result.update({'legacy_'+k: v for k, v in original.items()})
    return result
