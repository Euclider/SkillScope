"""Full-support equations from design sections 5 and 6 (W = identity)."""
from __future__ import annotations

import torch


def token_signals(old_o, new_o, old_control, new_control, actions, advantages,
                  *, tau_delta=1e-8, tau_c=0., epsilon=1e-12):
    shapes = {tuple(x.shape) for x in (old_o, new_o, old_control, new_control)}
    if len(shapes) != 1 or old_o.ndim != 2:
        raise ValueError("Four aligned token-by-vocabulary distributions are required")
    if actions.shape != advantages.shape or actions.shape != old_o.shape[:1]:
        raise ValueError("Actions and advantages must align to every decision token")
    if not all(torch.isfinite(x).all() for x in (old_o, new_o, old_control, new_control, advantages)):
        raise ValueError("Non-finite input to reward-directed signal")
    advantages = advantages.to(device=old_o.device, dtype=old_o.dtype)
    actions = actions.to(device=old_o.device, dtype=torch.int64)
    p = old_o.exp()
    d = -p * advantages[:, None]
    d.scatter_add_(1, actions[:, None], advantages[:, None])
    u = new_o - old_o
    base_u = new_control - old_control
    delta = u - base_u
    dnorm = d.square().sum(-1, dtype=torch.float64).sqrt()
    unorm = u.square().sum(-1, dtype=torch.float64).sqrt()
    delta_norm = delta.square().sum(-1, dtype=torch.float64).sqrt()
    dot_u = (d*u).sum(-1, dtype=torch.float64)
    dot_int = (d*delta).sum(-1, dtype=torch.float64)
    valid = dnorm > epsilon
    fidelity_valid = valid & (unorm > epsilon)
    fidelity = torch.where(fidelity_valid, dot_u/(dnorm*unorm).clamp_min(epsilon), torch.zeros_like(dnorm))
    projection = dot_int/(dnorm+epsilon)
    gate = fidelity_valid & (fidelity >= tau_c) & (delta_norm >= tau_delta)
    uc = u-u.mean(-1, keepdim=True)
    dc = delta-delta.mean(-1, keepdim=True)
    ucn = uc.square().sum(-1, dtype=torch.float64).sqrt()
    centered_fidelity = torch.where(valid & (ucn > epsilon), (d*uc).sum(-1, dtype=torch.float64)/(dnorm*ucn).clamp_min(epsilon), torch.zeros_like(dnorm))
    m = torch.logaddexp(old_o, new_o) - torch.log(torch.tensor(2., device=old_o.device))
    result = {
        "d_norm": dnorm, "u_original_norm": unorm,
        "u_control_norm": base_u.square().sum(-1, dtype=torch.float64).sqrt(),
        "delta_norm": delta_norm,
        "delta_centered_norm": dc.square().sum(-1, dtype=torch.float64).sqrt(),
        "C_upd": fidelity, "C_upd_centered": centered_fidelity,
        "direction_valid": valid, "fidelity_valid": fidelity_valid,
        "P_int": projection, "gate": gate,
        "D_contribution": gate*torch.relu(-projection),
        "D_ungated_contribution": valid*torch.relu(-projection),
        "forward_kl_original": (p*(old_o-new_o)).sum(-1, dtype=torch.float64),
        "js_original": .5*((p*(old_o-m)).sum(-1, dtype=torch.float64)+(new_o.exp()*(new_o-m)).sum(-1, dtype=torch.float64)),
        "chosen_u_original": u.gather(1, actions[:, None]).squeeze(-1),
        "chosen_delta": delta.gather(1, actions[:, None]).squeeze(-1),
        "advantage": advantages,
    }
    return {key: value.detach().cpu() for key, value in result.items()}
