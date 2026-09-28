"""Reward-calibrated realized-update secant readout; never reads utility labels.

This is NOT a per-action causal reward or the PPO/GRPO optimizer gradient.
The signed local frozen-advantage log-likelihood change calibrates a rank-one
reference built from the observed policy update. Centering is over vocabulary.
"""
from __future__ import annotations

import numpy as np

from .reward_variants import AGGREGATIONS, EPS, aggregation_weights

VERSION = 'realized_reward_secant_v1'
KEYS = ['decision_id', 'control', 'response_token_offset']
FIELDS = ('real_u_centered_norm_sq', 'real_u_delta_centered_dot',
          'real_projection_coefficient', 'real_q', 'real_D', 'real_A1',
          'real_absA', 'real_projection_only', 'real_small_u')


def vector_signals(old_o, new_o, old_control, new_control, actions, advantages,
                   *, epsilon=EPS, chunk_size=16):
    """Bound FP64 scratch, retaining every vocabulary coordinate and loss token."""
    import torch
    if epsilon <= 0 or not np.isfinite(epsilon) or chunk_size < 1:
        raise ValueError('Positive finite epsilon and chunk size required')
    if not len(actions) or actions.dtype not in (torch.int32, torch.int64):
        raise ValueError('Nonempty integer action IDs required')
    if not (old_o.shape == new_o.shape == old_control.shape == new_control.shape):
        raise ValueError('Four aligned log-probability matrices required')
    if old_o.ndim != 2 or len(actions) != len(old_o) or len(advantages) != len(old_o):
        raise ValueError('Token identity dimensions differ')
    chunks = []
    for start in range(0, len(actions), chunk_size):
        sl = slice(start, start+chunk_size)
        old, new, oc, nc = [x[sl].to(device=old_o.device, dtype=torch.float64)
                           for x in (old_o, new_o, old_control, new_control)]
        a = advantages[sl].to(device=old_o.device, dtype=torch.float64)
        ids = actions[sl].to(device=old_o.device, dtype=torch.int64)
        u = new-old
        delta = u-(nc-oc)
        v = u-u.mean(-1, keepdim=True)
        xi = delta-delta.mean(-1, keepdim=True)
        norm_sq = v.square().sum(-1)
        dot = (v*xi).sum(-1)
        coefficient = dot/(norm_sq+epsilon)
        chosen = u.gather(1, ids[:, None]).squeeze(-1)
        q = a*chosen  # RAW chosen-action log likelihood, not centered chosen u.
        fields = dict(zip(FIELDS, (norm_sq, dot, coefficient, q,
            -q*coefficient, -chosen*coefficient, -a.abs()*chosen*coefficient,
            -coefficient, norm_sq <= epsilon)))
        if not all(torch.isfinite(x).all() for x in fields.values()):
            raise ValueError('Non-finite realized readout; no automatic clipping')
        chunks.append({key: value.detach().cpu() for key, value in fields.items()})
    return {key: torch.cat([chunk[key] for chunk in chunks]) for key in FIELDS}


def registry():
    recipes = (
        ('D_real', 'reward', 'real_D', '-A*u(a)*dot(Hu,Hdelta)/(norm(Hu)^2+epsilon)', True),
        ('D_real_A1', 'reward_free', 'real_A1', '-u(a)*dot(Hu,Hdelta)/(norm(Hu)^2+epsilon)', False),
        ('D_real_absA', 'reward_sign_removed', 'real_absA', '-abs(A)*u(a)*dot(Hu,Hdelta)/(norm(Hu)^2+epsilon)', False),
        ('D_real_projection_only', 'geometry_only', 'real_projection_only', '-dot(Hu,Hdelta)/(norm(Hu)^2+epsilon)', False),
    )
    return [{'name': name, 'mode': mode, 'column': column, 'formula': formula,
        'score': f'{name}::{agg}::{mode}', 'aggregation': agg,
        'family': VERSION, 'signed': signed, 'geometry': True,
        'reward_removed_scope': 'All original rows including A=0; no advantage-based support mask'}
        for agg in AGGREGATIONS for name, mode, column, formula, signed in recipes]


def constant_safe_reduce(weights, matrix):
    """Keep exact constants exact; do not create spurious last-bit rankings."""
    matrix = np.asarray(matrix, dtype=float)
    values = weights @ matrix
    lo, hi = matrix.min(axis=0), matrix.max(axis=0)
    return np.where(lo == hi, lo, values)


def aggregate_scores(tokens):
    import pandas as pd
    records = []
    if tokens.duplicated(KEYS).any():
        raise ValueError('Duplicate token/control rows')
    for control in ('placebo', 'null'):
        part = tokens[tokens.control == control]
        for phase in ('all', 'initial', 'early', 'middle', 'late'):
            sub = part if phase == 'all' else part[part.phase == phase]
            for (context, skill), group in sub.groupby(['context_id', 'skill_id'], sort=True):
                for agg in AGGREGATIONS:
                    methods = [m for m in registry() if m['aggregation'] == agg]
                    matrix = group[[m['column'] for m in methods]].to_numpy(float)
                    vals = constant_safe_reduce(aggregation_weights(group, agg), matrix)
                    for method, value in zip(methods, vals):
                        records.append({'control': control, 'context_id': context, 'skill_id': skill,
                            'phase': phase, 'score': method['score'], 'value': float(value),
                            'token_count': len(group), 'decision_count': group.decision_id.nunique(),
                            'trajectory_count': group.trajectory_id.nunique(),
                            'game_count': group.game_id.nunique()})
    return pd.DataFrame(records)


def assert_same_scalars(fresh, recorded):
    """New geometry must reproduce old stable readouts on the same inputs."""
    import pandas as pd
    fields = KEYS + ['skill_id', 'trajectory_id', 'game_id', 'action_token_id',
        'advantage', 'd_norm', 'u_original_norm', 'u_control_norm', 'delta_norm',
        'delta_centered_norm', 'P_int', 'C_upd', 'C_upd_centered',
        'chosen_u_original', 'chosen_delta', 'direction_valid', 'D_contribution']
    a, b = [x.sort_values(KEYS).reset_index(drop=True)[fields] for x in (fresh, recorded)]
    pd.testing.assert_frame_equal(a, b, check_exact=True, check_dtype=False)

