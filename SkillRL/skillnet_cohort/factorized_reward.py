"""One fixed centered-magnitude x reward-orientation recipe, without gold input.

This is an exploratory scalar analysis, not a change to the production readout.
The orientation uses normalized log probabilities at the sampled action (NOT
vocabulary-centered action coordinates). H is applied to the magnitude only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .reward_variants import AGGREGATIONS, EPS, aggregation_weights

VERSION = 'factorized_centered_reward_v1'
GROUP_KEYS = ['control', 'context_id', 'phase', 'skill_id', 'aggregation']


def registry():
    recipes = [
        ('D_factor', '-B * R', True, 'reward'),
        ('D_orientation', '-R', True, 'reward'),
        ('D_factor_A1', '-B * R(A=1 on ALL rows)', False, 'reward_free'),
        ('D_orientation_A1', '-R(A=1 on ALL rows)', False, 'reward_free'),
    ]
    return [{'score': f'{name}::{agg}::{mode}', 'name': name,
             'aggregation': agg, 'mode': mode, 'signed': signed,
             'family': 'factorized_centered_reward', 'formula': formula,
             'geometry': False}
            for agg in AGGREGATIONS for name, formula, signed, mode in recipes]


def constant_safe_mean(values, weights):
    """Keep exact constants exact; no ulp-based ranking of constant controls."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if (values.ndim != 1 or values.shape != weights.shape or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError('Finite nonempty values and normalized nonnegative weights required')
    active = values[weights > 0]
    return float(active[0] if np.all(active == active[0]) else weights @ values)


def factors(magnitude, chosen_delta, advantage, weights):
    magnitude, chosen_delta, advantage = (np.asarray(x, float)
                                         for x in (magnitude, chosen_delta, advantage))
    if magnitude.shape != chosen_delta.shape or magnitude.shape != advantage.shape:
        raise ValueError('Factor inputs must have identical shapes')
    if (magnitude < 0).any() or not np.isfinite(advantage).all():
        raise ValueError('Nonnegative magnitude and finite advantage required')
    b = advantage * chosen_delta
    B = constant_safe_mean(magnitude, weights)
    numerator = constant_safe_mean(b, weights)
    absolute = constant_safe_mean(np.abs(b), weights)
    R = numerator / (absolute + EPS)
    b1 = constant_safe_mean(chosen_delta, weights)
    abs1 = constant_safe_mean(np.abs(chosen_delta), weights)
    R1 = b1 / (abs1 + EPS)
    if abs(R) > 1.+1e-12 or abs(R1) > 1.+1e-12:
        raise ValueError('Orientation outside its analytic range')
    return {'B': B, 'b_mean': numerator, 'b_absolute_mean': absolute, 'R': R,
            'A1_b_mean': b1, 'A1_b_absolute_mean': abs1, 'R_A1': R1,
            'D_factor': -B*R, 'D_orientation': -R,
            'D_factor_A1': -B*R1, 'D_orientation_A1': -R1}


def aggregate_scores(tokens):
    """All rows, including zero advantage, enter the fixed denominators."""
    scores, components = [], []
    for control in ('placebo', 'null'):
        part = tokens[tokens.control == control]
        for phase in ('all', 'initial', 'early', 'middle', 'late'):
            subset = part if phase == 'all' else part[part.phase == phase]
            for (context, skill), group in subset.groupby(['context_id', 'skill_id'], sort=True):
                counts = {'token_count': len(group), 'decision_count': group.decision_id.nunique(),
                          'trajectory_count': group.trajectory_id.nunique(), 'game_count': group.game_id.nunique(),
                          'nonzero_advantage_tokens': int(group.advantage.ne(0).sum())}
                for agg in AGGREGATIONS:
                    f = factors(group.delta_centered_norm, group.chosen_delta, group.advantage,
                                aggregation_weights(group, agg))
                    base = {'control': control, 'context_id': context, 'phase': phase, 'skill_id': skill}
                    components.append({**base, 'aggregation': agg, **counts, **f,
                                       'zero_magnitude': f['B'] == 0,
                                       'factor_sign_matches_action_adv': f['B'] == 0 or
                                           np.sign(f['D_factor']) == np.sign(-f['b_mean']),
                                       'epsilon_abstention_matches_action_adv':
                                           (abs(f['D_factor']) > EPS) == (abs(f['b_mean']) > EPS)})
                    for meta in registry():
                        if meta['aggregation'] == agg:
                            scores.append({**base, 'score': meta['score'], 'value': f[meta['name']], **counts})
    return pd.DataFrame(scores), pd.DataFrame(components)


def sign_null_group(group, aggregation, trajectory_ids, signs):
    """Same 512 trajectory-block signs; fixed |b| denominator and magnitude.

    Low/high guards retain exact constant numerators even when a sign pattern
    makes all contributing b values equal. No action/token-independent shuffle.
    """
    signs = np.asarray(signs, float)
    if signs.ndim != 2 or signs.shape[1] != len(trajectory_ids) or not np.isin(signs, [-1., 1.]).all():
        raise ValueError('Whole-trajectory +/-1 masks required')
    ids = group.trajectory_id.map({t: i for i, t in enumerate(trajectory_ids)})
    if ids.isna().any():
        raise ValueError('A trajectory has no registered sign mask')
    ids = ids.to_numpy(int)
    weights = aggregation_weights(group, aggregation)
    b = group.advantage.to_numpy(float)*group.chosen_delta.to_numpy(float)
    f = factors(group.delta_centered_norm, group.chosen_delta, group.advantage, weights)
    blocks = np.zeros(len(trajectory_ids))
    np.add.at(blocks, ids, weights*b)
    if not np.isclose(blocks.sum(), f['b_mean'], rtol=1e-12, atol=1e-14):
        raise ValueError('Block and row reward numerators disagree')
    lo = np.full(len(trajectory_ids), np.inf); hi = np.full_like(lo, -np.inf)
    np.minimum.at(lo, ids, b); np.maximum.at(hi, ids, b)
    occupied = np.isfinite(lo)
    low = np.where(signs[:, occupied] > 0, lo[occupied], -hi[occupied]).min(axis=1)
    high = np.where(signs[:, occupied] > 0, hi[occupied], -lo[occupied]).max(axis=1)
    numerator = np.where(low == high, low, signs @ blocks)
    r = numerator/(f['b_absolute_mean']+EPS)
    return np.column_stack([-f['B']*r, -r])
