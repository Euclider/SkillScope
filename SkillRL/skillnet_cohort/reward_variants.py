"""CPU-only, explicitly exploratory reward readouts over immutable scalar rows.

Nothing in this module loads a policy, changes the registered main D, or sees
utility labels. All recipe signs and scales are fixed before outcome analysis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

EPS = 1e-12
AGGREGATIONS = ('token', 'decision', 'game')


@dataclass(frozen=True)
class Recipe:
    name: str
    family: str
    formula: str
    signed: bool = True
    geometry: bool = True


RECIPES = (
    Recipe('D_original', 'rectification', 'g_raw * [-P]+', False),
    Recipe('D_centered_gate', 'rectification', 'g_centered * [-P]+', False),
    Recipe('D_ungated', 'rectification', 'valid * [-P]+', False),
    Recipe('D_signed', 'rectification', '-P'),
    Recipe('D_signed_gate', 'rectification', '-g_raw * P'),
    Recipe('D_sign_balance', 'rectification', '-valid * sign(P)'),
    Recipe('D_negative_fraction', 'rectification', 'valid * I(P < -epsilon)', False),
    Recipe('D_adv', 'reward_strength', '-abs(A) * P'),
    Recipe('D_adv_gate', 'reward_strength', '-g_raw * abs(A) * P'),
    Recipe('D_sqrtadv', 'reward_strength', '-sqrt(abs(A)) * P'),
    Recipe('D_work', 'reward_strength', '-valid * dot(d, delta)'),
    Recipe('D_work_gate', 'reward_strength', '-g_raw * dot(d, delta)'),
    Recipe('D_cos_raw', 'interaction_cosine', '-valid_delta * dot(d,delta)/(norm(d)*norm(delta)+epsilon)'),
    Recipe('D_cos_centered', 'interaction_cosine', '-valid_delta_centered * dot(d,delta)/(norm(d)*norm(delta_centered)+epsilon)'),
    Recipe('D_cos_centered_gate', 'interaction_cosine', '-g_centered * Q_centered'),
    Recipe('D_cos_adv', 'interaction_cosine', '-abs(A) * Q_centered'),
    Recipe('D_cos_negative', 'interaction_cosine', '[-Q_centered]+', False),
    Recipe('D_soft', 'soft_gate', '-[C_centered]+ * P'),
    Recipe('D_soft_negative', 'soft_gate', '[C_centered]+ * [-P]+', False),
    Recipe('D_soft_cos', 'soft_gate', '-[C_centered]+ * Q_centered'),
    Recipe('D_sigmoid', 'soft_gate', '-sigmoid(C_centered/median_abs_C) * valid * P'),
    *(Recipe('D_tauC_q'+str(q), 'threshold', '-P * I(C_centered >= Q'+str(q)+'(abs(C_centered))) * valid_delta_centered') for q in (25, 50, 75)),
    *(Recipe('D_delta_q'+str(q), 'threshold', '-P * I(norm(delta_centered) >= Q'+str(q)+'(norm(delta_centered))) * valid') for q in (25, 50, 75)),
    Recipe('D_winsor', 'robust', '-clip(P, -Q95(abs(P)), Q95(abs(P)))'),
    Recipe('D_tanh', 'robust', '-tanh(P/median_abs_P)'),
    Recipe('D_action_sign', 'sampled_action', '-sign(A) * chosen_delta', geometry=False),
    Recipe('D_action_adv', 'sampled_action', '-A * chosen_delta', geometry=False),
    Recipe('D_action_adv_clip', 'sampled_action', '-clip(A,-2,2) * chosen_delta', geometry=False),
    Recipe('D_ratio_adv', 'sampled_action', '-A * (clip(exp(chosen_u_O),.8,1.2)-clip(exp(chosen_u_control),.8,1.2))', geometry=False),
    Recipe('D_policy_action_adv', 'policy_only', '-A * chosen_u_O', geometry=False),
    Recipe('D_reward_only', 'reward_only', '-A', geometry=False),
    Recipe('D_adv_negative_fraction', 'reward_only', 'I(A < -epsilon)', False, False),
    Recipe('C_raw', 'reward_alignment_reference', '+C_raw', False),
    Recipe('C_centered', 'reward_alignment_reference', '+C_centered', False),
    Recipe('C_adv_centered', 'reward_alignment_reference', '+abs(A) * C_centered', False),
)

MAGNITUDES = {
    'M_delta_raw': 'delta_norm', 'M_delta_centered': 'delta_centered_norm',
    'M_original_norm': 'u_original_norm', 'M_control_norm': 'u_control_norm',
    'M_kl': 'forward_kl_original', 'M_js': 'js_original',
    'M_action_abs': 'abs(chosen_delta)',
}


def registry():
    rows = []
    for aggregation in AGGREGATIONS:
        for recipe in RECIPES:
            for mode in ('reward', 'unsigned'):
                rows.append({**asdict(recipe), 'aggregation': aggregation, 'mode': mode,
                    'score': score_id(recipe.name, aggregation, mode),
                    'reward_removed_scope': ('A=+1 on ORIGINAL nonzero-direction support; zero-A geometry unavailable'
                        if recipe.geometry else 'A=+1 on ALL recorded loss tokens') if mode == 'unsigned' else None})
        for name, formula in MAGNITUDES.items():
            rows.append({'name': name, 'family': 'magnitude_only', 'formula': formula,
                'signed': False, 'geometry': False, 'aggregation': aggregation,
                'mode': 'magnitude', 'score': score_id(name, aggregation, 'magnitude')})
    return rows


def score_id(name, aggregation, mode='reward'):
    return f'{name}::{aggregation}::{mode}'


def validate_tokens(t):
    keys = ['control', 'decision_id', 'response_token_offset']
    if t.duplicated(keys).any():
        raise ValueError('Duplicate actual token/control rows')
    for field in ('advantage', 'game_id', 'trajectory_id', 'group_id', 'skill_id', 'phase'):
        if (t.groupby(['control', 'decision_id'])[field].nunique() != 1).any():
            raise ValueError('Decision identity or advantage is not constant: '+field)
    # Unit-advantage projections cannot be recovered from A=0 scalar rows.
    # Require that this limitation is exactly the inactive set in this cohort.
    if not np.array_equal(t.direction_valid.to_numpy(bool), t.advantage.ne(0).to_numpy()):
        raise ValueError('Nonzero but invalid direction requires a separately registered common-support rule')
    fields = ['P_int', 'C_upd', 'C_upd_centered', 'd_norm', 'advantage', 'delta_norm',
              'delta_centered_norm', 'u_original_norm', 'chosen_u_original', 'chosen_delta']
    if not np.isfinite(t[fields].to_numpy(float)).all():
        raise ValueError('Non-finite input scalar')
    if len(set(t.control)) != 2 or set(t.control) != {'placebo', 'null'}:
        raise ValueError('Both control arms are required')
    a, b = (t[t.control == c].sort_values(keys[1:]).reset_index(drop=True) for c in ('placebo', 'null'))
    for field in keys[1:]+['advantage', 'game_id', 'trajectory_id', 'skill_id', 'C_upd', 'C_upd_centered']:
        if not a[field].equals(b[field]):
            raise ValueError('Control arm identity/reward drift: '+field)


def label_blind_scales(t):
    """Per-control pooled thresholds; never per skill, never outcome selected."""
    q = t[t.direction_valid]
    if q.empty:
        raise ValueError('No observed reward direction')
    result = {'tau_delta': 1e-8, 'epsilon': EPS,
              'p_median': max(float(q.P_int.abs().median()), EPS),
              'p_q95': max(float(q.P_int.abs().quantile(.95)), EPS),
              'c_median': max(float(q.C_upd_centered.abs().median()), EPS)}
    for percentile in (25, 50, 75):
        result[f'c_q{percentile}'] = float(q.C_upd_centered.abs().quantile(percentile/100))
        result[f'delta_q{percentile}'] = float(q.delta_centered_norm.quantile(percentile/100))
    return result


def token_matrix(t, scales, mode='reward', flip=None):
    """Return every fixed recipe. Inactive tokens stay in all denominators."""
    if mode not in ('reward', 'unsigned'):
        raise ValueError('Unknown reward ablation')
    a = t.advantage.to_numpy(float).copy()
    dn = t.d_norm.to_numpy(float).copy()
    p = t.P_int.to_numpy(float).copy()
    c = t.C_upd.to_numpy(float).copy()
    cc = t.C_upd_centered.to_numpy(float).copy()
    valid = t.direction_valid.to_numpy(bool)
    delta, deltac = (t[k].to_numpy(float) for k in ('delta_norm', 'delta_centered_norm'))
    if mode == 'unsigned':
        active = a != 0
        abs_a = np.abs(a)
        # Reconstruct q^T delta and ||q|| exactly from the stable scalar contract.
        dot_unit = np.divide(p*(dn+EPS), a, out=np.zeros_like(a), where=active)
        dot_u = c*np.maximum(dn*t.u_original_norm.to_numpy(float), EPS)
        dot_u_unit = np.divide(dot_u, a, out=np.zeros_like(a), where=active)
        dn = np.divide(dn, abs_a, out=np.zeros_like(a), where=active)
        p = dot_unit/(dn+EPS)
        c = np.divide(dot_u_unit, np.maximum(dn*t.u_original_norm.to_numpy(float), EPS))
        # Recover the centered denominator without needing the missing vector.
        # The recorded centered cosine denominator is clamped, so derive its
        # unit-A counterpart from the uncentered numerator only where unclamped.
        # We instead rescale the exact centered dot/denominator using the
        # stored original/centered cosine ratio when both are nonzero.
        numerator = t.C_upd.to_numpy(float)*np.maximum(t.d_norm.to_numpy(float)*t.u_original_norm.to_numpy(float), EPS)
        centered_denom = np.divide(numerator, t.C_upd_centered.to_numpy(float),
            out=np.full_like(a, EPS), where=t.C_upd_centered.to_numpy(float) != 0)
        identifiable = active & (t.C_upd_centered.to_numpy(float) != 0)
        if np.any(centered_denom[identifiable] <= EPS*(1+1e-10)):
            raise ValueError('Centered denominator was clamped; unit-advantage geometry not identifiable')
        # In the real cohort all nonzero-A centered denominators exceed EPS;
        # the runner verifies this rather than guessing near-clamp geometry.
        centered_unit_denom = np.maximum(np.divide(centered_denom, abs_a,
            out=np.zeros_like(a), where=active), EPS)
        cc = dot_u_unit/centered_unit_denom
        a = np.ones_like(a)  # Genuine full-row A=+1 for sampled-action recipes.
        p = np.where(valid, p, 0.); c = np.where(valid, c, 0.); cc = np.where(valid, cc, 0.)
    elif flip is not None:
        f = np.broadcast_to(np.asarray(flip, dtype=float), a.shape)
        if not np.isin(f, [-1., 1.]).all():
            raise ValueError('Sign null must preserve magnitude and use trajectory-block +/-1')
        a *= f; p *= f; c *= f; cc *= f
    gate = valid & t.fidelity_valid.to_numpy(bool) & (c >= 0) & (delta >= scales['tau_delta'])
    gatec = valid & t.fidelity_valid_centered.to_numpy(bool) & (cc >= 0) & (deltac >= scales['tau_delta'])
    vd = valid & (delta >= scales['tau_delta'])
    vdc = valid & (deltac >= scales['tau_delta'])
    dot = p*(dn+EPS)
    qr = np.where(vd, dot/(dn*delta+EPS), 0.)
    qc = np.where(vdc, dot/(dn*deltac+EPS), 0.)
    strength = np.abs(a)
    chosen = t.chosen_delta.to_numpy(float)
    chosen_o = t.chosen_u_original.to_numpy(float)
    ratio_o = np.exp(np.clip(chosen_o, np.log(.8), np.log(1.2)))
    ratio_b = np.exp(np.clip(chosen_o-chosen, np.log(.8), np.log(1.2)))
    negative = np.maximum(-p, 0.)
    values = {
        'D_original': gate*negative, 'D_centered_gate': gatec*negative,
        'D_ungated': valid*negative, 'D_signed': -p, 'D_signed_gate': -(gate*p),
        'D_sign_balance': -(valid*np.sign(p)), 'D_negative_fraction': valid*(p < -EPS),
        'D_adv': -strength*p, 'D_adv_gate': -(gate*strength*p),
        'D_sqrtadv': -np.sqrt(strength)*p, 'D_work': -(valid*dot), 'D_work_gate': -(gate*dot),
        'D_cos_raw': -qr, 'D_cos_centered': -qc, 'D_cos_centered_gate': -(gatec*qc),
        'D_cos_adv': -strength*qc, 'D_cos_negative': np.maximum(-qc, 0.),
        'D_soft': -np.maximum(cc, 0.)*p,
        'D_soft_negative': np.maximum(cc, 0.)*negative,
        'D_soft_cos': -np.maximum(cc, 0.)*qc,
        'D_sigmoid': -expit(cc/scales['c_median'])*valid*p,
        'D_winsor': -np.clip(p, -scales['p_q95'], scales['p_q95']),
        'D_tanh': -np.tanh(p/scales['p_median']),
        'D_action_sign': -np.sign(a)*chosen, 'D_action_adv': -a*chosen,
        'D_action_adv_clip': -np.clip(a, -2., 2.)*chosen,
        'D_ratio_adv': -a*(ratio_o-ratio_b), 'D_policy_action_adv': -a*chosen_o,
        'D_reward_only': -a, 'D_adv_negative_fraction': a < -EPS,
        'C_raw': c, 'C_centered': cc, 'C_adv_centered': strength*cc,
    }
    for percentile in (25, 50, 75):
        values[f'D_tauC_q{percentile}'] = -p*vdc*(cc >= scales[f'c_q{percentile}'])
        values[f'D_delta_q{percentile}'] = -p*valid*(deltac >= scales[f'delta_q{percentile}'])
    result = np.column_stack([values[r.name] for r in RECIPES]).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite derived score')
    return result


def magnitude_matrix(t):
    return np.column_stack([t[k].to_numpy(float) if not k.startswith('abs(')
        else t.chosen_delta.abs().to_numpy(float) for k in MAGNITUDES.values()])


def aggregation_weights(t, aggregation):
    """Linear weights sum to one; game mode = token->decision->trajectory->game."""
    if t.empty or aggregation not in AGGREGATIONS:
        raise ValueError('Nonempty rows and registered aggregation required')
    if aggregation == 'token':
        return np.full(len(t), 1/len(t))
    counts = t.groupby('decision_id').size()
    w = 1/t.decision_id.map(counts).to_numpy(float)
    if aggregation == 'decision':
        return w/len(counts)
    decisions = t.drop_duplicates('decision_id')
    per_trajectory = decisions.groupby('trajectory_id').size()
    trajectories = decisions.drop_duplicates('trajectory_id')
    per_game = trajectories.groupby('game_id').size()
    w /= t.trajectory_id.map(per_trajectory).to_numpy(float)
    w /= t.game_id.map(per_game).to_numpy(float)
    w /= len(per_game)
    if not np.isclose(w.sum(), 1., atol=1e-12):
        raise ValueError('Unnormalized hierarchy')
    return w


def aggregate(t, matrix, aggregation):
    if matrix.shape[0] != len(t):
        raise ValueError('Mismatched scalar row order')
    return aggregation_weights(t, aggregation) @ matrix
