"""Matched-pool descriptive statistics, never token-level significance tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

EPS = 1e-12


def point_metrics_many(scores, target, threshold=0.):
    """Vectorized standard threshold-grouped AP, including exact score ties."""
    scores = np.asarray(scores, float)
    target = np.asarray(target, float)
    if scores.ndim != 2 or scores.shape[1] != len(target) or not np.isfinite(scores).all() or not np.isfinite(target).all():
        raise ValueError('Finite matched scores and targets required')
    positive = target > threshold+EPS
    npos = positive.sum(); n = len(target)
    order = np.argsort(-scores, axis=1, kind='stable')
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    ends = np.concatenate([sorted_scores[:, :-1] != sorted_scores[:, 1:], np.ones((len(scores), 1), bool)], axis=1)
    cumulative = positive[order].cumsum(axis=1)
    previous = np.zeros(len(scores)); total = np.zeros(len(scores))
    for j in range(n):
        total += ends[:, j]*(cumulative[:, j]-previous)*cumulative[:, j]/(j+1)
        previous = np.where(ends[:, j], cumulative[:, j], previous)
    ap = total/npos if npos else np.full(len(scores), np.nan)
    ranks = rankdata(scores, axis=1)
    auc = ((ranks[:, positive].sum(axis=1)-npos*(npos+1)/2)/(npos*(n-npos))
           if 0 < npos < n else np.full(len(scores), np.nan))
    return {'average_precision': ap, 'auroc_decline_vs_rest': auc}


def metric_arrays(scores, targets, threshold=0.):
    """Fixed scores, repeated paired labels. Each result is [draw, method]."""
    scores, targets = np.asarray(scores, float), np.asarray(targets, float)
    if scores.ndim != 2 or targets.ndim != 2 or scores.shape[1] != targets.shape[1] or not np.isfinite(scores).all() or not np.isfinite(targets).all():
        raise ValueError('Complete fixed pools required; never drop skills inside a draw')
    n = scores.shape[1]; m = len(scores); b = len(targets)
    positive = targets > threshold+EPS
    negative = targets < -threshold-EPS
    npos = positive.sum(axis=1); nneg = negative.sum(axis=1)
    ap = np.full((b, m), np.nan)
    for i, row in enumerate(scores):
        order = np.argsort(-row, kind='stable')
        ends = np.r_[np.flatnonzero(np.diff(row[order]) != 0), n-1]
        cumulative = positive[:, order].cumsum(axis=1)[:, ends]
        increments = np.diff(cumulative, prepend=np.zeros((b, 1)), axis=1)
        ap[:, i] = np.divide((increments*cumulative/(ends+1)).sum(axis=1), npos,
            out=np.full(b, np.nan), where=npos > 0)
    ranks = rankdata(scores, axis=1)
    auc = np.divide(positive @ ranks.T - npos[:, None]*(npos[:, None]+1)/2,
        (npos*(n-npos))[:, None], out=np.full((b, m), np.nan), where=((npos > 0)&(npos < n))[:, None])
    # A conditional direction AUC is different from decline-vs-rest.
    conditional_auc = np.full((b, m), np.nan)
    for j in range(m):
        comparisons = (scores[j, :, None] > scores[j, None, :]).astype(float)
        comparisons += .5*(scores[j, :, None] == scores[j, None, :])
        numerator = ((positive @ comparisons)*negative).sum(axis=1)
        conditional_auc[:, j] = np.divide(numerator, npos*nneg,
            out=np.full(b, np.nan), where=(npos*nneg) > 0)
    xr = ranks-ranks.mean(axis=1, keepdims=True)
    yr = rankdata(targets, axis=1); yr -= yr.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(yr, axis=1)[:, None]*np.linalg.norm(xr, axis=1)[None, :]
    rho = np.divide(yr @ xr.T, denominator, out=np.full((b, m), np.nan), where=denominator > 0)
    nonzero = positive | negative
    calls = np.abs(scores) > EPS
    truth = np.sign(targets)
    score_sign = np.sign(scores)
    matched = (truth[:, None, :] == score_sign[None, :, :]) & nonzero[:, None, :] & calls[None, :, :]
    called = (nonzero[:, None, :] & calls[None, :, :]).sum(axis=2)
    nonzero_count = nonzero.sum(axis=1)[:, None]
    correct = matched.sum(axis=2)
    sign_acc = np.divide(correct, called, out=np.full((b, m), np.nan), where=called > 0)
    coverage = np.divide(called, nonzero_count, out=np.full((b, m), np.nan), where=nonzero_count > 0)
    unconditional = np.divide(correct, nonzero_count, out=np.full((b, m), np.nan), where=nonzero_count > 0)
    return {'average_precision': ap, 'auroc_decline_vs_rest': auc,
        'auroc_decline_vs_increase': conditional_auc, 'spearman': rho,
        'sign_accuracy_called': sign_acc, 'sign_coverage': coverage,
        'sign_accuracy_abstention_wrong': unconditional}


def selection_matrix(scores, k):
    scores = np.asarray(scores, float)
    if not 1 <= k <= scores.shape[1]:
        raise ValueError('Invalid budget')
    cutoff = np.sort(scores, axis=1)[:, -k]
    above, equal = scores > cutoff[:, None], scores == cutoff[:, None]
    return above.astype(float)+equal*(k-above.sum(axis=1))[:, None]/equal.sum(axis=1)[:, None]


def budget_arrays(scores, targets, k, threshold=0.):
    weights = selection_matrix(scores, k)
    positive = targets > threshold+EPS
    gain = np.maximum(targets, 0.)
    hits = positive @ weights.T
    npos = positive.sum(axis=1)[:, None]
    mass = gain.sum(axis=1)[:, None]
    return {'precision_at_k': hits/k,
        'recall_at_k': np.divide(hits, npos, out=np.full(hits.shape, np.nan), where=npos > 0),
        'captured_decline_mass': np.divide(gain @ weights.T, mass, out=np.full(hits.shape, np.nan), where=mass > EPS)}


def bootstrap_targets(margins, skills, control, repetitions, rng_seed):
    """Global paired game+continuation resampling, preserving shared-game dependence.

    Missing games for a skill invalidate the WHOLE common-pool draw. This is
    reported, not hidden through per-method complete-case selection.
    """
    q = margins[(margins.purpose == 'gold') & margins.skill_id.isin(skills)].copy()
    keys = ['skill_id', 'anchor_id', 'game_id', 'context_id', 'phase', 'trigger_step', 'continuation_seed']
    old = q[q['update'] == 0]; new = q[q['update'] == 5]
    paired = old.merge(new, on=keys, validate='one_to_one', how='outer', suffixes=('_old', '_new'), indicator=True)
    if not paired._merge.eq('both').all():
        raise ValueError('Missing endpoint pair')
    paired['delta'] = paired[f'M_{control}_new']-paired[f'M_{control}_old']
    if paired.duplicated(['game_id', 'skill_id', 'continuation_seed']).any():
        raise ValueError('This analysis requires one first-call source trajectory per skill/game')
    games = sorted(paired.game_id.unique()); seeds = sorted(map(int, paired.continuation_seed.unique()))
    index = pd.MultiIndex.from_product([games, skills], names=['game_id', 'skill_id'])
    matrix = paired.pivot(index=['game_id', 'skill_id'], columns='continuation_seed', values='delta').reindex(index=index, columns=seeds).to_numpy()
    matrix = matrix.reshape(len(games), len(skills), len(seeds))
    if np.any(np.isfinite(matrix).any(axis=2) != np.isfinite(matrix).all(axis=2)):
        raise ValueError('Missing continuation in an observed game')
    point_games = matrix.mean(axis=2)
    point = -np.nansum(point_games, axis=0)/np.isfinite(point_games).sum(axis=0)
    rng = np.random.default_rng(rng_seed)
    result = np.full((repetitions, len(skills)), np.nan)
    for start in range(0, repetitions, 100):
        b = min(100, repetitions-start)
        gi = rng.integers(len(games), size=(b, len(games)))
        ri = rng.integers(len(seeds), size=(b, len(games), len(seeds)))
        sampled = matrix[gi[:, :, None, None], np.arange(len(skills))[None, None, :, None], ri[:, :, None, :]].mean(axis=3)
        counts = np.isfinite(sampled).sum(axis=1)
        result[start:start+b] = np.divide(-np.nansum(sampled, axis=1), counts,
            out=np.full((b, len(skills)), np.nan), where=counts > 0)
    valid = np.isfinite(result).all(axis=1)
    return result, point, {'requested_draws': repetitions, 'complete_pool_draws': int(valid.sum()),
        'missing_any_skill_draws': int((~valid).sum()), 'distinct_source_games': len(games),
        'continuation_seeds': seeds, 'rng_seed': rng_seed,
        'scope': 'conditional gold-label uncertainty only; not training/seed uncertainty; not selection-adjusted'}
