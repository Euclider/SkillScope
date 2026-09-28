"""v2 numerical correction: an exact constant must remain an exact constant.

Preserves v1 source/results, candidate registry, formulas, and all nonconstant
reductions. Also protects the trajectory-sign null against the same artifact.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import reward_variant_analysis as base
from .common import REPO, file_hash, read_json, write_new_bytes, write_new_json
from .reward_variants import AGGREGATIONS, EPS, MAGNITUDES, RECIPES, magnitude_matrix, score_id, token_matrix

OUTPUT = base.COHORT/'reward-variants-s404-v2'
PRIOR = base.COHORT/'reward-variants-s404-v1'
BOUNDS = {}
FIX_ROWS = []
ORIGINAL_COMPUTE = base.compute_scores


def exact_constant_columns(matrix):
    low, high = np.min(matrix, axis=0), np.max(matrix, axis=0)
    return low == high, low


def correct_scores(t, scales):
    scores, blocks = ORIGINAL_COMPUTE(t, scales)
    index_columns = ['control', 'context_id', 'phase', 'skill_id', 'score']
    lookup = {tuple(row): i for i, row in enumerate(scores[index_columns].itertuples(index=False, name=None))}
    BOUNDS.clear(); FIX_ROWS.clear()
    for control in ('placebo', 'null'):
        part = t[t.control == control].reset_index(drop=True)
        plus = token_matrix(part, scales[control])
        minus = token_matrix(part, scales[control], flip=-1.)
        matrices = {'reward': plus, 'unsigned': token_matrix(part, scales[control], 'unsigned'),
                    'magnitude': magnitude_matrix(part)}
        trajectories = sorted(part.trajectory_id.unique())
        ti = {key: i for i, key in enumerate(trajectories)}
        for phase in ('all', 'initial', 'early', 'middle', 'late'):
            subset = part if phase == 'all' else part[part.phase == phase]
            for (context, skill), group in subset.groupby(['context_id', 'skill_id'], sort=True):
                ids = group.index.to_numpy()
                for mode, matrix in matrices.items():
                    constant, values = exact_constant_columns(matrix[ids])
                    names = [r.name for r in RECIPES] if mode != 'magnitude' else list(MAGNITUDES)
                    for j in np.flatnonzero(constant):
                        for aggregation in AGGREGATIONS:
                            sid = score_id(names[j], aggregation, mode)
                            ix = lookup[(control, context, phase, skill, sid)]
                            before = float(scores.at[ix, 'value']); after = float(values[j])
                            if before != after:
                                FIX_ROWS.append({'control': control, 'context_id': context, 'phase': phase,
                                    'skill_id': skill, 'score': sid, 'old': before, 'new': after,
                                    'absolute_change': abs(before-after), 'reason': 'all contributing token values exactly equal'})
                                scores.at[ix, 'value'] = after
                if phase == 'all':
                    pl = np.full((len(trajectories), len(RECIPES)), np.inf)
                    ph = np.full_like(pl, -np.inf); ml = pl.copy(); mh = ph.copy()
                    for trajectory, rows in group.groupby('trajectory_id'):
                        at = ti[trajectory]; ri = rows.index.to_numpy()
                        pl[at], ph[at] = plus[ri].min(axis=0), plus[ri].max(axis=0)
                        ml[at], mh[at] = minus[ri].min(axis=0), minus[ri].max(axis=0)
                    BOUNDS[(control, context, skill)] = (pl, ph, ml, mh)
    # A necessary analytical identity, not an outcome-dependent criterion.
    unsigned_reward_only = scores[scores.score.str.startswith('D_reward_only::') & scores.score.str.endswith('::unsigned')]
    if not unsigned_reward_only.value.eq(-1.).all():
        raise ValueError('Constant A=+1 ablation must equal -1 exactly')
    base.progress('CONSTANT_IDENTITY_CORRECTION', changed_scalar_cells=len(FIX_ROWS))
    return scores, blocks


def correct_null_constants(cube, flipped, control, pool):
    for si, skill in enumerate(pool['shared_skill_ids']):
        pl, ph, ml, mh = BOUNDS[(control, pool['context_id'], skill)]
        lo = np.where(flipped[:, :, None], ml[None], pl[None]).min(axis=1)
        hi = np.where(flipped[:, :, None], mh[None], ph[None]).max(axis=1)
        constant = lo == hi
        for ai, _ in enumerate(AGGREGATIONS):
            sl = slice(ai*len(RECIPES), (ai+1)*len(RECIPES))
            cube[:, sl, si] = np.where(constant, lo, cube[:, sl, si])


def sign_null_fixed(blocks, scores, units, pools, metadata, plan):
    reward = [m for m in metadata if m['mode'] == 'reward']
    trajectories = next(iter(blocks.values()))[0]
    if any(v[0] != trajectories for v in blocks.values()):
        raise ValueError('Control trajectory identities differ')
    rng = np.random.default_rng(plan['sign_null_rng_seed'])
    flipped = rng.integers(0, 2, size=(plan['sign_null_repetitions'], len(trajectories)))
    summaries, outputs, family_max = [], [], []
    for control in ('placebo', 'null'):
        pool, = [p for p in pools[control+'/stable_raw'] if p['phase'] == 'all']
        _, target, _ = base.pool_data(scores, units, pool, control, metadata)
        cube = np.empty((len(flipped), len(reward), len(target)))
        for si, skill in enumerate(pool['shared_skill_ids']):
            for ai, aggregation in enumerate(AGGREGATIONS):
                _, plus, minus = blocks[(control, pool['context_id'], skill, aggregation)]
                cube[:, ai*len(RECIPES):(ai+1)*len(RECIPES), si] = plus.sum(axis=0)+flipped @ (minus-plus)
        correct_null_constants(cube, flipped, control, pool)
        observed_matrix = scores[(scores.control == control)&(scores.phase == 'all')&(scores.context_id == pool['context_id'])].pivot(index='score', columns='skill_id', values='value').reindex(index=[m['score'] for m in reward], columns=pool['shared_skill_ids']).to_numpy()
        for threshold in plan['event_thresholds']:
            simulated = base.point_metrics_many(cube.reshape(-1, len(target)), target, threshold)
            observed = base.point_metrics_many(observed_matrix, target, threshold)
            arrays = {key: vals.reshape(len(flipped), len(reward)) for key, vals in simulated.items()}
            max_ap = np.max(arrays['average_precision'], axis=1)
            family_max.extend({'control': control, 'threshold': threshold, 'draw': j,
                'maximum_reward_candidate_AP': float(v)} for j, v in enumerate(max_ap))
            for i, meta in enumerate(reward):
                for key, values in arrays.items():
                    v = values[:, i]; obs = observed[key][i]
                    summaries.append({'control': control, 'phase': 'all', 'threshold': threshold,
                        'score': meta['score'], 'metric': key, 'observed': float(obs), **base.interval(v),
                        'reference_fraction_ge_observed': float(np.mean(v >= obs-EPS)) if np.isfinite(obs) else np.nan,
                        'family_max_fraction_ge_observed_AP': float(np.mean(max_ap >= obs-EPS)) if key == 'average_precision' and np.isfinite(obs) else np.nan,
                        'is_calibrated_p_value': False})
                outputs.extend({'control': control, 'threshold': threshold, 'score': meta['score'],
                    'draw': j, **{key: float(vals[j, i]) for key, vals in arrays.items()}} for j in range(len(flipped)))
        base.progress('SIGN_NULL_CONTROL_CONSTANT_SAFE', control=control, draws=len(flipped))
    return pd.DataFrame(summaries), pd.DataFrame(outputs), pd.DataFrame(family_max), pd.DataFrame(1-2*flipped, columns=trajectories)


def prepare(output):
    if output.exists() or output.parent != base.COHORT or not output.name.startswith('reward-variants-s404-'):
        raise ValueError('A new scoped output is required')
    prior = base.check_plan(PRIOR)
    plan = copy.deepcopy(prior)
    plan.update(output=str(output), created_utc=datetime.now(timezone.utc).isoformat(),
        version='reward_variants_seed404_exploratory_v2_constant_preserving',
        arithmetic_correction='exact constant input columns retained exactly; same guard in sign null',
        candidate_registry_unchanged=True, nonconstant_reductions_unchanged=True,
        prior_result=str(PRIOR), prior_result_preserved=True)
    plan['analysis_sources'] += [{'path': str(p), 'sha256': file_hash(p)} for p in
        [Path(__file__), REPO/'tests/skillnet_cohort/test_reward_variant_constant_fix.py']]
    plan['inputs'] += [{'path': str(PRIOR/n), 'sha256': file_hash(PRIOR/n)} for n in
        ['plan.json', 'complete.json', 'provenance.json', 'ranking_diagnostics.csv', 'skill_scores.csv']]
    note = REPO/'docs/experiments/phase12-independent-v4/REWARD-VARIANTS-CONSTANT-20260922-v2.md'
    plan['inputs'].append({'path': str(note), 'sha256': file_hash(note)})
    write_new_json(output/'plan.json', plan)
    base.csv(output/'registry.csv', pd.DataFrame(plan['registry']))
    base.progress('V2_PREPARED_CONSTANT_FIX_ONLY', output=str(output))


def run(output):
    base.compute_scores = correct_scores
    base.sign_null_analysis = sign_null_fixed
    base.run(output)
    base.csv(output/'constant-correction-cells.csv', pd.DataFrame(FIX_ROWS))
    write_new_json(output/'constant-correction-receipt.json', {'prior': str(PRIOR),
        'changed_cells': len(FIX_ROWS), 'nonconstant_reductions_unchanged': True,
        'same_candidate_registry': True, 'no_new_candidates_after_labels': True,
        'source_sha256': file_hash(Path(__file__)),
        'cell_changes_sha256': file_hash(output/'constant-correction-cells.csv')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    a = parser.parse_args()
    if a.mode == 'prepare':
        prepare(a.output.resolve())
    else:
        try:
            run(a.output.resolve())
        except Exception as error:
            if (a.output/'run-intent.json').exists() and not (a.output/'complete.json').exists():
                write_new_json(a.output/'failed.json', {'error': repr(error), 'time_unix': time.time(), 'automatic_retry': False})
            raise
