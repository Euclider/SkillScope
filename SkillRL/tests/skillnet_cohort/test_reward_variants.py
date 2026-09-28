import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from skillnet_cohort import reward_variants as rv
from skillnet_cohort import reward_variant_analysis as analysis
from skillnet_cohort.reward_variant_statistics import (bootstrap_targets, budget_arrays,
    metric_arrays, point_metrics_many, selection_matrix)


def tokens(advantages=None):
    import torch
    from phase2.stable_direction import token_signals
    a = np.array([1., -2., 0., .5]) if advantages is None else np.array(advantages, float)
    old = torch.tensor([[.5, .3, .2], [.2, .5, .3], [.7, .2, .1], [.1, .5, .4]], dtype=torch.float64).log()
    new = old+torch.tensor([[.2, -.1, .1], [.3, .1, -.2], [0., .1, -.2], [-.2, .3, .1]], dtype=torch.float64)
    rows = []
    for control, factor in [('placebo', .3), ('null', -.2)]:
        args = (old, new, old, old+(new-old)*factor, torch.tensor([0, 1, 2, 0]), torch.tensor(a))
        signals = token_signals(*args)
        for i in range(4):
            rows.append({'control': control, 'decision_id': 'd'+str(i), 'response_token_offset': 0,
                'trajectory_id': 't'+str(i//2), 'group_id': 'g', 'game_id': 'game',
                'skill_id': 's'+str(i%2), 'phase': 'early', 'context_id': 'all_alfworld',
                **{key: value[i].item() for key, value in signals.items()}})
    return pd.DataFrame(rows)


def test_registry_complete_unique_and_fixed_signs():
    records = rv.registry()
    assert len(records) == len(set(r['score'] for r in records))
    assert len(rv.RECIPES) == 39
    assert len(records) == 255
    assert all(r['formula'] for r in records)
    assert rv.score_id('D_signed', 'token') in {r['score'] for r in records}


def test_original_scores_exact_formulas_and_no_input_mutation():
    t = tokens(); before = t.copy(deep=True)
    rv.validate_tokens(t)
    q = t[t.control == 'placebo']; scales = rv.label_blind_scales(q)
    m = rv.token_matrix(q, scales); columns = {r.name: i for i, r in enumerate(rv.RECIPES)}
    np.testing.assert_allclose(m[:, columns['D_original']], q.D_contribution)
    np.testing.assert_allclose(m[:, columns['D_centered_gate']], q.D_centered_contribution)
    np.testing.assert_allclose(m[:, columns['D_signed']], -q.P_int)
    np.testing.assert_allclose(m[:, columns['D_adv']], -q.advantage.abs()*q.P_int)
    pd.testing.assert_frame_equal(t, before)


def test_unsigned_geometry_matches_fresh_unit_advantage_full_vectors():
    t = tokens(); q = t[t.control == 'placebo']; scales = rv.label_blind_scales(q)
    unit = tokens([1., 1., 0., 1.]); unit = unit[unit.control == 'placebo']
    got = rv.token_matrix(q, scales, 'unsigned')
    expected = rv.token_matrix(unit, scales)
    for i, recipe in enumerate(rv.RECIPES):
        if recipe.geometry:
            np.testing.assert_allclose(got[:, i], expected[:, i], rtol=1e-10, atol=1e-12, err_msg=recipe.name)
    pos = [r.name for r in rv.RECIPES].index('D_action_adv')
    np.testing.assert_allclose(got[:, pos], -q.chosen_delta)
    assert got[2, pos] != 0  # A=0 does NOT erase the full-row action ablation.


def test_reward_sign_flip_recomputes_gate_and_matches_full_vector_formula():
    q = tokens(); q = q[q.control == 'placebo']; scales = rv.label_blind_scales(q)
    flips = np.array([-1., -1., 1., 1.])
    fresh = tokens(q.advantage.to_numpy()*flips); fresh = fresh[fresh.control == 'placebo']
    np.testing.assert_allclose(rv.token_matrix(q, scales, flip=flips), rv.token_matrix(fresh, scales), atol=1e-12)
    with pytest.raises(ValueError, match='Sign null'):
        rv.token_matrix(q, scales, flip=0.)


def test_hierarchy_and_token_length_do_not_change_game_weights():
    t = pd.DataFrame({'decision_id': ['a', 'a', 'a', 'b', 'c', 'd'],
        'trajectory_id': ['t0', 't0', 't0', 't0', 't1', 't2'],
        'game_id': ['g0']*5+['g1']})
    np.testing.assert_allclose(rv.aggregation_weights(t, 'token'), np.ones(6)/6)
    np.testing.assert_allclose(rv.aggregation_weights(t, 'decision'), [1/12]*3+[.25]*3)
    np.testing.assert_allclose(rv.aggregation_weights(t, 'game'), [1/24]*3+[1/8, 1/4, 1/2])
    short = t.drop_duplicates('decision_id')
    values = np.array([1., 1., 1., 2., 3., 4.])
    for agg in ['decision', 'game']:
        assert rv.aggregation_weights(t, agg) @ values == pytest.approx(rv.aggregation_weights(short, agg) @ np.array([1., 2., 3., 4.]))


@pytest.mark.parametrize('threshold', [0., .05])
def test_metric_implementations_match_sklearn_and_scipy_including_ties(threshold):
    scores = np.array([[1., 1., 0., 0., -1.], [0.]*5, [.3, -.2, .1, .2, -.4]])
    targets = np.array([[.2, -.1, 0., .04, -.2], [0., 0., 0., 0., 0.], [.2]*5])
    arrays = metric_arrays(scores, targets, threshold)
    for j, y in enumerate(targets):
        labels = y > threshold+1e-12
        fast = point_metrics_many(scores, y, threshold)
        for i, score in enumerate(scores):
            if labels.any():
                assert arrays['average_precision'][j, i] == pytest.approx(average_precision_score(labels, score))
            else:
                assert np.isnan(arrays['average_precision'][j, i])
            if labels.any() and not labels.all():
                assert arrays['auroc_decline_vs_rest'][j, i] == pytest.approx(roc_auc_score(labels, score))
            else:
                assert np.isnan(arrays['auroc_decline_vs_rest'][j, i])
            if np.unique(score).size > 1 and np.unique(y).size > 1:
                assert arrays['spearman'][j, i] == pytest.approx(spearmanr(score, y).statistic)
        for key, values in fast.items():
            np.testing.assert_allclose(values, arrays[key][j], atol=1e-14, equal_nan=True)


def test_conditional_direction_auc_and_sign_abstention_are_distinct():
    scores = np.array([[2., 1., 4., 0.]])
    target = np.array([[.2, -.2, 0., -.3]])
    m = metric_arrays(scores, target)
    assert m['auroc_decline_vs_increase'][0, 0] == 1.
    assert m['auroc_decline_vs_rest'][0, 0] == pytest.approx(2/3)
    assert m['sign_accuracy_called'][0, 0] == .5
    assert m['sign_coverage'][0, 0] == pytest.approx(2/3)
    assert m['sign_accuracy_abstention_wrong'][0, 0] == pytest.approx(1/3)


def test_budget_ties_are_gold_blind_and_zero_events_undefined():
    scores = np.array([[1., 1., 0.]])
    np.testing.assert_array_equal(selection_matrix(scores, 1), [[.5, .5, 0.]])
    q = budget_arrays(scores, np.array([[.2, -.1, 0.], [0., 0., 0.]]), 1)
    assert q['precision_at_k'][0, 0] == .5
    assert q['recall_at_k'][0, 0] == .5
    assert np.isnan(q['recall_at_k'][1, 0])


def margins_fixture():
    rows = []
    for game in range(3):
        for skill in ['s0', 's1']:
            for seed in [11, 22]:
                for update in [0, 5]:
                    value = (game-1)*(1 if seed == 11 else .5) if update == 5 else 0.
                    rows.append({'game_id': 'g'+str(game), 'skill_id': skill,
                        'anchor_id': f'a{game}-{skill}', 'context_id': 'all_alfworld',
                        'phase': 'early', 'trigger_step': 1, 'continuation_seed': seed,
                        'update': update, 'purpose': 'gold', 'M_placebo': value, 'M_null': -value})
    return pd.DataFrame(rows)


def test_bootstrap_preserves_cross_skill_game_and_continuation_pairs():
    q = margins_fixture()
    values, point, receipt = bootstrap_targets(q, ['s0', 's1'], 'placebo', 100, 123)
    np.testing.assert_array_equal(values[:, 0], values[:, 1])
    np.testing.assert_array_equal(point, [0., 0.])
    assert receipt['complete_pool_draws'] == 100
    other, _, _ = bootstrap_targets(q, ['s0', 's1'], 'null', 100, 123)
    np.testing.assert_array_equal(values, -other)


def test_missing_bootstrap_skill_remains_missing_not_dropped():
    q = margins_fixture(); q = q[(q.skill_id == 's0') | (q.game_id == 'g0')]
    values, _, receipt = bootstrap_targets(q, ['s0', 's1'], 'placebo', 100, 123)
    assert 0 < receipt['missing_any_skill_draws'] < 100
    assert values.shape == (100, 2)
    assert np.isnan(values[:, 1]).any()


def test_duplicate_tokens_and_nonzero_invalid_directions_fail_closed():
    t = tokens()
    with pytest.raises(ValueError, match='Duplicate'):
        rv.validate_tokens(pd.concat([t, t.iloc[:1]]))
    t.loc[0, 'direction_valid'] = False
    with pytest.raises(ValueError, match='Nonzero but invalid'):
        rv.validate_tokens(t)


def test_block_linearization_equals_direct_trajectory_sign_randomization():
    t = tokens(); scales = {c: rv.label_blind_scales(t[t.control == c]) for c in ['placebo', 'null']}
    scores, blocks = analysis.compute_scores(t, scales)
    for (control, context, skill, aggregation), (trajectories, plus, minus) in blocks.items():
        flip = np.array([0., 1.])
        mixed = plus.sum(axis=0)+flip @ (minus-plus)
        q = t[(t.control == control)&(t.context_id == context)&(t.skill_id == skill)]
        signs = q.trajectory_id.map(dict(zip(trajectories, 1-2*flip))).to_numpy()
        expected = rv.aggregate(q, rv.token_matrix(q, scales[control], flip=signs), aggregation)
        np.testing.assert_allclose(mixed, expected, atol=1e-12)


def test_fixed_scores_do_not_depend_on_input_order():
    t = tokens(); scales = {c: rv.label_blind_scales(t[t.control == c]) for c in ['placebo', 'null']}
    a, _ = analysis.compute_scores(t, scales)
    b, _ = analysis.compute_scores(t.sample(frac=1., random_state=9), scales)
    keys = ['control', 'context_id', 'phase', 'skill_id', 'score']
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[keys], b[keys])
    np.testing.assert_allclose(a.value, b.value, atol=1e-12)


def synthetic_sources(path):
    source = path/'source'; source.mkdir()
    (source/'reports').mkdir(); (source/'reused-labels').mkdir()
    t = tokens(); scales = {c: rv.label_blind_scales(t[t.control == c]) for c in ['placebo', 'null']}
    scores, _ = analysis.compute_scores(t, scales)
    t.to_parquet(source/'token_signals.parquet', index=False)
    features = []
    mapping = {'D_original': ('D_contribution', 1), 'D_centered_gate': ('D_centered_contribution', 1),
        'D_ungated': ('D_ungated_contribution', 1), 'D_signed': ('P_int', -1),
        'C_raw': ('C_upd', 1), 'C_centered': ('C_upd_centered', 1)}
    mapping.update({k: (v, 1) for k, v in rv.MAGNITUDES.items() if not v.startswith('abs(')})
    for (c, context, skill, phase), group in scores.groupby(['control', 'context_id', 'skill_id', 'phase']):
        record = dict(control=c, context_id=context, skill_id=skill, phase=phase)
        for name, (column, sign) in mapping.items():
            mode = 'magnitude' if name in rv.MAGNITUDES else 'reward'
            record[column] = group[group.score == rv.score_id(name, 'token', mode)].value.iloc[0]*sign
        features.append(record)
    pd.DataFrame(features).to_parquet(source/'skill_context_features.parquet', index=False)
    margins = margins_fixture()
    margins.loc[(margins.skill_id == 's0')&(margins['update'] == 5), 'M_placebo'] -= .1
    margins.loc[(margins.skill_id == 's1')&(margins['update'] == 5), 'M_placebo'] += .2
    margins.to_parquet(source/'reused-labels/anchor_margins.parquet', index=False)
    units = []
    for control in ['placebo', 'null']:
        for phase in ['all', 'early']:
            for skill in ['s0', 's1']:
                value = margins[(margins['update'] == 5)&(margins.skill_id == skill)]['M_'+control].mean()
                units.append(dict(control=control, context_id='all_alfworld', phase=phase, skill_id=skill, delta_utility=value))
    pd.DataFrame(units).to_parquet(source/'reused-labels/utility_units.parquet', index=False)
    pools = {c+'/stable_raw': [dict(context_id='all_alfworld', phase=phase, shared_skill_ids=['s0', 's1']) for phase in ['all', 'early']] for c in ['placebo', 'null']}
    (source/'ranking-snapshots.json').write_text(json.dumps({'candidate_pools': pools}))
    references = [{**r, 'variant': 'stable_raw', **{c: 0. for c in analysis.REFERENCE_COLUMNS}} for r in units]
    pd.DataFrame(references).to_csv(source/'reports/scores_and_gold.csv', index=False)
    (source/'reports/coverage_and_effects.csv').write_text('skill_id\ns0\ns1\n')
    (source/'reports/phase2-results.md').write_text('# Original immutable report\n')
    return source


def test_complete_synthetic_pipeline_and_no_clobber(tmp_path, monkeypatch):
    source = synthetic_sources(tmp_path)
    output = tmp_path/'output'; output.mkdir()
    plan = {'registry': rv.registry(), 'bootstrap_repetitions': 20, 'bootstrap_rng_seed': 123,
        'sign_null_repetitions': 8, 'sign_null_rng_seed': 345, 'event_thresholds': [0., .05],
        'inputs': [], 'preserved_runtime_sources': [], 'analysis_sources': [],
        'setting_document': {'path': str(source/'reports/phase2-results.md'),
            'sha256': analysis.file_hash(source/'reports/phase2-results.md')}}
    (output/'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(analysis, 'SOURCE', source)
    monkeypatch.setattr(analysis, 'check_plan', lambda p: plan)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    analysis.run(output)
    assert json.loads((output/'complete.json').read_text())['all_score_columns'] == 261
    assert (source/'reports/phase2-results.md').read_text() == '# Original immutable report\n'
    assert (output/'reports/phase2-results-expanded.md').read_text().startswith('# Original immutable report\n')
    with pytest.raises(FileExistsError, match='No automatic'):
        analysis.run(output)
    provenance = json.loads((output/'provenance.json').read_text())
    analysis.verify_records([{'path': str(output/r['path']), 'sha256': r['sha256']} for r in provenance['files']])
