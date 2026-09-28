import math

import numpy as np
import pandas as pd
import pytest

from skillnet_cohort.factorized_reward import (aggregate_scores, constant_safe_mean,
    factors, registry, sign_null_group)
from skillnet_cohort.factorized_reward_analysis import metric_arrays_extended, scope
from skillnet_cohort.reward_variants import EPS, aggregation_weights


def frame():
    return pd.DataFrame({'control': ['placebo']*6, 'context_id': ['x']*6,
        'skill_id': ['s']*6, 'phase': ['initial']*6,
        'decision_id': ['a', 'a', 'b', 'c', 'd', 'e'],
        'trajectory_id': ['t1', 't1', 't1', 't2', 't3', 't3'],
        'game_id': ['g1', 'g1', 'g1', 'g1', 'g2', 'g2'],
        'delta_centered_norm': [3., 5., 4., 2., 7., 9.],
        'chosen_delta': [1., -2., 4., -3., 1., 2.],
        'advantage': [1., 1., 0., -2., 2., -1.]})


def test_registry_is_one_formula_and_controls():
    r = registry()
    assert len(r) == len({x['score'] for x in r}) == 12
    assert {x['name'] for x in r} == {'D_factor', 'D_orientation', 'D_factor_A1', 'D_orientation_A1'}
    assert all(not x['signed'] for x in r if x['mode'] == 'reward_free')


@pytest.mark.parametrize('agg', ['token', 'decision', 'game'])
def test_factors_match_direct_definition(agg):
    t = frame(); w = aggregation_weights(t, agg)
    f = factors(t.delta_centered_norm, t.chosen_delta, t.advantage, w)
    b = t.advantage*t.chosen_delta
    B = math.fsum(w*t.delta_centered_norm)
    R = math.fsum(w*b)/(math.fsum(w*abs(b))+EPS)
    assert f['B'] == pytest.approx(B)
    assert f['R'] == pytest.approx(R)
    assert f['D_factor'] == pytest.approx(-B*R)
    assert f['D_orientation'] == pytest.approx(-R)
    assert abs(f['D_factor']) <= B+1e-12
    assert np.sign(f['D_factor']) == np.sign(-f['b_mean'])


@pytest.mark.parametrize('a,chosen,expected_sign', [(1, 1, -1), (1, -1, 1), (-1, 1, 1), (-1, -1, -1)])
def test_direction_semantics(a, chosen, expected_sign):
    f = factors([2.], [chosen], [a], [1.])
    assert np.sign(f['D_factor']) == expected_sign


def test_cancellation_is_not_rectified():
    f = factors([10., 10.], [1., -1.], [2., 2.], [.5, .5])
    assert f['R'] == f['D_factor'] == 0.
    assert f['b_absolute_mean'] == 2.


def test_zero_advantage_remains_in_all_denominators():
    f = factors([10., 2.], [4., 1.], [0., 1.], [.5, .5])
    assert f['B'] == 6.
    assert f['b_mean'] == f['b_absolute_mean'] == .5
    assert f['A1_b_mean'] == 2.5


def test_entire_zero_advantage_keeps_real_A1():
    f = factors([10., 2.], [4., 1.], [0., 0.], [.5, .5])
    assert f['R'] == f['D_factor'] == 0.
    assert f['D_factor_A1'] < 0


def test_zero_magnitude_degeneracy():
    f = factors([0.], [2.], [3.], [1.])
    assert f['D_factor'] == 0.
    assert f['D_orientation'] < 0.


def test_reward_flip_is_odd_and_magnitude_even():
    t = frame(); w = aggregation_weights(t, 'game')
    a = factors(t.delta_centered_norm, t.chosen_delta, t.advantage, w)
    b = factors(t.delta_centered_norm, t.chosen_delta, -t.advantage, w)
    assert a['D_factor'] == -b['D_factor']
    assert a['B'] == b['B']
    assert a['D_factor_A1'] == b['D_factor_A1']


@pytest.mark.parametrize('value', [-1., 0., 1., 17.3])
def test_exact_constants_preserved(value):
    assert constant_safe_mean([value]*7, np.full(7, 1/7)) == value


@pytest.mark.parametrize('values,weights', [([], []), ([1.], [2.]), ([1., 2.], [2., -1.]), ([np.nan], [1.])])
def test_invalid_mean_rejected(values, weights):
    with pytest.raises(ValueError):
        constant_safe_mean(values, weights)


@pytest.mark.parametrize('agg', ['token', 'decision', 'game'])
def test_block_signs_match_full_row_recalculation(agg):
    t = frame(); ids = ['t1', 't2', 't3']
    signs = np.array([[1, 1, 1], [-1, -1, -1], [1, -1, 1], [-1, 1, -1]])
    actual = sign_null_group(t, agg, ids, signs)
    for i, signs_i in enumerate(signs):
        a = t.advantage*t.trajectory_id.map(dict(zip(ids, signs_i)))
        f = factors(t.delta_centered_norm, t.chosen_delta, a, aggregation_weights(t, agg))
        np.testing.assert_allclose(actual[i], [f['D_factor'], f['D_orientation']], rtol=1e-13, atol=1e-14)


def test_block_sign_constant_guard():
    t = frame(); t['advantage'] = 1.; t['chosen_delta'] = 1.
    out = sign_null_group(t, 'game', ['t1', 't2', 't3'], np.ones((2, 3)))
    assert out[0, 1] == -1./(1.+EPS)
    assert np.array_equal(out[0], out[1])


def test_aggregate_preserves_null_control_and_all_phases():
    t = frame(); t = pd.concat([t, t.assign(control='null')], ignore_index=True)
    scores, components = aggregate_scores(t)
    assert set(scores.control) == {'placebo', 'null'}
    assert set(scores.phase) == {'all', 'initial'}
    assert len(scores) == 48 and len(components) == 12
    assert components.factor_sign_matches_action_adv.all()
    assert scores.token_count.eq(6).all()


def test_balanced_accuracy_and_abstention():
    x = np.array([[-1., -1., -1.], [1., -1., 0.]])
    y = np.array([[1., -1., -2.], [0., 0., 0.]])
    m = metric_arrays_extended(x, y, 0.)
    np.testing.assert_allclose(m['balanced_accuracy_abstention_wrong'][0], [.5, .75])
    assert np.isnan(m['balanced_accuracy_abstention_wrong'][1]).all()


def test_does_not_collapse_to_mean_reward_work():
    f = factors([100., 1.], [1., -2.], [1., 1.], [.5, .5])
    assert f['D_factor'] != -f['b_mean']
    assert np.sign(f['D_factor']) == np.sign(-f['b_mean'])


def test_scope_does_not_allow_overwriting_prior_run(tmp_path):
    with pytest.raises(PermissionError):
        scope(tmp_path/'realized-reward-s404-v2')


def test_complete_synthetic_scalar_pipeline_and_no_retry(tmp_path, monkeypatch):
    from skillnet_cohort import factorized_reward_analysis as analysis
    from skillnet_cohort.common import write_new_bytes, write_new_json, read_json, file_hash
    from skillnet_cohort.realized_reward import registry as real_registry
    from skillnet_cohort.reward_variants import registry as old_registry
    from skillnet_cohort.reward_variant_analysis import point_tables, REFERENCE_COLUMNS
    from tests.skillnet_cohort.test_realized_reward import fixture_tables

    source, prior, draws, output = [tmp_path/x for x in ('source', 'prior', 'draws', 'factorized-reward-s404-test')]
    t = fixture_tables()
    t['group_id'] = t.game_id
    t['direction_valid'] = t.advantage.ne(0)
    t['delta_centered_norm'] = 2.+t.chosen_u_original.abs()
    t['chosen_delta'] = .5*t.chosen_u_original
    for col in ('P_int', 'C_upd', 'C_upd_centered', 'd_norm', 'delta_norm', 'u_original_norm'):
        t[col] = 1.
    write_new_bytes(source/'token_signals.parquet', t.to_parquet(index=False))
    metadata = old_registry()+[{'score': 'B_'+n, 'name': n, 'family': 'retained_reference',
        'aggregation': 'original', 'mode': 'reference', 'signed': False, 'geometry': False}
        for n in REFERENCE_COLUMNS]+real_registry()
    rows, labels, pools = [], [], {}
    for control in ('placebo', 'null'):
        pool = []
        for phase in ('all', 'initial'):
            pool.append({'context_id': 'all_alfworld', 'phase': phase, 'shared_skill_ids': ['s0', 's1', 's2']})
            for skill in range(3):
                common = {'control': control, 'context_id': 'all_alfworld', 'phase': phase, 'skill_id': f's{skill}'}
                labels.append({**common, 'delta_utility': [-.1, .1, 0.][skill]})
                g = t[(t.control == control)&(t.skill_id == f's{skill}')]
                for m in metadata:
                    value = float(skill)
                    if m['name'] in ('M_delta_centered', 'D_action_adv') and m['mode'] != 'unsigned':
                        f = factors(g.delta_centered_norm, g.chosen_delta, g.advantage,
                                    aggregation_weights(g, m['aggregation']))
                        value = f['B'] if m['name'] == 'M_delta_centered' else -f['b_mean']
                    rows.append({**common, 'score': m['score'], 'value': value})
        pools[control+'/stable_raw'] = pool
    old_scores, units = pd.DataFrame(rows), pd.DataFrame(labels)
    old_diag, _ = point_tables(old_scores, units, pools, metadata, [0., .05])
    for name, data in [('skill_scores.csv', old_scores), ('ranking_diagnostics.csv', old_diag), ('registry.csv', pd.DataFrame(metadata))]:
        analysis.csv(prior/name, data)
    write_new_json(source/'ranking-snapshots.json', {'candidate_pools': pools})
    write_new_bytes(source/'reused-labels/utility_units.parquet', units.to_parquet(index=False))
    write_new_bytes(source/'reports/coverage_and_effects.csv', b'skill_id\ns0\ns1\ns2\n')
    write_new_bytes(prior/'reports/phase2-results-expanded.md', b'Original report fixture')
    for control in ('placebo', 'null'):
        d = pd.DataFrame(np.tile([.1, -.1, 0.], (2000, 1)), columns=['s0', 's1', 's2'])
        d.iloc[0, 0] = np.nan
        write_new_bytes(draws/f'bootstrap-label-draws-{control}.parquet', d.to_parquet(index=False))
    rng = np.random.default_rng(100)
    masks = pd.DataFrame(1-2*rng.integers(0, 2, (512, 128)), columns=[f't{i:03d}' for i in range(128)])
    write_new_bytes(draws/'sign-null-trajectory-masks.parquet', masks.to_parquet(index=False))
    plan = {'expected_token_control_rows': len(t), 'expected_decisions': t.decision_id.nunique(),
            'event_thresholds': [0., .05], 'bootstrap_draws': 2000, 'preserved_prior_source_count': 0}
    plan['expected_decisions'] = int(plan['expected_decisions'])
    write_new_json(output/'plan.json', plan)
    monkeypatch.setattr(analysis, 'COHORT', tmp_path)
    monkeypatch.setattr(analysis, 'SOURCE', source)
    monkeypatch.setattr(analysis, 'PRIOR', prior)
    monkeypatch.setattr(analysis, 'DRAWS', draws)
    monkeypatch.setattr(analysis, 'binding', lambda *args: plan)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    analysis.run(output)
    complete = read_json(output/'complete.json')
    assert complete['score_columns'] == 285
    assert complete['metric_rows'] == 285*2*2*2
    assert read_json(output/'score-commit.json')['new_scoring_received_gold'] is False
    receipt = read_json(output/'independent-verification.json')
    assert receipt['ranking_metrics']['passed'] and receipt['direction_metrics']['passed']
    assert receipt['factor_sign_identity_passed']
    assert (output/'reports/phase2-results-expanded.md').read_bytes().startswith(b'Original report fixture')
    assert analysis.read_csv(output/'bootstrap_summary.csv').complete_pool_draws.eq(1999).all()
    for item in read_json(output/'provenance.json')['files']:
        assert file_hash(output/item['path']) == item['sha256']
    with pytest.raises(FileExistsError):
        analysis.run(output)
