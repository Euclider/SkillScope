from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from skillnet_cohort.common import file_hash, write_new_bytes, write_new_json, read_json
from skillnet_cohort.realized_reward import (EPS, FIELDS, KEYS, VERSION, aggregate_scores,
    assert_same_scalars, constant_safe_reduce, registry, vector_signals)
from skillnet_cohort import realized_reward_run as runner
from skillnet_cohort import realized_reward_analysis as analysis
from skillnet_cohort.reward_variants import aggregation_weights, registry as old_registry
from skillnet_cohort.reward_variant_analysis import point_tables, REFERENCE_COLUMNS


def vector_fixture(n=19):
    rng = torch.Generator().manual_seed(617)
    values = [torch.randn(n, 11, generator=rng, dtype=torch.float64).log_softmax(-1) for _ in range(4)]
    actions = torch.arange(n) % 11
    advantage = torch.linspace(-2, 2, n, dtype=torch.float64)
    return (*values, actions, advantage)


def test_vector_equation_direct_numpy():
    args = vector_fixture()
    actual = vector_signals(*args)
    old, new, oc, nc, ids, advantage = [x.numpy() for x in args]
    u = new-old; delta = u-(nc-oc)
    v = u-u.mean(1, keepdims=True); xi = delta-delta.mean(1, keepdims=True)
    coefficient = (v*xi).sum(1)/((v*v).sum(1)+EPS)
    q = advantage*u[np.arange(len(ids)), ids]
    np.testing.assert_allclose(actual['real_D'], -q*coefficient, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(actual['real_q'], q, rtol=1e-14, atol=0)


@pytest.mark.parametrize('chunk', [1, 7, 16, 32])
def test_chunk_invariance(chunk):
    args = vector_fixture()
    a, b = vector_signals(*args, chunk_size=chunk), vector_signals(*args, chunk_size=16)
    for key in FIELDS:
        assert torch.equal(a[key], b[key])


def test_centering_q_is_uncentered():
    args = list(vector_fixture())
    base = vector_signals(*args)
    args[1] = args[1]+3.
    shifted = vector_signals(*args)
    torch.testing.assert_close(base['real_projection_coefficient'], shifted['real_projection_coefficient'])
    torch.testing.assert_close(shifted['real_q']-base['real_q'], 3*args[-1])
    assert not torch.allclose(base['real_D'], shifted['real_D'])


def test_control_common_component_invariant():
    args = list(vector_fixture()); base = vector_signals(*args)
    args[3] = args[3]+7.
    shifted = vector_signals(*args)
    torch.testing.assert_close(base['real_D'], shifted['real_D'])


def test_zero_advantage_has_genuine_fullrow_A1():
    args = list(vector_fixture()); args[-1] = torch.zeros_like(args[-1])
    value = vector_signals(*args)
    assert value['real_D'].eq(0).all()
    assert value['real_A1'].ne(0).any()


def test_zero_update_is_finite_zero():
    args = list(vector_fixture()); args[1] = args[0].clone()
    value = vector_signals(*args)
    assert value['real_small_u'].all()
    assert value['real_D'].eq(0).all()
    assert value['real_projection_coefficient'].eq(0).all()


def test_constant_update_is_zero_centered_without_action_centering():
    old = torch.zeros(2, 4, dtype=torch.float64)
    v = vector_signals(old, old+2, old, old+1, torch.tensor([0, 1]), torch.tensor([1., -1.]))
    assert torch.equal(v['real_q'], torch.tensor([2., -2.], dtype=torch.float64))
    assert v['real_D'].eq(0).all()


def test_signed_reward_inverts_but_A1_does_not():
    args = list(vector_fixture()); plus = vector_signals(*args)
    args[-1] = -args[-1]; minus = vector_signals(*args)
    assert torch.equal(plus['real_D'], -minus['real_D'])
    assert torch.equal(plus['real_A1'], minus['real_A1'])
    assert torch.equal(plus['real_absA'], minus['real_absA'])
    assert plus['real_D'].gt(0).any() and plus['real_D'].lt(0).any()


@pytest.mark.parametrize('epsilon', [0., -1., np.nan, np.inf])
def test_invalid_epsilon_rejected(epsilon):
    with pytest.raises(ValueError):
        vector_signals(*vector_fixture(), epsilon=epsilon)


def test_invalid_identity_rejected():
    args = list(vector_fixture()); args[4] = args[4].float()
    with pytest.raises(ValueError):
        vector_signals(*args)


def test_nonfinite_rejected():
    args = list(vector_fixture()); args[0][0, 0] = np.nan
    with pytest.raises(ValueError):
        vector_signals(*args)


@pytest.mark.parametrize('agg', ['token', 'decision', 'game'])
def test_constants_exact(agg):
    t = pd.DataFrame({'decision_id': ['a', 'a', 'a', 'b', 'c', 'c', 'd'],
        'trajectory_id': ['a', 'a', 'a', 'b', 'c', 'c', 'c'], 'game_id': ['x']*4+['y']*3})
    vals = constant_safe_reduce(aggregation_weights(t, agg), np.full((len(t), 4), -1.))
    assert np.array_equal(vals, np.full(4, -1.))


def test_registry_is_one_formula_with_controls():
    rows = registry()
    assert len(rows) == 12 and len({r['score'] for r in rows}) == 12
    assert sum(r['signed'] for r in rows) == 3
    assert {r['name'] for r in rows} == {'D_real', 'D_real_A1', 'D_real_absA', 'D_real_projection_only'}


def test_adapter_preserves_original_module_and_matched_backend():
    from skillnet_cohort.realized_reward_measure import adapter
    from skillnet_cohort import first_calls_measure as original
    original_forward, original_score = original.forward, original.token_signals
    config = {'signals': {'tau_C': 0., 'epsilon': EPS}, '_frozen_tau_delta': 1e-8}
    with adapter(config) as calls:
        for i in range(4):
            value = original.token_signals(*vector_fixture(3))
            assert ('real_D' in value) == (i % 2 == 0)
            assert 'legacy_P_int' in value
        assert len(calls) == 4
    assert original.forward is original_forward and original.token_signals is original_score


def test_adapter_restores_on_error():
    from skillnet_cohort.realized_reward_measure import adapter
    from skillnet_cohort import first_calls_measure as original
    saved = original.forward, original.token_signals
    with pytest.raises(RuntimeError):
        with adapter({}):
            raise RuntimeError('synthetic failure')
    assert (original.forward, original.token_signals) == saved


@pytest.mark.parametrize('wave', [0, 1])
def test_gpu_pairs_disjoint(wave):
    pairs = [runner.gpu_pair(i) for i in range(4*wave, 4*wave+4)]
    assert pairs == ['0,1', '2,3', '4,5', '6,7']


@pytest.mark.parametrize('shard', [-1, 8, 0.5, None])
def test_wrong_shard_rejected(shard):
    with pytest.raises(ValueError):
        runner.gpu_pair(shard)


def test_worker_environment_no_api_or_training_command():
    env = runner.worker_env(2)
    assert env['CUDA_VISIBLE_DEVICES'] == '4,5' and env['HF_HUB_OFFLINE'] == '1'
    assert runner.worker_env()['CUDA_VISIBLE_DEVICES'] == ''
    with pytest.raises(ValueError):
        runner.command('train', Path('/tmp'))


def test_scope_rejects_other_seed():
    with pytest.raises(PermissionError):
        runner.validate_scope({'seed': 505}, Path('/tmp'))


def test_artifacts_no_clobber(tmp_path):
    p = tmp_path/'a.json'; write_new_json(p, {'a': 1})
    write_new_json(p, {'a': 1})
    with pytest.raises(FileExistsError):
        write_new_json(p, {'a': 2})
    with pytest.raises(PermissionError):
        runner.binding(tmp_path)


def test_provenance_excludes_live_and_temporary_files(tmp_path):
    for name in ('complete-data.json', '.publish-example', 'workflow.log',
                 'heartbeats/analysis-000001.json', 'logs/shard-1.log', 'reports/result.md'):
        write_new_bytes(tmp_path/name, b'fixture')
    assert [str(p.relative_to(tmp_path)) for p in analysis.sealed_artifacts(tmp_path)] == [
        'complete-data.json', 'reports/result.md']


def fixture_tables():
    rows = []
    for arm in ('placebo', 'null'):
        for tr in range(128):
            for skill in range(3):
                a = float((tr % 3)-1); chosen = (skill+1)*.01+(tr-64)*.003
                coefficient = 1.+.2*skill
                rows.append({'control': arm, 'context_id': 'all_alfworld', 'phase': 'initial',
                    'decision_id': f'd{tr}-{skill}', 'trajectory_id': f't{tr:03d}', 'game_id': f'g{tr//8}',
                    'skill_id': f's{skill}', 'response_token_offset': 0, 'advantage': a,
                    'chosen_u_original': chosen, 'real_q': a*chosen,
                    'real_u_centered_norm_sq': 2., 'real_u_delta_centered_dot': coefficient*(2+EPS),
                    'real_projection_coefficient': coefficient, 'real_D': -a*chosen*coefficient,
                    'real_A1': -chosen*coefficient, 'real_absA': -abs(a)*chosen*coefficient,
                    'real_projection_only': -coefficient, 'real_small_u': False})
    return pd.DataFrame(rows)


def test_duplicate_tokens_rejected():
    t = fixture_tables()
    with pytest.raises(ValueError):
        aggregate_scores(pd.concat([t, t.iloc[:1]]))


def test_zero_rows_retained_and_all_phases():
    t = fixture_tables(); scores = aggregate_scores(t)
    assert set(scores.phase) == {'all', 'initial'}
    assert set(scores.token_count) == {128}
    assert set(scores.game_count) == {16}


def test_full_synthetic_analysis_and_sealing(tmp_path, monkeypatch):
    """All eight shards -> score commit -> 273 columns -> bootstrap/null/report."""
    output, prior, source = [tmp_path/x for x in ('output', 'prior', 'source')]
    output.mkdir(); prior.mkdir(); source.mkdir()
    t = fixture_tables()
    base = t.drop(columns=list(FIELDS))
    write_new_bytes(source/'token_signals.parquet', base.to_parquet(index=False))
    for shard in range(8):
        ids = sorted(t.decision_id.unique())[shard::8]
        rows = t[t.decision_id.isin(ids)][KEYS+list(FIELDS)]
        write_new_bytes(output/f'tokens-shard-{shard}.parquet', rows.to_parquet(index=False))
        write_new_json(output/f'input-audit-shard-{shard}.json', [])
        write_new_bytes(output/f'witness-shard-{shard}.pt', b'fixture-not-a-real-model')
        write_new_json(output/f'shard-{shard}.json', {
            'tokens_sha256': file_hash(output/f'tokens-shard-{shard}.parquet'),
            'input_audit_sha256': file_hash(output/f'input-audit-shard-{shard}.json'),
            'witness_sha256': file_hash(output/f'witness-shard-{shard}.pt'),
            'all_stable_scalar_signals_exact': True})
    old_meta = old_registry()+[{'score': 'B_'+n, 'name': n, 'family': 'retained_reference',
        'aggregation': 'original', 'mode': 'reference', 'signed': False, 'geometry': False} for n in REFERENCE_COLUMNS]
    old_rows, labels, pools = [], [], {}
    for control in ('placebo', 'null'):
        pool = []
        for phase in ('all', 'initial'):
            pool.append({'context_id': 'all_alfworld', 'phase': phase, 'shared_skill_ids': ['s0', 's1', 's2']})
            for skill in range(3):
                labels.append({'control': control, 'context_id': 'all_alfworld', 'phase': phase,
                    'skill_id': f's{skill}', 'delta_utility': [-.1, .1, 0][skill]})
                for m in old_meta:
                    old_rows.append({'control': control, 'context_id': 'all_alfworld', 'phase': phase,
                        'skill_id': f's{skill}', 'score': m['score'], 'value': float(skill)})
        pools[control+'/stable_raw'] = pool
    old_scores = pd.DataFrame(old_rows); units = pd.DataFrame(labels)
    old_diag, _ = point_tables(old_scores, units, pools, old_meta, [0., .05])
    analysis.csv(prior/'skill_scores.csv', old_scores)
    analysis.csv(prior/'ranking_diagnostics.csv', old_diag)
    write_new_json(prior/'plan.json', {'registry': old_registry()})
    write_new_json(source/'ranking-snapshots.json', {'candidate_pools': pools})
    write_new_bytes(source/'reused-labels/utility_units.parquet', units.to_parquet(index=False))
    write_new_bytes(source/'reports/coverage_and_effects.csv', b'skill_id\ns0\ns1\ns2\n')
    write_new_bytes(prior/'reports/phase2-results-expanded-v2.md', b'Original report fixture')
    for control in ('placebo', 'null'):
        draws = pd.DataFrame(np.tile([.1, -.1, 0.], (2000, 1)), columns=['s0', 's1', 's2'])
        draws.iloc[0, 0] = np.nan
        write_new_bytes(prior/f'bootstrap-label-draws-{control}.parquet', draws.to_parquet(index=False))
    rng = np.random.default_rng(100)
    masks = pd.DataFrame(1-2*rng.integers(0, 2, (512, 128)), columns=[f't{i:03d}' for i in range(128)])
    write_new_bytes(prior/'sign-null-trajectory-masks.parquet', masks.to_parquet(index=False))
    plan = {'source': str(source), 'prior': str(prior), 'expected_token_control_rows': len(t),
        'event_thresholds': [0., .05]}
    write_new_json(output/'plan.json', plan)
    monkeypatch.setattr(runner, 'binding', lambda *a, **k: plan)
    analysis.analyze(output)
    complete = read_json(output/'complete.json')
    assert complete['all_score_columns'] == 273
    assert complete['metric_rows'] == 273*2*2*2
    assert read_json(output/'independent-verification.json')['passed']
    assert read_json(output/'score-commit.json')['gold_read_for_new_scoring'] is False
    assert (output/'reports/phase2-results-expanded.md').read_bytes().startswith(b'Original report fixture')
    di = analysis.read_csv(output/'direction-confusions.csv')
    assert di.loc[~di.signed, 'balanced_accuracy_abstention_wrong'].isna().all()
    boot = analysis.read_csv(output/'bootstrap_summary.csv')
    assert boot.complete_pool_draws.eq(1999).all()
    with pytest.raises(FileExistsError):
        analysis.analyze(output)
