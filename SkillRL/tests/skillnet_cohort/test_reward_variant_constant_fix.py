import json

import numpy as np
import pandas as pd

from skillnet_cohort import reward_variant_constant_fix as fix
from skillnet_cohort import reward_variants as rv
from tests.skillnet_cohort.test_reward_variants import tokens, synthetic_sources


def test_constant_counterpart_exact_for_variable_lengths_and_all_aggregations():
    t = tokens()
    parts = []
    for _, row in t.iterrows():
        count = {'d0': 7, 'd1': 31, 'd2': 49, 'd3': 103}[row.decision_id]
        part = pd.DataFrame([row.to_dict()]*count)
        part['response_token_offset'] = np.arange(count)
        parts.append(part)
    t = pd.concat(parts, ignore_index=True)
    scales = {c: rv.label_blind_scales(t[t.control == c]) for c in ('placebo', 'null')}
    scores, _ = fix.correct_scores(t, scales)
    q = scores[scores.score.str.startswith('D_reward_only::') & scores.score.str.endswith('::unsigned')]
    assert len(q) == 24
    assert np.array_equal(q.value.to_numpy(), np.full(len(q), -1.))


def test_only_provably_constant_columns_changed():
    t = tokens(); scales = {c: rv.label_blind_scales(t[t.control == c]) for c in ('placebo', 'null')}
    before, _ = fix.ORIGINAL_COMPUTE(t, scales)
    after, _ = fix.correct_scores(t, scales)
    for i in np.flatnonzero(before.value.to_numpy() != after.value.to_numpy()):
        row = after.iloc[i]
        q = t[(t.control == row.control)&(t.skill_id == row.skill_id)]
        if row.phase != 'all':
            q = q[q.phase == row.phase]
        name, aggregation, mode = row.score.split('::')
        matrix = rv.token_matrix(q, scales[row.control], mode)
        col = [r.name for r in rv.RECIPES].index(name)
        assert np.min(matrix[:, col]) == np.max(matrix[:, col]) == row.value


def test_sign_null_restores_exact_constant_across_different_trajectories():
    nr = len(rv.RECIPES)
    pl = np.full((2, nr), -1.); ph = pl.copy()
    ml = np.full((2, nr), 1.); mh = ml.copy()
    fix.BOUNDS[('placebo', 'context', 'skill')] = (pl, ph, ml, mh)
    cube = np.full((3, len(rv.AGGREGATIONS)*nr, 1), -.9999999999999998)
    flipped = np.array([[0, 0], [1, 1], [0, 1]])
    fix.correct_null_constants(cube, flipped, 'placebo', {'shared_skill_ids': ['skill'], 'context_id': 'context'})
    assert np.all(cube[0] == -1.)
    assert np.all(cube[1] == 1.)
    assert np.all(cube[2] == -.9999999999999998)  # A mixed nonconstant row is untouched.


def test_v2_end_to_end_constant_baseline_has_chance_ranking(tmp_path, monkeypatch):
    source = synthetic_sources(tmp_path)
    output = tmp_path/'output'; output.mkdir()
    plan = {'registry': rv.registry(), 'bootstrap_repetitions': 20, 'bootstrap_rng_seed': 123,
        'sign_null_repetitions': 8, 'sign_null_rng_seed': 345, 'event_thresholds': [0., .05],
        'inputs': [], 'preserved_runtime_sources': [], 'analysis_sources': [],
        'setting_document': {'path': str(source/'reports/phase2-results.md'),
            'sha256': fix.file_hash(source/'reports/phase2-results.md')}}
    (output/'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(fix.base, 'SOURCE', source)
    monkeypatch.setattr(fix.base, 'check_plan', lambda p: plan)
    monkeypatch.setattr(fix.base, 'compute_scores', fix.base.compute_scores)
    monkeypatch.setattr(fix.base, 'sign_null_analysis', fix.base.sign_null_analysis)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    fix.run(output)
    d = pd.read_csv(output/'ranking_diagnostics.csv', keep_default_na=False, na_values=[''])
    q = d[(d.control == 'placebo') & d.score.str.startswith('D_reward_only::') & d.score.str.endswith('::unsigned')]
    np.testing.assert_array_equal(q.average_precision.to_numpy(), q.declines/q.candidates)
    np.testing.assert_array_equal(q.auroc_decline_vs_rest, np.full(len(q), .5))
    assert q.spearman.isna().all()
