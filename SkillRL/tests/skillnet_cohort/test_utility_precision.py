"""Synthetic CPU-only protocol/reuse/statistics regression. No live rollout."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from skillnet_cohort.common import file_hash, read_json, write_new_bytes, write_new_json
from skillnet_cohort.utility_precision import (COHORT, GOLD, NEW_GOLD, OLD_GOLD,
    expanded_config, validate_expansion, jobs_with_ids, import_endpoint, scope, check_tests)
from skillnet_cohort.utility_precision_analysis import (subset_frame, repeat_diagnostics,
    analyze_frame, uncertainty)


def config(tmp_path):
    anchors = [{'anchor_id': f'a{i}', 'skill_id': 's', 'context_id': 'x',
        'game_id': f'g{i}', 'source_trajectory_id': f't{i}', 'state_id': f'st{i}',
        'trigger_step': 1, 'environment_seed': 77, 'source_eval_seed': 98,
        'prefix_actions': ['look'], 'max_steps': 50} for i in range(2)]
    ap = tmp_path/'anchors.jsonl'; pp = tmp_path/'placebo.json'
    write_new_bytes(ap, ''.join(json.dumps(a)+'\n' for a in anchors).encode())
    write_new_json(pp, {'text': 'neutral'})
    return {'root': str(tmp_path/'source'), 'run_id': 'r', 'rl_path_id': 'path404',
        'evaluation': {'gold_seeds': list(OLD_GOLD), 'old_evidence_seeds': [62011],
            'arms': ['original', 'placebo', 'null'], 'shards': 8, 'anchor_count': 2,
            'max_steps': 50, 'temperature': .4, 'anchor_sets': [{'skill_id': 's', 'context_id': 'x',
                'anchors_path': str(ap), 'anchors_sha256': file_hash(ap),
                'placebo_path': str(pp), 'placebo_sha256': file_hash(pp)}]},
        'runtime': {'router_backend': 'skillrl_embedding_state_batch', 'max_api_calls': 0,
            'cache_path': str(tmp_path/'source/router.sqlite3'), 'max_local_calls': 1800,
            'authorization_path': str(tmp_path/'source/permit.json')}}


def source_endpoint(tmp_path, update=0):
    old = config(tmp_path); source = tmp_path/'source'
    write_new_json(source/'protocol.json', old)
    grouped = [[] for _ in range(8)]
    for rank, tid, identity, job in jobs_with_ids(old, update):
        anchor = job[1]; path = source/'evaluations'/f'u{update:04d}'/'trajectories'/'s'/(tid+'.json')
        row = {**identity, 'trajectory_id': tid, 'trajectory_path': str(path),
            'game_id': anchor['game_id'], 'context_id': 'x', 'phase': 'early', 'trigger_step': 1,
            'success': True, 'prefix_replay_verified': True}
        write_new_json(path, {**row, 'steps': [{'reward': 1}], 'actual_continuation_seed': identity['continuation_seed'],
                              'original_anchor': anchor})
        grouped[rank].append(row)
    for rank, rows in enumerate(grouped):
        directory = source/'evaluations'/f'u{update:04d}'
        write_new_bytes(directory/f'shard-{rank}.jsonl', ''.join(json.dumps(r)+'\n' for r in rows).encode())
        write_new_json(directory/f'shard-{rank}-complete.json', {'jobs': len(rows), 'shard': rank,
            'shards': 8, 'max_jobs': None, 'protocol_sha256': file_hash(source/'protocol.json')})
    return source, old


def test_exact_repeat_registry():
    assert len(GOLD) == len(set(GOLD)) == 8
    assert GOLD[:2] == OLD_GOLD and len(NEW_GOLD) == 6 and NEW_GOLD[0] == 404
    assert 62011 not in GOLD


@pytest.mark.parametrize('seed', NEW_GOLD)
def test_new_step_rng_ranges_do_not_overlap(seed):
    for other in GOLD:
        if other != seed:
            assert abs(seed-other) >= 50


def test_config_does_not_mutate_original(tmp_path):
    old = config(tmp_path); before = copy.deepcopy(old)
    result = expanded_config(old, tmp_path/'new')
    assert old == before and result['evaluation']['gold_seeds'] == list(GOLD)
    assert result['evaluation']['anchor_sets'] == old['evaluation']['anchor_sets']
    assert result['runtime']['max_local_calls'] == 2*2*3*6*50
    validate_expansion(old, result, tmp_path/'new')


@pytest.mark.parametrize('change', ['seed', 'temperature', 'arms', 'anchor', 'api', 'router', 'run_id'])
def test_scope_rejects_unrequested_changes(tmp_path, change):
    old = config(tmp_path); new = expanded_config(old, tmp_path/'new')
    if change == 'seed': new['evaluation']['gold_seeds'][-1] += 1
    if change == 'temperature': new['evaluation']['temperature'] = .8
    if change == 'arms': new['evaluation']['arms'] = ['original', 'placebo']
    if change == 'anchor': new['evaluation']['anchor_sets'][0]['anchors_sha256'] = 'bad'
    if change == 'api': new['runtime']['max_api_calls'] = 100
    if change == 'router': new['runtime']['router_backend'] = 'external_llm'
    if change == 'run_id': new['run_id'] = 'new-rl'
    with pytest.raises(ValueError, match='Only gold'):
        validate_expansion(old, new, tmp_path/'new')


def test_scope_rejects_other_seed_or_parent():
    assert scope(COHORT/'utility-precision-s404-v1').parent == COHORT
    for path in (COHORT/'utility-precision-s505-v1', COHORT.parent/'utility-precision-s404-v1'):
        with pytest.raises(PermissionError): scope(path)


def test_endpoint_identity_preserved_and_new_jobs_paired(tmp_path):
    old = config(tmp_path); new = expanded_config(old, tmp_path/'new')
    for update in (0, 5):
        a = {r[1] for r in jobs_with_ids(old, update)}
        b = {r[1] for r in jobs_with_ids(new, update)}
        assert a < b and len(b-a) == 2*6*3
    counts = {}
    for update in (0, 5):
        for _, tid, identity, job in jobs_with_ids(new, update):
            key = (identity['anchor_id'], identity['purpose'], identity['continuation_seed'])
            counts.setdefault(key, set()).add((update, identity['arm']))
            assert job[1]['environment_seed'] == 77 and job[1]['source_eval_seed'] == 98
    assert all(len(v) == 6 for v in counts.values())


@pytest.mark.parametrize('update', [0, 5])
def test_import_is_audited_non_destructive_and_repartitioned(tmp_path, update):
    from skillnet_cohort.first_calls_recovery import audit_endpoint
    source, old = source_endpoint(tmp_path, update)
    before = {str(p): file_hash(p) for p in source.rglob('*') if p.is_file()}
    output = tmp_path/'new'; new = expanded_config(old, output)
    write_new_json(output/'protocol.json', new)
    import_endpoint(source, output, old, new, update)
    audit = audit_endpoint(output, update)
    assert audit['completed'] == 18 and audit['missing'] == 36
    assert audit['complete_shards'] == []
    assert before == {str(p): file_hash(p) for p in source.rglob('*') if p.is_file()}
    assert read_json(output/'reuse'/f'u{update:04d}.json')['reused'] == 18


def test_import_rejects_corrupt_old_trajectory(tmp_path):
    source, old = source_endpoint(tmp_path)
    path = next((source/'evaluations/u0000/trajectories/s').glob('*.json'))
    # Simulate corruption only in a pytest-owned temporary fixture.
    path.write_text('{}')
    output = tmp_path/'new'; new = expanded_config(old, output)
    with pytest.raises(ValueError, match='identity/replay'):
        import_endpoint(source, output, old, new, 0)


def panel():
    anchors, rows = [], []
    for i, skill in enumerate(['s1', 's2']):
        for game in range(2):
            aid = f'{skill}-{game}'
            anchors.append({'skill_id': skill, 'anchor_id': aid, 'source_trajectory_id': aid})
            for update in (0, 5):
                for seed in [62011, *GOLD]:
                    for arm in ('original', 'placebo', 'null'):
                        success = (arm == 'original') and ((update == 0) == (i == 0))
                        rows.append({'skill_id': skill, 'anchor_id': aid, 'game_id': f'g{game}',
                            'context_id': 'x', 'phase': 'early', 'trigger_step': 1, 'update': update,
                            'purpose': 'evidence' if seed == 62011 else 'gold',
                            'continuation_seed': seed, 'arm': arm, 'success': success})
    return pd.DataFrame(rows), anchors


@pytest.mark.parametrize('count', [2, 4, 8])
def test_subset_keeps_evidence_disjoint(count):
    frame, _ = panel(); q = subset_frame(frame, GOLD[:count])
    assert q[q.purpose == 'gold'].continuation_seed.nunique() == count
    assert set(q[q.purpose == 'evidence'].continuation_seed) == {62011}


def test_repeat_diagnostics_preserves_pairing():
    from phase2.utilities import margins
    frame, _ = panel(); repeats, stability = repeat_diagnostics(margins(frame))
    assert set(repeats[repeats.skill_id == 's1'].delta_utility) == {-1.}
    assert set(repeats[repeats.skill_id == 's2'].delta_utility) == {1.}
    assert set(stability.repeat_count) == {8}
    assert stability.opposite_sign_fraction.eq(0).all()


def test_missing_endpoint_is_not_silently_dropped():
    from phase2.utilities import margins
    frame, _ = panel(); frame = frame[~((frame['update'] == 5) & (frame.anchor_id == 's1-0'))]
    with pytest.raises(ValueError, match='endpoint pair'):
        repeat_diagnostics(margins(frame))


def score_fixture():
    metadata = [{'score': name, 'mode': 'reward' if name.startswith('D_') else 'magnitude',
        'name': name.split('::')[0], 'aggregation': 'token', 'signed': name.startswith('D_')}
        for name in ('D_original::token::reward', 'M_delta_centered::token::magnitude')]
    scores = pd.DataFrame([{'control': c, 'context_id': 'x', 'phase': 'all', 'skill_id': s,
        'score': m['score'], 'value': v} for c in ('placebo', 'null') for m in metadata
        for s, v in [('s1', 1.), ('s2', -1.)]])
    pools = {c+'/stable_raw': [{'context_id': 'x', 'phase': 'all', 'shared_skill_ids': ['s1', 's2']}]
             for c in ('placebo', 'null')}
    return scores, pools, metadata


def test_analysis_and_paired_bootstrap_end_to_end():
    frame, anchors = panel(); scores, pools, metadata = score_fixture()
    units, games, margins, diag, budgets = analyze_frame(frame, anchors, scores, pools, metadata, 20)
    assert diag.average_precision.eq(1).all() and diag.auroc_decline_vs_increase.eq(1).all()
    plan = {'bootstrap_repetitions': 20, 'bootstrap_rng_seed': 7, 'event_thresholds': [0., .05]}
    summary, paired, stability, draws, receipts = uncertainty(scores, units, margins, pools, metadata, plan)
    assert paired.point_difference.eq(0).all()
    assert set(draws) == {'placebo', 'null'} and len(draws['placebo']) == 20
    assert receipts['placebo']['continuation_seeds'] == sorted(GOLD)
    assert stability[stability.skill_id == 's1'].bootstrap_fraction_decline.eq(1).all()
    assert not summary.empty and not budgets.empty


def test_zero_threshold_identifiability_separate_from_practical_direction():
    frame, anchors = panel(); scores, pools, metadata = score_fixture()
    units, *_ = analyze_frame(frame, anchors, scores, pools, metadata, 20)
    assert units[units.skill_id == 's1'].direction_at_zero.eq('decline_resolved').all()
    assert units[units.skill_id == 's2'].direction_at_zero.eq('increase_resolved').all()
    assert units.degenerate_interval.all()


def test_single_game_interval_stays_undefined_not_resolved():
    frame, anchors = panel(); scores, pools, metadata = score_fixture()
    frame = frame[frame.game_id == 'g0']
    units, *_ = analyze_frame(frame, anchors, scores, pools, metadata, 20)
    assert units.direction_at_zero.eq('interval_undefined').all()
    assert units.ci_low.isna().all() and not units.degenerate_interval.any()


@pytest.mark.parametrize('corrupt', [False, True])
def test_historical_statistics_guard(tmp_path, monkeypatch, corrupt):
    from skillnet_cohort import utility_precision_analysis as analysis
    frame, anchors = panel(); scores, pools, metadata = score_fixture()
    units, _, _, diagnostics, _ = analyze_frame(subset_frame(frame, GOLD[:2]), anchors, scores, pools, metadata, 20)
    (tmp_path/'window_metrics').mkdir()
    units.to_parquet(tmp_path/'window_metrics/utility_units.parquet', index=False)
    diagnostics.to_csv(tmp_path/'ranking_diagnostics.csv', index=False)
    monkeypatch.setattr(analysis, 'SOURCE', tmp_path)
    monkeypatch.setattr(analysis, 'SCORES', tmp_path)
    if corrupt:
        units.loc[0, 'delta_utility'] += .1
        with pytest.raises(ValueError, match='did not reproduce'):
            analysis.verify_historical(units, diagnostics)
    else:
        checked = analysis.verify_historical(units, diagnostics)
        assert checked['original_ranking_rows_reproduced'] == len(diagnostics)


@pytest.mark.parametrize('bad', ['failures', 'errors', 'skipped', 'few'])
def test_test_gate_fails_closed(tmp_path, bad):
    path = tmp_path/'tests.xml'
    path.write_text(f'<testsuites><testsuite tests="{2 if bad == "few" else 20}" '
                    f'{bad if bad != "few" else "failures"}="{0 if bad == "few" else 1}"/></testsuites>')
    with pytest.raises(ValueError): check_tests(path)


def test_test_gate_accepts(tmp_path):
    path = tmp_path/'tests.xml'; path.write_text('<testsuites><testsuite tests="20" failures="0"/></testsuites>')
    check_tests(path)
