"""CPU-only amendment tests. No optimizer, real environment, API or GPU use."""
import copy
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from skillnet_cohort.common import REPO, file_hash, read_json, write_new_bytes, write_new_json
from skillnet_cohort.first_calls_support import call_anchor, training_coverage


def toy_trajectory():
    return {'trajectory_id': 'trajectory', 'game_id': 'game', 'task_description': 'task',
        'steps': [{'step_index': i, 'selected_skill_id': skill, 'observation': f'obs{i}',
            'projected_action': f'act{i}', 'admissible_actions': [f'act{i}'], 'reward': 0.}
            for i, skill in enumerate(['a', 'b', 'a', 'b', 'b'])]}


def test_only_first_invocation_per_skill_and_trajectory():
    t = toy_trajectory(); index = {'max_steps': 50, 'split': 'valid_unseen',
        'environment_seed': 404, 'eval_seed': 1, 'checkpoint_id': 'u0000', 'context_id': 'all_alfworld'}
    a = call_anchor(t, '/source.json', index, 0)
    b = call_anchor(t, '/source.json', index, 1)
    assert a['anchor_id'] != b['anchor_id'] and a['remaining_steps'] == 50
    assert b['prefix_actions'] == ['act0'] and b['remaining_steps'] == 49
    for step in (2, 3, 4):
        with pytest.raises(ValueError, match='Later target'):
            call_anchor(t, '/source.json', index, step)


@pytest.mark.parametrize('skill,step', [('a', 0), ('b', 1)])
def test_first_anchor_is_semantically_identical_to_legacy(skill, step):
    from phase1.first_invocation import build_anchor
    from skillnet_cohort.first_calls_reuse import ANCHOR_SEMANTIC_FIELDS
    t = toy_trajectory(); index = {'max_steps': 50, 'split': 'valid_unseen',
        'environment_seed': 404, 'eval_seed': 1, 'checkpoint_id': 'u0000', 'context_id': 'all_alfworld'}
    old = build_anchor(trajectory=t, trajectory_path='/source.json', skill_id=skill, source_index=index)
    new = call_anchor(t, '/source.json', index, step)
    assert all(old.get(k) == new.get(k) for k in ANCHOR_SEMANTIC_FIELDS)


def test_training_coverage_does_not_deduplicate_real_later_decisions(tmp_path):
    metas = [{'decision_id': name, 'trajectory_id': 't', 'environment_step': step,
        'info': {'selected_skill_id': skill, 'extra.gamefile': 'g'}}
        for name, step, skill in [('a0', 0, 'a'), ('a1', 1, 'a'), ('a1', 1, 'a'), ('b2', 2, 'b')]]
    batch = {'metadata': metas, 'tensors': {'phase2_actual_loss_mask': torch.tensor([[1, 0]]*4),
        'advantages': torch.tensor([[1., float('nan')], [2., 0.], [2., 0.], [0., 0.]])}}
    torch.save(batch, tmp_path/'batch.pt')
    rows = {(r['skill_id'], r['phase']): r for r in training_coverage(tmp_path/'batch.pt', {'a', 'b', 'c'})}
    assert rows['a', 'all']['train_decisions'] == 2
    assert rows['a', 'all']['train_nonzero_advantage_decisions'] == 2
    assert not rows['a', 'all']['legacy_count_thresholds_met']
    assert rows['b', 'all']['readout_input_available']
    assert not rows['b', 'all']['reward_direction_observed']
    assert not rows['c', 'all']['readout_input_available']


def test_training_coverage_rejects_nonfinite_actual_advantage(tmp_path):
    torch.save({'metadata': [{'decision_id': 'd', 'trajectory_id': 't', 'environment_step': 0,
        'info': {'selected_skill_id': 'a', 'extra.gamefile': 'g'}}],
        'tensors': {'phase2_actual_loss_mask': torch.ones(1, 1), 'advantages': torch.full((1, 1), float('nan'))}}, tmp_path/'b.pt')
    with pytest.raises(ValueError, match='Nonfinite'):
        training_coverage(tmp_path/'b.pt', {'a'})


def test_no_count_filter_does_not_turn_missing_readout_into_zero():
    from phase2.aggregate import aggregate_features, MEAN_SIGNALS
    records = []
    for skill, advantage in [('one', 1.), ('zero', 0.)]:
        for control in ('placebo', 'null'):
            records.append({**{key: 0. for key in MEAN_SIGNALS}, 'P_int': -advantage, 'C_upd': 1.,
                'delta_norm': 1., 'advantage': advantage, 'skill_id': skill, 'context_id': 'all_alfworld',
                'decision_id': skill, 'game_id': 'g', 'trajectory_id': 't', 'phase': 'initial',
                'control': control, 'direction_valid': bool(advantage), 'fidelity_valid': True})
    frame = pd.DataFrame(records)
    config = {'signals': {'minimum_nonzero_advantage_decisions': 0, 'minimum_training_games': 0,
        'minimum_training_trajectories': 0}, 'evaluation': {'anchor_sets': [
            {'skill_id': s, 'context_id': 'all_alfworld'} for s in ('one', 'zero', 'missing')]}}
    feature, _ = aggregate_features(frame, frame.copy(), config, 5, 1e-8)
    q = feature[(feature.phase == 'all') & (feature.control == 'placebo')].set_index('skill_id')
    assert q.loc['one', 'supported'] and q.loc['one', 'D_contribution'] == 1.
    assert q.loc['zero', 'supported'] and q.loc['zero', 'D_contribution'] == 0.
    assert not q.loc['missing', 'supported'] and pd.isna(q.loc['missing', 'D_contribution'])


def outcome_frame(games=('g1', 'g2')):
    rows, anchors = [], []
    for i, game in enumerate(games):
        aid = f'a{i}'; anchors.append({'skill_id': 's', 'anchor_id': aid, 'source_trajectory_id': f't{i}'})
        for update in (0, 5):
            for seed in (11, 21):
                for arm in ('original', 'placebo', 'null'):
                    rows.append({'update': update, 'purpose': 'gold', 'skill_id': 's', 'anchor_id': aid,
                        'game_id': game, 'context_id': 'all_alfworld', 'phase': 'initial', 'trigger_step': 0,
                        'continuation_seed': seed, 'arm': arm, 'success': int(arm == 'original' and update == 0)})
    return pd.DataFrame(rows), anchors


def test_single_game_is_not_filtered_and_has_no_population_interval():
    from skillnet_cohort.first_calls_report import clustered_units
    frame, anchors = outcome_frame(('g',))
    units, _, _ = clustered_units(frame, anchors, repetitions=30)
    assert len(units) == 4 and units.delta_utility.eq(-1).all()
    assert units.ci_low.isna().all() and units.direction.eq('uncertain').all()
    assert units.anchor_count.eq(1).all() and units.game_count.eq(1).all()


def test_paired_all_first_call_bootstrap_retains_both_endpoints_and_seed_pairs():
    from skillnet_cohort.first_calls_report import clustered_units
    from phase2.utilities import _units_from_margins
    frame, anchors = outcome_frame()
    new, _, m = clustered_units(frame, anchors, repetitions=50)
    old, _, _ = _units_from_margins(m, [{'start': 0, 'end': 5, 'role': 'test'}], repetitions=50)
    cols = ['skill_id', 'control', 'phase', 'utility_old', 'utility_new', 'delta_utility', 'ci_low', 'ci_high']
    pd.testing.assert_frame_equal(new[cols], old[cols])


def test_repeated_source_skill_anchor_rejected():
    from skillnet_cohort.first_calls_report import clustered_units
    frame, anchors = outcome_frame()
    anchors[1]['source_trajectory_id'] = anchors[0]['source_trajectory_id']
    with pytest.raises(ValueError, match='one first-call'):
        clustered_units(frame, anchors)


@pytest.mark.parametrize('damage', ['arm', 'endpoint', 'source'])
def test_utility_pairing_integrity_is_not_a_quantity_filter(damage):
    from skillnet_cohort.first_calls_report import clustered_units
    frame, anchors = outcome_frame()
    if damage == 'arm':
        frame = frame.iloc[1:]
    elif damage == 'endpoint':
        frame = frame[~((frame['update'] == 5) & (frame.anchor_id == 'a0'))]
    else:
        anchors[0]['anchor_id'] = 'wrong'
    with pytest.raises(ValueError):
        clustered_units(frame, anchors, repetitions=10)


def publication(tmp_path):
    root = tmp_path/'amendment/seed-404'; target = tmp_path/'old/reports/result.md'
    archive = tmp_path/'amendment/archived-reports/result.md'
    write_new_bytes(target, b'old scientific evidence'); write_new_bytes(archive, target.read_bytes())
    write_new_bytes(root/'reports/result.md', b'new complete coverage')
    write_new_json(root/'sealed.json', {'start': 0, 'end': 5, 'files': [
        {'path': 'reports/result.md', 'sha256': file_hash(root/'reports/result.md')}]})
    write_new_json(root.parent/'plan.json', {'legacy_reports': [
        {'path': str(target), 'archive': str(archive), 'sha256': file_hash(target)}],
        'jobs': [{'training_root': str(target.parent.parent)}]})
    return root, target, archive


def test_report_publication_needs_complete_analysis(tmp_path):
    from skillnet_cohort.first_calls_report import publish
    root, target, archive = publication(tmp_path)
    with pytest.raises(FileNotFoundError):
        publish(root)
    assert target.read_bytes() == archive.read_bytes()


def test_authorized_report_replacement_keeps_archive(tmp_path):
    from skillnet_cohort.first_calls_report import publish
    root, target, archive = publication(tmp_path)
    write_new_json(root/'analysis-complete.json', {'status': 'complete'})
    publish(root)
    assert target.read_bytes() == b'new complete coverage'
    assert archive.read_bytes() == b'old scientific evidence'
    assert read_json(root/'report-publication.json')['old_reports_recoverable']


@pytest.mark.parametrize('change', ['target', 'archive'])
def test_changed_report_or_backup_stops_before_replacement(tmp_path, change):
    from skillnet_cohort.first_calls_report import publish
    root, target, archive = publication(tmp_path)
    write_new_json(root/'analysis-complete.json', {'status': 'complete'})
    (target if change == 'target' else archive).write_bytes(b'independent edit')
    before = target.read_bytes()
    with pytest.raises(ValueError, match='no clobber'):
        publish(root)
    assert target.read_bytes() == before


def test_process_identity_and_pid_reuse_guard():
    from skillnet_cohort.first_calls_defer import process_identity, same_process
    binding = process_identity(os.getpid())
    assert same_process(binding)
    assert not same_process({**binding, 'start_ticks': binding['start_ticks']+1})
    assert not same_process({**binding, 'command_sha256': 'changed'})


def test_waiting_training_boundary_cannot_signal_any_process(monkeypatch):
    from skillnet_cohort import first_calls_defer as defer
    monkeypatch.setattr(defer, 'training_boundary', lambda p: None)
    monkeypatch.setattr(defer.os, 'kill', lambda *a: pytest.fail('signal during RL'))
    with pytest.raises(PermissionError, match='RL is still running'):
        defer.pause_coordinators({})


def test_pause_targets_only_two_coordinators_and_resumes_on_partial_error(monkeypatch):
    from skillnet_cohort import first_calls_defer as defer
    records = {'queue': {'pid': 123}, 'supervisor': {'pid': 456}, 'training': {'pid': 789}}
    monkeypatch.setattr(defer, 'training_boundary', lambda p: {'exit_code': 0})
    monkeypatch.setattr(defer, 'same_process', lambda b: b['pid'] != 456)
    monkeypatch.setattr(defer, 'process_identity', lambda p: {'state': 'S'})
    calls = []; monkeypatch.setattr(defer.os, 'kill', lambda *a: calls.append(a))
    with pytest.raises(ProcessLookupError):
        defer.pause_coordinators({'handoff_processes': records})
    assert calls == [(123, signal.SIGSTOP), (123, signal.SIGCONT)]


def test_resume_ignores_reused_pid(monkeypatch):
    from skillnet_cohort import first_calls_defer as defer
    monkeypatch.setattr(defer, 'same_process', lambda b: False)
    monkeypatch.setattr(defer.os, 'kill', lambda *a: pytest.fail('signal reused PID'))
    defer.resume_coordinators([{'pid': 123}])


@pytest.fixture(scope='module')
def actual_support(tmp_path_factory):
    """Read-only integration against retained seed404 data; temporary output only."""
    from skillnet_cohort.first_calls_support import build
    cohort = REPO/'artifacts/phase12/skillnet37-independent-s404-505-606-v4'
    source = cohort/'seed-404'; legacy = source/'windows/u0000-u0005-valid_unseen'
    if not (legacy/'sealed.json').is_file():
        pytest.skip('Local historical integration artifact is not installed')
    old = read_json(legacy/'protocol.json'); tmp = tmp_path_factory.mktemp('first-call-cpu-integration')
    root = tmp/'seed-404'
    support = build(old['runtime']['preparation'], source/'evaluations/u0000-valid_unseen-anchors', source, root/'support')
    config = copy.deepcopy(old); config['root'] = str(root)
    config['evaluation']['anchor_sets'] = support['anchor_sets']; config['evaluation']['anchor_count'] = support['anchor_count']
    config['coverage_amendment'] = {'legacy_window': str(legacy), 'prior_target_labels_available': True}
    (root/'models').symlink_to(source/'models', target_is_directory=True)
    write_new_json(root/'protocol.json', config)
    write_new_json(root.parent/'plan.json', {'legacy_seal': {'path': str(legacy/'sealed.json'), 'sha256': file_hash(legacy/'sealed.json')}})
    return root, config, old, legacy, support


def test_actual_404_has_all25_skills_404_first_anchors_and_descriptive_calls(actual_support):
    root, config, old, legacy, support = actual_support
    assert support['anchor_count'] == 404 and support['observed_skill_count'] == 25
    assert support['natural_call_count'] == 5558 and len(support['coverage']) == 37*5
    all_rows = pd.DataFrame(support['coverage']).query("phase == 'all'")
    assert all_rows.selected_anchors.sum() == 404
    assert all_rows.source_calls.sum() == 5558
    assert all_rows.later_calls_descriptive_only.sum() == 5154
    one = all_rows[all_rows.source_games == 1]
    assert not one.empty and one.utility_evaluable.all()
    assert not all_rows.quantity_filter_applied.any()


def test_actual_old_anchors_and_controls_are_a_semantic_subset(actual_support):
    from skillnet_cohort.first_calls_reuse import validate_compatibility
    root, config, old, legacy, support = actual_support
    validate_compatibility(config, old, root, legacy)


@pytest.mark.parametrize('field', ['temperature', 'gold_seeds', 'max_steps'])
def test_incompatible_episode_semantics_cannot_reuse(actual_support, field):
    from skillnet_cohort.first_calls_reuse import validate_compatibility
    root, config, old, legacy, _ = actual_support
    changed = copy.deepcopy(config); changed['evaluation'][field] = None
    with pytest.raises(ValueError, match='semantics'):
        validate_compatibility(changed, old, root, legacy)


def test_actual_episode_reuse_checks_identity_and_repartitions_without_rollouts(actual_support):
    from phase1.archive import stable_hash
    from phase2.protocol import evaluation_jobs, evaluation_identity
    from skillnet_cohort.first_calls_reuse import import_endpoint
    root, config, old, legacy, _ = actual_support
    before = file_hash(legacy/'sealed.json')
    for update in (0, 5):
        if update == 5:
            write_new_json(root/'window_signals/u0000-u0005/prediction.json', {'coverage_amendment': config['coverage_amendment']})
        assert import_endpoint(root, update) == 540
        audit = read_json(root/'reuse'/f'u{update:04d}.json')
        assert audit['expected_total'] == 3636 and audit['missing_to_evaluate'] == 3096
        jobs = evaluation_jobs(config, REPO)
        for rank in range(8):
            expected = {stable_hash(evaluation_identity(config, update, j))[:24] for j in jobs[rank::8]}
            path = root/'evaluations'/f'u{update:04d}'/f'shard-{rank}.jsonl'
            for line in path.read_text().splitlines() if path.exists() else []:
                row = json.loads(line)
                assert row['trajectory_id'] in expected and not row['reused_evidence']['episode_reexecuted']
        with pytest.raises(FileExistsError, match='already opened'):
            import_endpoint(root, update)
    assert file_hash(legacy/'sealed.json') == before


def test_coverage_sources_do_not_modify_legacy_source_freeze():
    from skillnet_cohort.seed_queue import verify_sources
    path = REPO/'artifacts/phase12/skillnet37-independent-s404-505-606-v4/recovery-v4/cohort-recovery.json'
    if not path.exists():
        pytest.skip('Local running queue is absent')
    plan = read_json(path)
    verify_sources(plan)
    assert not any('/first_calls_' in name for name in plan['source_sha256'])


def test_incremental_decision_scoring_matches_original_equations_on_cpu(tmp_path, monkeypatch):
    from skillnet_cohort import first_calls_measure as measure
    from phase2.direction import token_signals
    log = lambda values: torch.tensor(values, dtype=torch.float32).log_softmax(-1)
    live = log([[1., 2., 3.], [3., 1., 2.]])
    distributions = {('old', 0): live.clone(), ('new', 0): log([[2., 1., 3.], [1., 2., 3.]]),
        ('old', 1): log([[3., 2., 1.], [2., 1., 3.]]), ('new', 1): log([[2., 3., 1.], [1., 3., 2.]]),
        ('old', 2): log([[1., 3., 2.], [3., 2., 1.]]), ('new', 2): log([[3., 1., 2.], [1., 2., 3.]])}
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    monkeypatch.setattr('skillnet_cohort.lossless_tensor.load', lambda p: {
        'token_positions': torch.tensor([0, 1]), 'token_ids': torch.tensor([1, 2]), 'log_probs': live})
    def forward(model, ids, *args):
        result = distributions[model, int(ids[0])]
        return result, {layer: result.clone() for layer in (8, 16, 24, 32)}
    monkeypatch.setattr(measure, 'forward', forward)
    monkeypatch.setattr(measure, 'counter_input', lambda tok, meta, b, row, arm, text:
        (torch.tensor([1 if arm == 'placebo' else 2]), torch.ones(1), torch.zeros(1)))
    b = {'phase2_actual_loss_mask': torch.ones(1, 2), 'responses': torch.tensor([[1, 2]]),
         'advantages': torch.tensor([[2., -1.]]), 'input_ids': torch.zeros(1, 2, dtype=torch.long),
         'attention_mask': torch.ones(1, 2), 'position_ids': torch.zeros(1, 2)}
    meta = {'decision_id': 'd', 'trajectory_id': 't', 'group_id': 'grp', 'environment_step': 2,
            'info': {'selected_skill_id': 's', 'extra.gamefile': 'g'}}
    tokens, decisions, witness, noise = measure.score_decision((tmp_path, 'old', 'new'), None,
        b, 0, meta, {'text': 'neutral'}, 1., repeat=True)
    for arm, number in (('placebo', 1), ('null', 2)):
        expected = token_signals(live, distributions['new', 0], distributions['old', number],
            distributions['new', number], b['responses'][0], b['advantages'][0])
        actual = tokens[tokens.control == arm]
        for key, value in expected.items():
            np.testing.assert_array_equal(actual[key].to_numpy(), value.numpy())
    assert len(decisions) == 2 and noise == [0.]*4
    assert torch.equal(witness['end_original_replay'], distributions['new', 0])


def test_actual_old_scores_unchanged_when_count_eligibility_is_removed(actual_support):
    from phase2.aggregate import aggregate_features, MEAN_SIGNALS
    _, config, old, legacy, _ = actual_support
    directory = legacy/'window_signals/u0000-u0005'
    tokens = pd.concat([pd.read_parquet(directory/f'tokens-shard-{i}.parquet') for i in range(8)])
    decisions = pd.concat([pd.read_parquet(directory/f'decisions-shard-{i}.parquet') for i in range(8)])
    tau = read_json(legacy/'signals/calibration.json')['tau_delta']
    before, _ = aggregate_features(tokens, decisions, old, 5, tau)
    changed = copy.deepcopy(old); changed['signals'].update(minimum_nonzero_advantage_decisions=0,
        minimum_training_games=0, minimum_training_trajectories=0)
    after, _ = aggregate_features(tokens, decisions, changed, 5, tau)
    pd.testing.assert_frame_equal(before[list(MEAN_SIGNALS)], after[list(MEAN_SIGNALS)])
    assert before.supported.sum() < after.supported.sum()


def test_complete_assessment_order_never_invokes_RL_or_export(tmp_path, monkeypatch):
    from skillnet_cohort import first_calls_run as run
    root = tmp_path/'seed-404'; root.mkdir()
    plan = {'jobs': [{'run_root': str(root)}], 'model_inventory': {'u0000': {}, 'u0005': {}},
        'source_sha256': {}, 'projected_recording_upper_bytes': 1, 'wait_for_seed505_RL': True}
    write_new_json(tmp_path/'plan.json', plan)
    write_new_json(tmp_path/'handoff.json', {'status': 'ready', 'plan_sha256': file_hash(tmp_path/'plan.json'),
        'boundary': {'exit_code': 0}, 'paused_coordinators': [{'pid': 123}]})
    write_new_json(root/'protocol.json', {})
    obj = run.Assessment.__new__(run.Assessment); obj.path = tmp_path/'plan.json'
    obj.root = root; obj.plan = plan; obj.disk = lambda *a: None
    events = []; obj.commands = lambda jobs: events.append(('commands', [j[0][0] for j in jobs]))
    obj.evaluate = lambda update: events.append(('evaluate', update))
    monkeypatch.setattr('skillnet_cohort.first_calls_defer.gpu_users', lambda: [])
    monkeypatch.setattr('skillnet_cohort.first_calls_defer.same_process', lambda b: True)
    monkeypatch.setattr('skillnet_cohort.first_calls_defer.process_identity', lambda p: {'state': 'T'})
    monkeypatch.setattr('skillnet_cohort.assets.model_inventory', lambda p: {})
    monkeypatch.setattr('phase2.protocol.validate_extended', lambda *a: None)
    monkeypatch.setattr(run, 'verify_legacy_seal', lambda p: None)
    monkeypatch.setattr(run, 'lock_prediction', lambda r: events.append(('prediction',)))
    monkeypatch.setattr('skillnet_cohort.first_calls_reuse.import_endpoint', lambda r, u: events.append(('import', u)))
    monkeypatch.setattr('skillnet_cohort.first_calls_report.report', lambda r: events.append(('report',)))
    monkeypatch.setattr('skillnet_cohort.first_calls_report.publish', lambda r: events.append(('publish',)))
    monkeypatch.setattr('skillnet_cohort.window_storage.seal_window', lambda *a: events.append(('seal',)))
    obj.run()
    assert events == [('commands', ['skillnet_cohort.first_calls_measure']*8), ('commands', ['phase2.aggregate']),
        ('import', 0), ('evaluate', 0), ('prediction',), ('import', 5), ('evaluate', 5),
        ('report',), ('seal',), ('publish',)]
    assert not read_json(root/'complete.json')['training_reexecuted']


def test_protocol_preparation_with_real_source_never_changes_old_files(actual_support, tmp_path, monkeypatch):
    from skillnet_cohort import first_calls_protocol as protocol
    from skillnet_cohort.common import require_authorization
    from skillnet_cohort.first_calls_reuse import validate_compatibility
    _, _, old, legacy, support = actual_support
    original_cohort = legacy.parents[2]; source = original_cohort/'seed-404'
    # Use a disposable cohort that points at retained immutable source evidence.
    fake = tmp_path/'cohort'; fake.mkdir(); (fake/'seed-404').symlink_to(source, target_is_directory=True)
    previous = read_json(original_cohort/'recovery-v4/cohort-recovery.json'); previous['root'] = str(fake)
    queue_plan = tmp_path/'queue.json'; write_new_json(queue_plan, previous)
    evidence = tmp_path/'cpu-tests.xml'
    write_new_bytes(evidence, b'<testsuites><testsuite tests="27" failures="0" errors="0"/></testsuites>')
    monkeypatch.setattr('skillnet_cohort.assets.model_inventory', lambda p:
        support['source_checkpoint_identity'] if Path(p).name == 'u0000' else {'fake_unit_test_only': True})
    target = fake/'all-first-calls-v1'
    plan = protocol.prepare(queue_plan, target, handoff_processes={'fake_unit_test_only': True}, test_report=evidence)
    config = read_json(target/'seed-404/protocol.json')
    assert plan['continuations_per_endpoint'] == 3636 and plan['expected_skills'] == 25
    assert config['evaluation']['anchor_count'] == 404
    assert config['signals']['minimum_training_games'] == 0
    assert config['signals']['minimum_training_trajectories'] == 0
    assert config['signals']['minimum_nonzero_advantage_decisions'] == 0
    assert plan['followup_seeds'] == [505, 606]
    validate_compatibility(config, old, target/'seed-404', legacy)
    permit = require_authorization(target/'permit.json', old['runtime']['preparation'], 'evaluation')
    assert 'training' not in permit['operations'] and permit['router_max_api_calls'] == 0
    for row in plan['legacy_reports']:
        assert file_hash(row['path']) == file_hash(row['archive']) == row['sha256']


def test_full_report_join_and_missing_reason_output_use_only_complete_data(tmp_path, monkeypatch):
    from skillnet_cohort import first_calls_report as reporting
    from phase2.ranking import score_snapshot
    root = tmp_path/'amendment/seed-404'; source = tmp_path/'source'
    frame, anchors = outcome_frame()
    frame['trajectory_id'] = [f'synthetic-test-{i}' for i in range(len(frame))]
    settings = {'scores': {'P_int': -1, 'D_contribution': 1}, 'budgets_k': [1, 3],
        'budgets_fraction': [.25], 'event_thresholds': [0., .05]}
    write_new_json(root.parent/'plan.json', {'jobs': [{'seed': 404, 'training_root': str(source)}]})
    write_new_json(root/'protocol.json', {'evaluation': {'bootstrap_repetitions': 40, 'tau_M': .05},
        'ranking': settings, 'coverage_amendment': {'legacy_window': str(source/'old-window')}})
    coverage = pd.DataFrame([{'skill_id': skill, 'phase': 'all', 'context_id': 'all_alfworld',
        'utility_evaluable': skill == 's', 'source_calls': 10 if skill == 's' else 0,
        'source_games': 2 if skill == 's' else 0, 'train_decisions': 2, 'train_games': 2,
        'train_nonzero_games': 2, 'missing_utility_reason': None if skill == 's' else 'no_first_invocation'}
        for skill in ['s']+[f'unobserved-{i}' for i in range(36)]])
    write_new_bytes(root/'support/coverage.csv', coverage.to_csv(index=False).encode())
    features = pd.DataFrame([{'skill_id': 's', 'phase': 'all', 'context_id': 'all_alfworld',
        'control': 'placebo', 'supported': True, 'P_int': -.2, 'D_contribution': .2,
        'C_upd': 1., 'direction_coverage': 1., 'gate_coverage': 1.}])
    directory = root/'window_signals/u0000-u0005'
    write_new_bytes(directory/'skill_context_features.parquet', features.to_parquet(index=False))
    write_new_json(directory/'prediction.json', {'ranking_scores': score_snapshot(features, settings)})
    performance = b'seed,update,split,task,episodes,success_rate\n404,0,valid_unseen,all,134,0.33333333333333331\n'
    write_new_bytes(source/'reports/performance.csv', performance)
    monkeypatch.setattr(reporting, 'read_evaluations', lambda r: frame)
    monkeypatch.setattr('phase2.protocol.expected_trajectory_ids', lambda cfg, repo, u: frame[frame['update'] == u].trajectory_id.tolist())
    monkeypatch.setattr('phase2.protocol.anchor_sets', lambda cfg, repo: [{'anchors': anchors}])
    reporting.report(root)
    assert read_json(root/'analysis-complete.json')['skills_with_utility'] == 1
    assert (root/'reports/performance.csv').read_bytes() == performance
    complete = pd.read_csv(root/'window_metrics/coverage_and_effects.csv')
    assert len(complete) == 37 and complete.delta_utility.isna().sum() == 36
    assert len(list((root/'reports').iterdir())) == 7
    assert read_json(root/'window_metrics/summary.json')['status'] == 'complete'
