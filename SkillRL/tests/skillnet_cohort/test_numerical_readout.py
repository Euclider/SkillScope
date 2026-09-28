import copy
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from skillnet_cohort import numerical_readout as nr
from skillnet_cohort import numerical_readout_run as runner
from skillnet_cohort import numerical_readout_report as reporting
from skillnet_cohort.common import write_new_json, file_hash


def token_frame():
    return pd.DataFrame({'decision_id': ['d0', 'd0'], 'control': ['placebo', 'null'],
        'response_token_offset': [0, 0], 'skill_id': ['s', 's'], 'trajectory_id': ['t', 't'],
        'game_id': ['g', 'g'], 'action_token_id': [2, 2], 'advantage': [1., 1.],
        'P_int': [.2, .3], 'C_upd': [.4, .5], 'direction_valid': [True, True]})


def revised_tokens(old):
    new = old.copy()
    for name in ('advantage', 'P_int', 'C_upd', 'direction_valid'):
        new['legacy_'+name] = old[name]
    new['P_int'] = [.21, .31]
    return new


def test_legacy_comparator_handles_string_identity_and_bool_without_isnan_type_error():
    old = token_frame(); new = revised_tokens(old)
    nr.assert_legacy_equal(new.iloc[::-1], old)


@pytest.mark.parametrize('name,value', [('legacy_P_int', 99.), ('legacy_C_upd', float('nan')),
    ('legacy_direction_valid', False), ('game_id', 'different'), ('skill_id', 'different'),
    ('action_token_id', 3), ('legacy_advantage', 2.)])
def test_comparator_rejects_numerical_or_input_changes(name, value):
    old = token_frame(); new = revised_tokens(old); new.loc[0, name] = value
    with pytest.raises(ValueError):
        nr.assert_legacy_equal(new, old)


def test_comparator_accepts_equal_nan_only_not_missing_identity():
    old = token_frame(); old['P_int'] = np.nan
    new = revised_tokens(old); new['legacy_P_int'] = np.nan
    nr.assert_legacy_equal(new, old)
    with pytest.raises(ValueError):
        nr.assert_legacy_equal(new.iloc[:1], old)


def test_adapter_restores_original_functions_even_on_error():
    from skillnet_cohort import first_calls_measure as original
    before = original.forward, original.token_signals
    config = {'signals': {'tau_C': 0., 'epsilon': 1e-12}, '_frozen_tau_delta': 1e-8}
    with pytest.raises(RuntimeError):
        with nr.scoring_adapter(config):
            assert original.forward is not before[0]
            raise RuntimeError('synthetic')
    assert (original.forward, original.token_signals) == before


def test_adapter_keeps_forward_on_model_device_and_returns_to_primary(monkeypatch):
    from phase2 import measure
    from skillnet_cohort import first_calls_measure as original
    observed = []
    @contextmanager
    def cuda_device(device):
        observed.append(device); yield
    class Probability:
        def to(self, device):
            observed.append(('return', device)); return self
    class Model:
        def parameters(self):
            return iter([type('Parameter', (), {'device': 'cuda:1'})()])
    monkeypatch.setattr(torch.cuda, 'device', cuda_device)
    monkeypatch.setattr(measure, 'forward', lambda *a: (Probability(), {'hidden': 'unchanged'}))
    cfg = {'signals': {'tau_C': 0., 'epsilon': 1e-12}, '_frozen_tau_delta': 1e-8}
    with nr.scoring_adapter(cfg):
        _, hidden = original.forward(Model())
    assert observed == ['cuda:1', ('return', 'cuda:0')]
    assert hidden == {'hidden': 'unchanged'}


def test_centered_gate_variant_does_not_replace_P_with_separate_centered_score():
    legacy = pd.DataFrame({'C_upd': [.1], 'P_int': [.2]})
    stable = pd.DataFrame({'C_upd': [.11], 'P_int': [.21], 'P_int_centered': [99.],
        'C_upd_centered': [.4], 'D_contribution': [.3], 'D_centered_contribution': [.5],
        'gate_coverage': [.6], 'gate_centered_coverage': [.7]})
    copied = stable.copy()
    got = reporting.variant_features(legacy, stable, 'stable_centered_gate')
    assert got.P_int.iloc[0] == .21 and got.C_upd.iloc[0] == .4 and got.D_contribution.iloc[0] == .5
    pd.testing.assert_frame_equal(stable, copied)
    pd.testing.assert_frame_equal(reporting.variant_features(legacy, stable, 'legacy_recorded'), legacy)
    with pytest.raises(ValueError):
        reporting.variant_features(legacy, stable, 'best_after_gold')


def test_old_margin_is_U0_evidence_and_game_weighted():
    features = pd.DataFrame([{'skill_id': 's', 'context_id': 'c', 'phase': 'all'}])
    margins = pd.DataFrame([{'skill_id': 's', 'context_id': 'c', 'phase': 'early',
        'update': u, 'purpose': purpose, 'game_id': game, 'M_placebo': value}
        for u, purpose, game, value in [(0, 'evidence', 'g1', .2), (0, 'evidence', 'g1', .4),
            (0, 'evidence', 'g2', .9), (5, 'evidence', 'g3', -50.), (0, 'gold', 'g4', -50.)]])
    got = reporting.attach_old_margin(features, margins, 'placebo', ['all'])
    assert got.old_margin.iloc[0] == pytest.approx(.6)


@pytest.mark.parametrize('seed,mode', [(606, 'measure'), (404, 'training'), (505, 'evaluate'), (404, 'export')])
def test_commands_never_expand_to_RL_or_environment(seed, mode):
    with pytest.raises(PermissionError):
        runner.command_jobs(Path('/plan.json'), seed, mode)


def test_gpu_pairs_are_disjoint_and_logical_shards_preserved():
    jobs = runner.command_jobs(Path('/plan.json'), 404, 'measure', range(4))
    assert [j[2] for j in jobs] == ['0,1', '2,3', '4,5', '6,7']
    assert [j[0][-1] for j in jobs] == ['0', '1', '2', '3']
    assert runner.command_jobs(Path('/plan.json'), 404, 'measure', [4])[0][2] == '0,1'
    with pytest.raises(ValueError):
        runner.command_jobs(Path('/plan.json'), 404, 'measure', [0, 4])
    with pytest.raises(ValueError):
        runner.command_jobs(Path('/plan.json'), 404, 'measure', [0, 0])


def test_pipeline_enforces_404_report_before_505_and_commit_before_resume(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(runner, 'binding', lambda path: {'root': str(tmp_path)})
    monkeypatch.setattr(runner, 'paused_tree', lambda p: [])
    monkeypatch.setattr(runner, 'commands', lambda path, jobs: events.extend(j[1] for j in jobs))
    monkeypatch.setattr(runner, 'verify_report_completion', lambda p: events.append('verified-'+p.name))
    monkeypatch.setattr(runner, 'resume_505', lambda p: events.append('resume-505'))
    monkeypatch.setattr(runner, 'wait_505', lambda p: events.append('wait-505'))
    runner.pipeline(tmp_path/'plan.json')
    assert events.index('seed404-report') < events.index('seed505-measure-shard0')
    assert events.index('seed505-aggregate') < events.index('resume-505') < events.index('wait-505') < events.index('seed505-report')
    assert len([x for x in events if '-measure-shard' in x]) == 16


def test_numerical_failure_does_not_auto_resume_505(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'binding', lambda path: {'root': str(tmp_path)})
    monkeypatch.setattr(runner, 'paused_tree', lambda p: [])
    def fail(*args):
        raise ValueError('synthetic numerical failure')
    monkeypatch.setattr(runner, 'commands', fail)
    monkeypatch.setattr(runner, 'resume_505', lambda p: pytest.fail('Premature resume'))
    with pytest.raises(ValueError):
        runner.pipeline(tmp_path/'plan.json')


def test_resume_cannot_bypass_completed_404_report(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'binding', lambda path: {'root': str(tmp_path)})
    with pytest.raises(FileNotFoundError):
        runner.resume_505(tmp_path/'plan.json')


def test_report_receipt_and_every_output_are_verified(tmp_path):
    output = tmp_path/'report.md'; output.write_text('fixed numerical report')
    provenance = tmp_path/'report-provenance.json'
    write_new_json(provenance, {'files': [{'path': 'report.md', 'sha256': file_hash(output)}]})
    write_new_json(tmp_path/'complete.json', {'status': 'complete', 'all_comparators_present': True,
        'provenance_sha256': file_hash(provenance)})
    runner.verify_report_completion(tmp_path)
    output.write_text('changed')
    with pytest.raises(ValueError):
        runner.verify_report_completion(tmp_path)


def test_bindings_reject_wrong_scope_and_mutated_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(nr, 'COHORT', tmp_path)
    root = tmp_path/'numerical-readout-v1'
    source = tmp_path/'input.json'; source.write_text('{}')
    plan = {'schema_version': nr.SCHEMA, 'approved': True, 'seeds': [404, 505],
        'automatic_retry': False, 'root': str(root), 'resume_505_only_after_404_report': True,
        'rewrite_original_reports': False, 'source_sha256': {},
        'bindings': [{'path': str(source), 'sha256': file_hash(source)}]}
    write_new_json(root/'plan.json', plan)
    assert nr.binding(root/'plan.json')['seeds'] == [404, 505]
    source.write_text('{"changed": true}')
    with pytest.raises(ValueError):
        nr.binding(root/'plan.json')
    plan['approved'] = False
    (root/'plan.json').write_text(json.dumps(plan))
    with pytest.raises(PermissionError):
        nr.binding(root/'plan.json')


def test_stable_chunking_preserves_full_legacy_shape_and_output(monkeypatch):
    import phase2.stable_direction as stable
    actual = stable.legacy_token_signals; seen = []
    def observed(*args, **kwargs):
        seen.append(args[0].shape); return actual(*args, **kwargs)
    monkeypatch.setattr(stable, 'legacy_token_signals', observed)
    p = torch.full((40, 7), 1/7., dtype=torch.float32).log()
    args = (p, p+.01, p, p, torch.zeros(40, dtype=torch.long), torch.ones(40))
    got = stable.token_signals(*args, include_legacy=True)
    assert seen == [torch.Size([40, 7])]
    expected = actual(*args)
    for k, v in expected.items():
        assert torch.equal(got['legacy_'+k], v)
    assert got['P_int'].shape == (40,)
    assert got['P_centering_abs_error'].max() < 1e-12


def test_largest_action_is_scored_first_but_original_order_is_recoverable():
    selected = [(i, {'decision_id': f'd{i}'}) for i in range(3)]
    recorded = {'d0': range(12), 'd1': range(1024), 'd2': range(12)}
    ordered = nr.ordered_decisions(selected, recorded)
    assert [i for i, _ in ordered] == [1, 0, 2]
    assert [item for _, item in sorted(ordered)] == selected


def test_paused_tree_rejects_progress_or_ownership_change(monkeypatch, tmp_path):
    from skillnet_cohort import first_calls_defer
    parent = {'pid': 11, 'parent': 1, 'state': 'T', 'start_ticks': 123, 'command_sha256': 'parent'}
    child = {'pid': 12, 'parent': 11, 'state': 'T', 'start_ticks': 124, 'command_sha256': 'child'}
    pause = tmp_path/'pause.json'; index = tmp_path/'shard-0.jsonl'; progress = tmp_path/'progress.json'
    index.write_text('{}\n')
    write_new_json(pause, {'supervisor_pid': 11, 'after': [parent, child]})
    write_new_json(progress, {'endpoints': {'u0000': {'shards': [{'path': str(index), 'sha256': file_hash(index)}]}}})
    plan = {'pause': {'path': str(pause), 'sha256': file_hash(pause)},
        'pause_verification': {'path': str(progress), 'sha256': file_hash(progress)}}
    monkeypatch.setattr(first_calls_defer, 'process_identity', lambda p: {11: parent, 12: child}.get(p))
    monkeypatch.setattr(first_calls_defer, 'descendants', lambda p: [child])
    assert len(nr.paused_tree(plan)) == 2
    index.write_text('{}\n{}\n')
    with pytest.raises(ValueError, match='utility advanced'):
        nr.paused_tree(plan)
    child['state'] = 'S'
    with pytest.raises(ProcessLookupError):
        nr.paused_tree(plan)


def test_witness_publication_is_lossless_no_clobber_and_no_partial_file(tmp_path):
    value = {'a': torch.tensor([[1., 2.]]), 'metadata': {'decision_id': 'd'}}
    target = tmp_path/'witness.pt'
    nr.write_new_tensor(target, value)
    got = torch.load(target, weights_only=False)
    assert torch.equal(got['a'], value['a']) and got['metadata'] == value['metadata']
    assert sorted(p.name for p in tmp_path.iterdir()) == ['witness.pt']
    with pytest.raises(FileExistsError):
        nr.write_new_tensor(target, {'different': True})
