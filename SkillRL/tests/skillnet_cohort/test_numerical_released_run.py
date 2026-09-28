from pathlib import Path

import pytest

from skillnet_cohort import numerical_released_run as runner
from skillnet_cohort import numerical_readout as nr
from skillnet_cohort import numerical_readout_report as reporting
from skillnet_cohort import released_505_evaluation as utility
from skillnet_cohort.common import file_hash, write_new_json


def minimal_plan(root):
    return {'schema_version': runner.SCHEMA, 'root': str(root), 'approved': True,
        'seeds': [404, 505], 'automatic_retry': False,
        'seed505_restart_mode': 'fresh_processes_reuse_disk_results',
        'minimum_free_gpu_mib': 28000, 'rewrite_original_reports': False,
        'external_api_calls': 0, 'training_reexecuted': False, 'seed606_started': False,
        'source_sha256': {}, 'bindings': [], 'dependency_sources': [],
        'source_roots': {str(k): str(v) for k, v in nr.SOURCES.items()}}


@pytest.mark.parametrize('field,value', [('approved', False), ('seeds', [404, 606]),
    ('automatic_retry', True), ('minimum_free_gpu_mib', 15000), ('rewrite_original_reports', True),
    ('external_api_calls', 1), ('training_reexecuted', True), ('seed606_started', True),
    ('seed505_restart_mode', 'SIGCONT')])
def test_bindings_reject_scope_expansion(monkeypatch, tmp_path, field, value):
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    plan = minimal_plan(tmp_path); plan[field] = value
    write_new_json(tmp_path/'plan.json', plan)
    with pytest.raises(PermissionError):
        runner.binding(tmp_path/'plan.json')


def test_correct_binding_and_dependency_hash_required(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    dependency = tmp_path/'executor.py'; dependency.write_text('original')
    plan = minimal_plan(tmp_path)
    plan['dependency_sources'] = [{'path': str(dependency), 'sha256': file_hash(dependency)}]
    write_new_json(tmp_path/'plan.json', plan)
    assert runner.binding(tmp_path/'plan.json', 404) == plan
    with pytest.raises(PermissionError):
        runner.binding(tmp_path/'plan.json', 606)
    dependency.write_text('changed')
    with pytest.raises(ValueError):
        runner.binding(tmp_path/'plan.json')


def test_numerical_adapter_changes_only_binding_and_restores_on_error():
    before = nr.binding, nr.paused_tree, reporting.binding, nr.scoring_adapter
    with pytest.raises(RuntimeError):
        with runner.numerical_adapter():
            assert nr.binding is runner.binding and nr.paused_tree is runner.released_guard
            assert nr.scoring_adapter is before[3]
            raise RuntimeError('synthetic')
    assert (nr.binding, nr.paused_tree, reporting.binding, nr.scoring_adapter) == before


def test_numerical_jobs_preserve_gpu_pairs_and_logical_partition():
    path = Path('/new/plan.json')
    jobs = runner.command_jobs(path, 404, 'measure', list(range(4)))
    assert [j[2] for j in jobs] == ['0,1', '2,3', '4,5', '6,7']
    assert all(j[0][0] == runner.MODULE for j in jobs)
    assert [j[0][-1] for j in jobs] == ['0', '1', '2', '3']
    assert runner.command_jobs(path, 505, 'report')[0][0][-1] == '--report'
    with pytest.raises(ValueError):
        runner.command_jobs(path, 404, 'measure', [0, 4])


@pytest.mark.parametrize('seed,mode', [(606, 'measure'), (404, 'training'), (505, 'export'), (404, 'evaluate')])
def test_no_RL_or_performance_jobs(seed, mode):
    with pytest.raises(PermissionError):
        runner.command_jobs(Path('/plan.json'), seed, mode)


@pytest.mark.parametrize('update,shards', [(1, [0]), (0, []), (5, [8]), (0, [0, 0])])
def test_only_registered505_utility_endpoints(update, shards):
    with pytest.raises(PermissionError):
        runner.utility_jobs(Path('/plan.json'), update, shards)


def test_utility_jobs_have_single_distinct_gpus():
    jobs = runner.utility_jobs(Path('/plan.json'), 5, list(range(8)))
    assert [j[2] for j in jobs] == list(map(str, range(8)))
    assert all(j[0][0] == 'skillnet_cohort.released_505_evaluation' for j in jobs)


def test_pipeline_404_complete_before505_readout_before_fresh_restart(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    monkeypatch.setattr(runner, 'binding', lambda p: {'root': str(tmp_path)})
    monkeypatch.setattr(runner, 'released_guard', lambda p: None)
    monkeypatch.setattr(runner, 'commands', lambda path, jobs: events.extend(j[1] for j in jobs))
    monkeypatch.setattr(runner.previous_run, 'verify_report_completion', lambda p: events.append('verify-'+p.name))
    monkeypatch.setattr(runner, 'restart505', lambda p: events.append('fresh505-utility'))
    runner.pipeline(tmp_path/'plan.json')
    assert events.index('seed404-report') < events.index('seed505-measure-shard0')
    assert events.index('seed505-aggregate') < events.index('fresh505-utility') < events.index('seed505-report')
    assert len([x for x in events if '-measure-shard' in x]) == 16


def test_404_failure_never_starts505(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'binding', lambda p: {'root': str(tmp_path)})
    monkeypatch.setattr(runner, 'released_guard', lambda p: None)
    def fail(*args):
        raise RuntimeError('synthetic readout failure')
    monkeypatch.setattr(runner, 'commands', fail)
    monkeypatch.setattr(runner, 'restart505', lambda p: pytest.fail('505 started early'))
    with pytest.raises(RuntimeError):
        runner.pipeline(tmp_path/'plan.json')


def test_505_restart_needs_complete404_report(tmp_path):
    with pytest.raises(FileNotFoundError):
        runner.verify_restart_requirements({'root': str(tmp_path)})


def test_505_restart_needs_committed_exact_scores(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.previous_run, 'verify_report_completion', lambda p: None)
    with pytest.raises(FileNotFoundError):
        runner.verify_restart_requirements({'root': str(tmp_path)})
    write_new_json(tmp_path/'seed-505/committed.json', {'seed': 505, 'all_legacy_scalar_signals_exact': False})
    with pytest.raises(PermissionError):
        runner.verify_restart_requirements({'root': str(tmp_path)})


def test_released505_indices_cannot_advance_early(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'released', lambda r: None)
    receipt = tmp_path/'release.json'
    write_new_json(receipt, {'status': 'RELEASED_BY_USER', 'disk_results_preserved': True})
    index = tmp_path/'shard0.jsonl'; index.write_text('retained\n')
    progress = tmp_path/'pause.json'
    write_new_json(progress, {'endpoints': {'u0000': {'shards': [{'path': str(index), 'sha256': file_hash(index)}]}}})
    plan = {'root': str(tmp_path), 'release_505': {'path': str(receipt), 'sha256': file_hash(receipt)},
            'pause_verification': {'path': str(progress)}}
    runner.released_guard(plan)
    index.write_text('retained\nnew\n')
    with pytest.raises(ValueError):
        runner.released_guard(plan)


def test_command_validation_rejects_overlapping_gpus_before_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'binding', lambda p: {'root': str(tmp_path)})
    monkeypatch.setattr(runner.nr, 'disk', lambda p: None)
    monkeypatch.setattr(runner, 'released_guard', lambda p: None)
    path = tmp_path/'plan.json'
    jobs = runner.command_jobs(path, 404, 'measure', [0])+runner.command_jobs(path, 404, 'measure', [4])
    with pytest.raises(PermissionError, match='overlap'):
        runner.commands(path, jobs)


def test_utility_worker_cannot_bypass_restart_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'binding', lambda p: {'root': str(tmp_path)})
    monkeypatch.setattr(runner, 'released_guard', lambda p: None)
    with pytest.raises(FileNotFoundError):
        utility.evaluate(tmp_path/'plan.json', 0, 0)


def test_fresh_utility_wrapper_does_not_allow_other_commands(monkeypatch, tmp_path):
    monkeypatch.setattr(utility, 'admitted', lambda p: {})
    instance = object.__new__(utility.ReleasedAssessment)
    instance.root = tmp_path/'seed-505'; instance.recovery = {'root': str(tmp_path)}
    with pytest.raises(PermissionError):
        instance.commands([(['phase2.export_model'], 'export', 0)])
