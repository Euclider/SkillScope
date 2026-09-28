"""CPU-only storage-race and explicit assessment-resume regression tests."""
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from skillnet_cohort.common import REPO, file_hash, read_json, write_new_json
from skillnet_cohort import first_calls_storage as storage
from skillnet_cohort import first_calls_recovery as recovery
# Import before fixture monkeypatches protocol.evaluation_jobs: this module
# intentionally binds the original function at import time.
from skillnet_cohort import first_calls_reuse


@pytest.mark.parametrize('suffix', ['journal', 'wal', 'shm'])
def test_disappearing_sqlite_sidecar_only_is_tolerated(tmp_path, monkeypatch, suffix):
    db = tmp_path/'router.sqlite3'; db.write_bytes(b'database')
    side = tmp_path/f'router.sqlite3-{suffix}'; side.write_bytes(b'transient')
    original = Path.lstat
    def raced(path):
        if path == side:
            raise FileNotFoundError(path)
        return original(path)
    monkeypatch.setattr(Path, 'lstat', raced)
    assert storage.storage_bytes(tmp_path) == 8


def test_actual_sqlite_commit_between_enumeration_and_stat(tmp_path, monkeypatch):
    db = tmp_path/'router.sqlite3'
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE cache (k TEXT)')
        connection.commit()
        connection.execute("INSERT INTO cache VALUES ('x')")
        assert (tmp_path/'router.sqlite3-journal').is_file()
        original = storage.os.walk
        def walked(root, **kwargs):
            rows = list(original(root, **kwargs))
            connection.commit()  # SQLite, not the test, deletes its own journal.
            yield from rows
        monkeypatch.setattr(storage.os, 'walk', walked)
        assert storage.storage_bytes(tmp_path) == db.stat().st_size


@pytest.mark.parametrize('name', ['model.safetensors', 'router.sqlite3', 'unknown.sqlite3-journal',
                                  'router.sqlite3-journal.backup', '.unregistered.json.tmp'])
def test_missing_permanent_or_unknown_files_still_fail(tmp_path, monkeypatch, name):
    target = tmp_path/name; target.write_bytes(b'evidence')
    original = Path.lstat
    def raced(path):
        if path == target:
            raise FileNotFoundError(path)
        return original(path)
    monkeypatch.setattr(Path, 'lstat', raced)
    with pytest.raises(FileNotFoundError):
        storage.storage_bytes(tmp_path)


def test_permission_error_is_not_treated_as_disappearance(tmp_path, monkeypatch):
    target = tmp_path/'router.sqlite3-journal'; target.write_bytes(b'evidence')
    original = Path.lstat
    def raced(path):
        if path == target:
            raise PermissionError(path)
        return original(path)
    monkeypatch.setattr(Path, 'lstat', raced)
    with pytest.raises(PermissionError):
        storage.storage_bytes(tmp_path)


def test_sidecar_exemption_requires_real_database(tmp_path):
    with pytest.raises(FileNotFoundError):
        storage.transient_file(tmp_path/'router.sqlite3-journal')
    target = tmp_path/'other'; target.write_bytes(b'db')
    (tmp_path/'router.sqlite3').symlink_to(target)
    assert not storage.transient_file(tmp_path/'router.sqlite3-journal')


def test_present_sidecars_count_and_symlinks_are_not_followed(tmp_path):
    owned = tmp_path/'owned'; owned.mkdir()
    (owned/'router.sqlite3').write_bytes(b'123')
    (owned/'router.sqlite3-journal').write_bytes(b'12345')
    outside = tmp_path/'outside'; outside.mkdir(); (outside/'big').write_bytes(b'x'*10000)
    (owned/'linked-directory').symlink_to(outside, target_is_directory=True)
    (owned/'linked-file').symlink_to(outside/'big')
    assert storage.storage_bytes(owned) == 8


@pytest.mark.parametrize('relative', ['forward_progress/.rank-1.json.abc', '.runtime-status.json.abc', '.publish-abc'])
def test_existing_progress_exemptions_are_preserved(relative):
    assert storage.transient_file(Path('/run')/relative)


@pytest.mark.parametrize('relative', ['evaluations/u0000/trajectories/s/'+('a'*24)+'.json',
                                     'evaluations/u0000/shard-1-complete.json'])
def test_disappeared_atomic_evaluation_temp_requires_published_result(tmp_path, relative):
    target = tmp_path/relative; target.parent.mkdir(parents=True)
    temporary = target.with_name('.'+target.name+'.abc')
    with pytest.raises(FileNotFoundError):
        storage.transient_file(temporary)
    target.write_bytes(b'{}')
    assert storage.transient_file(temporary)
    assert not storage.transient_file(tmp_path/temporary.name)


def test_disk_guards_still_enforce_free_and_total_capacity(tmp_path, monkeypatch):
    (tmp_path/'evidence').write_bytes(b'1234567890')
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100))
    assert storage.disk_gate(tmp_path, 20, minimum_free_bytes=80, maximum_run_bytes=30)['used_bytes'] == 10
    with pytest.raises(OSError):
        storage.disk_gate(tmp_path, 21, minimum_free_bytes=80, maximum_run_bytes=1000)
    with pytest.raises(OSError):
        storage.disk_gate(tmp_path, 20, minimum_free_bytes=0, maximum_run_bytes=29)


def test_cohort_counting_and_launched_parent_guard(tmp_path, monkeypatch):
    cohort = tmp_path/'cohort'; root = cohort/'seed-404'; root.mkdir(parents=True)
    write_new_json(root/'resource_limits.json', {'cohort_storage_root': str(cohort)})
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10000))
    with pytest.raises(ValueError, match='launched parent'):
        storage.disk_gate(root, minimum_free_bytes=0, maximum_run_bytes=10000)
    write_new_json(cohort/'queue_launch.json', {})
    (cohort/'other-seed').write_bytes(b'x'*100)
    result = storage.disk_gate(root, minimum_free_bytes=0, maximum_run_bytes=10000)
    assert result['used_bytes'] == sum(p.stat().st_size for p in cohort.rglob('*') if p.is_file())


@pytest.fixture
def episodes(tmp_path, monkeypatch):
    from phase1.archive import stable_hash
    from phase2 import protocol
    root = tmp_path/'run'; root.mkdir()
    config = {'run_id': 'run', 'evaluation': {'shards': 8}}
    write_new_json(root/'protocol.json', config)
    jobs = [('s', {'anchor_id': f'a{i}', 'state_id': f'state{i}'}, 'gold', i, 'original', {}) for i in range(16)]
    monkeypatch.setattr(protocol, 'evaluation_jobs', lambda c, repo: jobs)
    rows = []
    for i, job in enumerate(jobs[:8]):
        identity = protocol.evaluation_identity(config, 0, job); tid = stable_hash(identity)[:24]
        target = root/'evaluations/u0000/trajectories/s'/(tid+'.json')
        row = {**identity, 'trajectory_id': tid, 'trajectory_path': str(target),
               'prefix_replay_verified': True, 'success': False}
        write_new_json(target, {**row, 'original_anchor': job[1], 'steps': [{'a': 'look'}],
            'actual_continuation_seed': identity['continuation_seed']})
        index = root/'evaluations/u0000'/f'shard-{i}.jsonl'
        index.write_text(json.dumps(row)+'\n'); rows.append((index, target, row))
    return root, rows


def test_partial_endpoint_validates_and_reports_only_missing_work(episodes):
    root, rows = episodes; report = recovery.audit_endpoint(root, 0)
    assert (report['completed'], report['missing'], report['expected']) == (8, 8, 16)
    assert report['complete_shards'] == []
    recovery.verify_retained(report)
    index, _, _ = rows[0]
    with index.open('a') as stream:
        stream.write('{}\n')
    recovery.verify_retained(report)  # Original bytes preserved by append.


@pytest.mark.parametrize('damage', ['duplicate', 'wrong_shard', 'orphan', 'tail', 'seed', 'anchor', 'steps', 'replay', 'marker'])
def test_partial_endpoint_rejects_corruption_instead_of_rerunning(episodes, damage):
    root, rows = episodes; index, target, row = rows[0]
    if damage == 'duplicate':
        index.write_text(index.read_text()*2)
    elif damage == 'wrong_shard':
        rows[1][0].write_text(index.read_text())
    elif damage == 'orphan':
        index.write_text('')
    elif damage == 'tail':
        index.write_text(index.read_text()+'{"incomplete":')
    elif damage == 'marker':
        write_new_json(index.with_name('shard-0-complete.json'), {'jobs': 2})
    else:
        value = read_json(target)
        if damage == 'seed': value['actual_continuation_seed'] = 999
        if damage == 'anchor': value['original_anchor']['state_id'] = 'wrong'
        if damage == 'steps': value['steps'] = []
        if damage == 'replay': value['prefix_replay_verified'] = False
        target.write_text(json.dumps(value))
    with pytest.raises((ValueError, KeyError)):
        recovery.audit_endpoint(root, 0)


def test_retained_hashes_and_index_prefix_are_not_waived(episodes):
    root, rows = episodes; report = recovery.audit_endpoint(root, 0)
    rows[0][0].write_text('{}\n')
    with pytest.raises(ValueError, match='prefix changed'):
        recovery.verify_retained(report)
    rows[0][1].write_text('{}')
    with pytest.raises(ValueError, match='evidence changed'):
        recovery.verify_retained(report)


def test_completed_shards_never_start_a_policy(monkeypatch, tmp_path):
    obj = recovery.RecoveryAssessment.__new__(recovery.RecoveryAssessment)
    obj.root = tmp_path/'run'; obj.audit = tmp_path/'audit'
    report = {'files': [], 'shards': [], 'complete_shards': list(range(8)), 'missing': 0}
    monkeypatch.setattr(recovery, 'audit_endpoint', lambda *a: report)
    monkeypatch.setattr(obj, 'commands', lambda _: pytest.fail('Completed endpoint must not run a policy'))
    obj.evaluate(0)
    assert (obj.audit/'endpoint-u0000.json').is_file()


def test_missing_shards_use_resumable_evaluator_only(monkeypatch, tmp_path):
    obj = recovery.RecoveryAssessment.__new__(recovery.RecoveryAssessment)
    obj.root = tmp_path/'run'; obj.audit = tmp_path/'audit'
    states = iter([{'files': [], 'shards': [], 'complete_shards': [0, 2]},
                   {'files': [], 'shards': [], 'complete_shards': list(range(8)), 'missing': 0}])
    monkeypatch.setattr(recovery, 'audit_endpoint', lambda *a: next(states))
    calls = []; monkeypatch.setattr(obj, 'commands', calls.extend)
    obj.evaluate(0)
    assert [gpu for _, _, gpu in calls] == [1, 3, 4, 5, 6, 7]
    assert all(args[0] == 'phase2.evaluate' and '--max-jobs' not in args for args, _, _ in calls)


def queue_fixture(tmp_path, monkeypatch, alive=False):
    from skillnet_cohort import first_calls_defer as defer
    monkeypatch.setattr(defer, 'same_process', lambda _: alive)
    monkeypatch.setattr(defer, 'process_identity', lambda _: {'state': 'S'})
    audit = tmp_path/'legacy'; audit.mkdir()
    path = tmp_path/'legacy-plan.json'; write_new_json(path, {'recovery': {'audit_root': str(audit)}})
    plan = {'legacy_queue_plan': str(path), 'cohort_root': str(tmp_path/'cohort'),
            'handoff_processes': {'queue': {'pid': 123}}}
    return plan, audit


def test_running_legacy_queue_waits_without_signalling(tmp_path, monkeypatch):
    plan, _ = queue_fixture(tmp_path, monkeypatch, alive=True)
    monkeypatch.setattr(recovery.os, 'kill', lambda *a: pytest.fail('Never signal legacy queue'))
    assert recovery.queue_finished(plan) is None


def test_reviewed_storage_stop_can_reuse_completed_seeds_only(tmp_path, monkeypatch):
    plan, audit = queue_fixture(tmp_path, monkeypatch)
    write_new_json(audit/'queue_finished.json', {'status': 'budget_stop', 'completed_seeds': [404, 505], 'not_completed_seeds': [606]})
    write_new_json(audit/'not-started-606.json', {'reason': 'remaining_shared_disk_budget',
        'not_started_seeds': [606], 'no_files_deleted': True})
    assert recovery.queue_finished(plan) == audit/'queue_finished.json'


@pytest.mark.parametrize('reason', ['remaining_shared_time_budget', 'failed_training'])
def test_other_stop_reasons_never_silently_continue(tmp_path, monkeypatch, reason):
    plan, audit = queue_fixture(tmp_path, monkeypatch)
    write_new_json(audit/'queue_finished.json', {'status': 'budget_stop', 'completed_seeds': [404, 505], 'not_completed_seeds': [606]})
    write_new_json(audit/'not-started-606.json', {'reason': reason, 'not_started_seeds': [606], 'no_files_deleted': True})
    with pytest.raises(RuntimeError, match='unreviewed'):
        recovery.queue_finished(plan)


def test_failed_legacy_queue_is_not_restarted(tmp_path, monkeypatch):
    plan, _ = queue_fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match='without completion'):
        recovery.queue_finished(plan)


def test_dispatcher_finishes404505_but_cannot_train_or_publish606(tmp_path, monkeypatch):
    from skillnet_cohort import first_calls_defer, first_calls_protocol, first_calls_report, seed_queue
    original = tmp_path/'plan.json'; output = tmp_path/'recovery-v1'
    legacy_plan = tmp_path/'legacy-plan.json'; write_new_json(legacy_plan, {})
    write_new_json(original, {'root': str(tmp_path), 'legacy_queue_plan': str(legacy_plan),
        'legacy_queue_plan_sha256': file_hash(legacy_plan)})
    finished = tmp_path/'queue-finished.json'
    write_new_json(finished, {'status': 'budget_stop', 'completed_seeds': [404, 505], 'not_completed_seeds': [606]})
    path = output/'plan.json'
    write_new_json(path, {'root': str(output), 'original_plan': str(original),
        'original_plan_sha256': file_hash(original), 'approved': True, 'seeds': [404, 505, 606],
        'retained': {'files': [], 'shards': []}, 'engineering_tests': {'path': '/cpu-tests.xml'}})
    monkeypatch.setattr(recovery, 'queue_finished', lambda _: finished)
    monkeypatch.setattr(seed_queue, 'verify_sources', lambda _: None)
    monkeypatch.setattr(first_calls_defer, 'gpu_users', lambda: [])
    monkeypatch.setattr(first_calls_report, 'publish_cohort', lambda _: pytest.fail('Never publish an incomplete cohort'))
    monkeypatch.setattr(recovery.os, 'kill', lambda *a: pytest.fail('Never signal legacy processes'))
    events = []
    def prepare_followup(queue, destination, **kw):
        assert kw['seed'] == 505
        events.append('prepare505')
        write_new_json(destination/'plan.json', {'seed': 505})
    monkeypatch.setattr(first_calls_protocol, 'prepare', prepare_followup)
    class FakeAssessment:
        def __init__(self, plan, audit, record):
            self.root = Path(audit)/'artifacts'; self.active_stage = 'test'
        def resume404(self): events.append('resume404')
        def fresh_completed_seed(self, receipt):
            assert receipt == finished; events.append('assess505')
    monkeypatch.setattr(recovery, 'RecoveryAssessment', FakeAssessment)
    recovery.run(path)
    assert events == ['resume404', 'prepare505', 'assess505']
    assert read_json(output/'pending.json')['pending_seeds'] == [606]
    assert read_json(output/'deferred-status.json')['state'] == 'BLOCKED_LEGACY_STORAGE'
    assert not (output/'complete.json').exists()
    with pytest.raises(FileExistsError, match='already started'):
        recovery.run(path)


def test_resume404_order_cannot_repeat_readout_training_or_old_import(tmp_path, monkeypatch):
    from skillnet_cohort import assets, first_calls_run, first_calls_reuse, first_calls_report, window_storage, seed_queue
    obj = recovery.RecoveryAssessment.__new__(recovery.RecoveryAssessment)
    obj.root = tmp_path/'seed-404'; obj.audit = tmp_path/'audit'
    obj.plan = {'model_inventory': {'u0000': {}, 'u0005': {}}, 'projected_recording_upper_bytes': 123}
    obj.recovery = {'retained': {'files': [], 'shards': []}}
    events = []
    monkeypatch.setattr(assets, 'model_inventory', lambda _: {})
    monkeypatch.setattr(seed_queue, 'verify_sources', lambda _: None)
    monkeypatch.setattr(first_calls_run, 'verify_legacy_seal', lambda _: None)
    monkeypatch.setattr(recovery, 'validate_readout', lambda _: None)
    monkeypatch.setattr(obj, 'disk', lambda n: events.append(('disk', n)))
    monkeypatch.setattr(obj, 'commands', lambda _: pytest.fail('No readout, RL or export command during resume404'))
    monkeypatch.setattr(obj, 'evaluate', lambda u: events.append(('evaluate', u)))
    monkeypatch.setattr(first_calls_run, 'lock_prediction', lambda _: events.append(('lock', None)))
    monkeypatch.setattr(first_calls_reuse, 'import_endpoint', lambda _, u: events.append(('import', u)))
    monkeypatch.setattr(first_calls_report, 'report', lambda _: events.append(('report', None)))
    monkeypatch.setattr(window_storage, 'seal_window', lambda *a: events.append(('seal', None)))
    monkeypatch.setattr(first_calls_report, 'publish', lambda _: events.append(('publish', None)))
    obj.resume404()
    assert events == [('disk', 123), ('evaluate', 0), ('lock', None), ('import', 5),
                      ('evaluate', 5), ('report', None), ('seal', None), ('publish', None)]


@pytest.mark.parametrize('module', ['skillnet_cohort.segmented_training', 'skillnet_cohort.evaluate', 'phase2.export_model'])
def test_commands_cannot_train_export_or_resample_performance(module):
    obj = recovery.RecoveryAssessment.__new__(recovery.RecoveryAssessment)
    with pytest.raises(PermissionError, match='only permits'):
        obj.commands([([module, '--execute'], 'forbidden', 0)])


def test_fresh505_assessment_order_and_per_seed_handoff(tmp_path, monkeypatch):
    from skillnet_cohort import assets, first_calls_run, first_calls_report, window_storage, seed_queue, first_calls_defer
    from phase2 import protocol
    obj = recovery.RecoveryAssessment.__new__(recovery.RecoveryAssessment)
    obj.root = tmp_path/'seed-505'; obj.audit = tmp_path/'audit'; obj.path = tmp_path/'plan.json'
    write_new_json(obj.path, {}); write_new_json(obj.root/'protocol.json', {})
    obj.plan = {'jobs': [{'seed': 505}], 'model_inventory': {'u0000': {}, 'u0005': {}},
        'projected_recording_upper_bytes': 123, 'policy_registration': str(obj.path),
        'policy_registration_sha256': file_hash(obj.path)}
    obj.recovery = {}
    finished = tmp_path/'queue-finished.json'
    write_new_json(finished, {'status': 'budget_stop', 'completed_seeds': [404, 505]})
    events = []
    monkeypatch.setattr(assets, 'model_inventory', lambda _: {})
    monkeypatch.setattr(seed_queue, 'verify_sources', lambda _: None)
    monkeypatch.setattr(first_calls_defer, 'gpu_users', lambda: [])
    monkeypatch.setattr(first_calls_run, 'verify_legacy_seal', lambda _: None)
    monkeypatch.setattr(protocol, 'validate_extended', lambda *a: None)
    monkeypatch.setattr(obj, 'disk', lambda n: None)
    monkeypatch.setattr(obj, 'commands', lambda jobs: events.extend(args[0] for args, _, _ in jobs))
    monkeypatch.setattr(obj, 'evaluate', lambda u: events.append(f'evaluate{u}'))
    monkeypatch.setattr(first_calls_run, 'lock_prediction', lambda _: events.append('lock'))
    monkeypatch.setattr(first_calls_reuse, 'import_endpoint', lambda _, u: events.append(f'import{u}'))
    monkeypatch.setattr(first_calls_report, 'report', lambda _: events.append('report'))
    monkeypatch.setattr(window_storage, 'seal_window', lambda *a: events.append('seal'))
    monkeypatch.setattr(first_calls_report, 'publish', lambda _: events.append('publish'))
    obj.fresh_completed_seed(finished)
    assert events == ['skillnet_cohort.first_calls_measure']*8 + ['phase2.aggregate', 'import0',
        'evaluate0', 'lock', 'import5', 'evaluate5', 'report', 'seal', 'publish']


def test_legacy105_and_amendment112_source_hashes_unchanged():
    from skillnet_cohort.seed_queue import verify_sources
    q = REPO/'artifacts/phase12/skillnet37-independent-s404-505-606-v4'
    for relative, count in [('recovery-v4/cohort-recovery.json', 105), ('all-first-calls-v1/plan.json', 112)]:
        plan = read_json(q/relative)
        assert len(plan['source_sha256']) == count
        verify_sources(plan)


def test_actual_1871_partial_episodes_validate_without_reexecution():
    root = REPO/'artifacts/phase12/skillnet37-independent-s404-505-606-v4/all-first-calls-v1/seed-404'
    report = recovery.audit_endpoint(root, 0)
    # Counts may grow after explicit recovery starts; the registered prefix must exist.
    assert report['expected'] == 3636 and report['completed'] >= 1871
    recovery.verify_retained(report)
