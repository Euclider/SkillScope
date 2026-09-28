"""CPU-only explicit local-cache resumption, using the real frozen router API."""
import copy
import json
from pathlib import Path
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_system.memory.router_cache import RouterCache, RouterCacheError, RouterBudgetExceeded, digest
from skillnet_cohort.common import REPO, read_json, file_hash, write_new_json
from skillnet_cohort.explicit_router_resume import (
    ExplicitLocalResumeCache, audit_cache, install, load_authorization, read_only, row_signature,
)
from tests.skill_router.test_embedding_batch_router import runtime, request


@pytest.fixture
def bundle(tmp_path):
    run = tmp_path/'seed-404'; run.mkdir()
    memory, router, encoder = runtime(run, budget=100)
    items = []
    for step in range(5):
        state = request(memory, step); state.pop('candidate_bundle')
        visible = router._visible_input(**state)
        key = digest({'protocol_hash': router.protocol_hash, 'input_hash': digest(visible)})
        items.append((key, visible))
    reserved = router.cache.reserve_many_local(items)
    router.route_many([request(memory, 10)])  # One successful decision must remain byte-identical.
    router.route_many([request(memory, 10)])  # Preserve original cache-hit history, too.
    output = tmp_path/'recovery-v2'; output.mkdir()
    snapshot = output/'router-before-resume.sqlite3'; shutil.copyfile(router.cache.path, snapshot)
    allowed = []
    with read_only(router.cache.path) as c:
        for key, _ in items:
            row = c.execute('SELECT id,key,started,status,data FROM attempts WHERE id=?', (reserved[key],)).fetchone()
            allowed.append({'id': row[0], 'key': key, 'started': row[2], 'status': row[3], 'row_sha256': row_signature(row)})
    protocol = {'runtime': {'router_backend': 'skillrl_embedding_state_batch', 'cache_path': str(router.cache.path),
        'max_api_calls': 0, 'max_local_calls': 100}}
    write_new_json(run/'protocol.json', protocol)
    auth = {'schema_version': 'skillnet.explicit_local_query_resume.v1', 'approved': True,
        'recovery_root': str(output), 'run_root': str(run), 'cache_path': str(router.cache.path),
        'protocol_hash': router.protocol_hash, 'evaluation_protocol_sha256': file_hash(run/'protocol.json'),
        'max_local_calls': 100, 'snapshot': {'path': str(snapshot), 'sha256': file_hash(snapshot)},
        'allowed': allowed, 'external_api_calls': 0, 'max_reissues_per_key': 1}
    path = output/'router-resume-authorization.json'; write_new_json(path, auth)
    write_new_json(output/'plan.json', {'approved': True, 'source_sha256': {},
        'router_resume': {'path': str(path), 'sha256': file_hash(path)}})
    return memory, router, encoder, items, path, load_authorization(path)


def adapted(bundle):
    _, router, _, _, _, auth = bundle
    return ExplicitLocalResumeCache(router.cache.path, router.protocol, 100, authorization=auth)


def test_five_frozen_interruptions_pass_readonly_preflight(bundle):
    _, router, _, items, _, auth = bundle
    before = file_hash(router.cache.path)
    value = audit_cache(router.cache.path, auth)
    assert value['status'] == 'PASS' and value['successful_decisions'] == 1
    assert {r['key'] for r in value['unresolved']} == {key for key, _ in items}
    assert file_hash(router.cache.path) == before
    with pytest.raises(RouterCacheError, match='before GPU'):
        audit_cache(router.cache.path)


def test_normal_cache_still_refuses_same_keys(bundle):
    _, router, _, items, _, _ = bundle
    with pytest.raises(RouterCacheError, match='no automatic retry'):
        router.cache.reserve_many_local(items[:1])


def test_all_five_reissued_once_real_router_preserves_successes_and_old_ledger(bundle):
    memory, router, encoder, _, _, auth = bundle
    router.cache = adapted(bundle)
    rows = router.route_many([request(memory, i) for i in range(5)])
    assert [r['selected_skill_id'] for r in rows] == list(memory.bank.skill_ids[:5])
    assert all(r['skill_router_api']['local_calls_this_step'] == 1 for r in rows)
    before = len(encoder.calls)
    again = router.route_many([request(memory, i) for i in range(5)])
    assert len(encoder.calls) == before and all(r['skill_router_api']['cache_hit'] for r in again)
    after = audit_cache(router.cache.path, auth)
    assert after['unresolved'] == [] and after['successful_decisions'] == 6
    assert after['attempts'] == 11 and after['original_rows_unchanged']
    with read_only(router.cache.path) as c:
        assert c.execute("SELECT COUNT(*) FROM attempts WHERE status='started'").fetchone()[0] == 5
        recs = [json.loads(r[0]) for r in c.execute('SELECT record FROM decisions')]
        resumed = [r for r in recs if 'explicit_local_resume' in r]
        assert len(resumed) == 5
        assert all(r['explicit_local_resume']['authorization_sha256'] == auth['_authorization_sha256'] for r in resumed)


def test_successful_query_is_never_reserved_again(bundle):
    memory, router, _, _, _, _ = bundle
    cache = adapted(bundle)
    state = request(memory, 10); state.pop('candidate_bundle'); visible = router._visible_input(**state)
    key = digest({'protocol_hash': router.protocol_hash, 'input_hash': digest(visible)})
    with pytest.raises(RouterCacheError, match='reused'):
        cache.reserve_many_local([(key, visible)])


def test_consumed_reissue_cannot_be_reissued_even_after_process_restart(bundle):
    _, _, _, items, _, _ = bundle
    first = adapted(bundle).reserve_many_local(items[:1])
    assert first
    with pytest.raises(RouterCacheError, match='consumed'):
        adapted(bundle).reserve_many_local(items[:1])


def test_failed_authorized_reissue_cannot_retry(bundle):
    _, router, _, items, _, auth = bundle
    cache = adapted(bundle); attempt = cache.reserve_many_local(items[:1])[items[0][0]]
    cache.fail(attempt, {'error_type': 'test', 'external_api_calls': 0})
    with pytest.raises(RouterCacheError, match='consumed'):
        adapted(bundle).reserve_many_local(items[:1])
    with pytest.raises(RouterCacheError, match='before GPU'):
        audit_cache(router.cache.path, auth)


def test_concurrent_reservation_consumes_allowance_exactly_once(bundle):
    _, router, _, items, _, _ = bundle
    def reserve():
        try:
            return bool(adapted(bundle).reserve_many_local(items[:1]))
        except RouterCacheError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(lambda _: reserve(), range(8))) == 1
    with read_only(router.cache.path) as c:
        assert c.execute('SELECT COUNT(*) FROM attempts WHERE key=?', (items[0][0],)).fetchone()[0] == 2


def test_unregistered_interruptions_fail_before_any_gpu_work(bundle):
    memory, router, _, _, _, auth = bundle
    state = request(memory, 30); state.pop('candidate_bundle'); visible = router._visible_input(**state)
    key = digest({'protocol_hash': router.protocol_hash, 'input_hash': digest(visible)})
    router.cache.reserve_many_local([(key, visible)])
    with pytest.raises(RouterCacheError, match='before GPU'):
        audit_cache(router.cache.path, auth)
    with pytest.raises(RouterCacheError, match='not authorized'):
        adapted(bundle).reserve_many_local([(key, visible)])


def test_mixed_batch_rejects_atomically_without_consuming_good_allowance(bundle):
    _, router, _, items, _, _ = bundle
    cache = adapted(bundle); changed = copy.deepcopy(items[1][1]); changed['step_index'] = 999
    with pytest.raises(RouterCacheError, match='identity changed'):
        cache.reserve_many_local([items[0], (items[1][0], changed)])
    with read_only(router.cache.path) as c:
        assert c.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 6
    assert cache.reserve_many_local(items[:1])


def test_original_status_is_not_changed_to_hide_interruption(bundle):
    _, router, _, items, _, auth = bundle
    with sqlite3.connect(router.cache.path) as c:
        c.execute("UPDATE attempts SET status='failed' WHERE key=?", (items[0][0],))
    with pytest.raises(ValueError, match='Original router ledger rows changed'):
        audit_cache(router.cache.path, auth)
    with pytest.raises(RouterCacheError, match='not authorized'):
        adapted(bundle).reserve_many_local(items[:1])


@pytest.mark.parametrize('table', ['protocol', 'decisions', 'cache_hits'])
def test_old_rows_cannot_be_changed(bundle, table):
    memory, router, _, _, _, auth = bundle
    with sqlite3.connect(router.cache.path) as c:
        if table == 'protocol': c.execute("UPDATE protocol SET hash='invalid'")
        elif table == 'decisions': c.execute("DELETE FROM decisions")
        else: c.execute("DELETE FROM cache_hits")
    with pytest.raises((ValueError, PermissionError)):
        audit_cache(router.cache.path, auth)


def test_call_budget_counts_original_and_reissued_attempts(bundle):
    _, _, _, items, _, _ = bundle
    cache = adapted(bundle); cache.max_api_calls = 6
    with pytest.raises(RouterBudgetExceeded): cache.reserve_many_local(items[:1])


def test_duplicate_and_api_reservations_are_forbidden(bundle):
    _, _, _, items, _, _ = bundle; cache = adapted(bundle)
    with pytest.raises(RouterCacheError, match='Duplicate'): cache.reserve_many_local(items[:1]*2)
    with pytest.raises(PermissionError, match='API'): cache.reserve(*items[0])


@pytest.mark.parametrize('change', ['model', 'device', 'api', 'path', 'budget', 'authorization'])
def test_constructor_rejects_scope_changes(bundle, tmp_path, change):
    _, router, _, _, _, auth = bundle
    protocol = copy.deepcopy(router.protocol); auth = copy.deepcopy(auth); path = router.cache.path; cap = 100
    if change == 'model': protocol['config']['model'] = 'policy-model'
    if change == 'device': protocol['device'] = 'cuda:0'
    if change == 'api': protocol['external_api_calls'] = 1
    if change == 'path': path = tmp_path/'foreign.sqlite3'
    if change == 'budget': cap = 101
    if change == 'authorization': auth['approved'] = False
    with pytest.raises(PermissionError): ExplicitLocalResumeCache(path, protocol, cap, authorization=auth)


def test_authorization_snapshot_and_plan_binding_are_required(bundle):
    _, _, _, _, path, auth = bundle
    assert load_authorization(path)['protocol_hash'] == auth['protocol_hash']
    plan_path = path.parent/'plan.json'; plan = read_json(plan_path)
    plan['router_resume']['sha256'] = 'wrong'; plan_path.write_text(json.dumps(plan))
    with pytest.raises(PermissionError): load_authorization(path)


def test_snapshot_corruption_is_rejected(bundle):
    _, _, _, _, path, auth = bundle
    Path(auth['snapshot']['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='snapshot changed'): load_authorization(path)


def test_install_scoped_to_this_evaluation_does_not_patch_api_cache(bundle, monkeypatch):
    from agent_system.memory import skillrl_embedding_batch_router as batch
    memory, router, _, _, path, auth = bundle
    original = batch.RouterCache; monkeypatch.setattr(batch, 'RouterCache', original)
    with pytest.raises(PermissionError): install(path, Path(auth['run_root']).parent, 0)
    with pytest.raises(PermissionError): install(path, auth['run_root'], 1)
    install(path, auth['run_root'], 0)
    instance = batch.RouterCache(router.cache.path, router.protocol, 100)
    assert isinstance(instance, ExplicitLocalResumeCache)
    from agent_system.memory import router_cache
    assert router_cache.RouterCache is RouterCache


def test_symlink_cache_is_rejected(bundle, tmp_path):
    _, router, _, _, _, auth = bundle
    link = tmp_path/'linked.sqlite3'; link.symlink_to(router.cache.path)
    with pytest.raises(ValueError, match='symlink'): audit_cache(link, auth)


def test_actual_five_query_snapshot_unchanged_before_launch():
    f = REPO/'artifacts/phase12/skillnet37-independent-s404-505-606-v4/all-first-calls-v1'
    receipt = read_json(f/'recovery-v1/active-monitor-handoff.json')
    assert len(receipt['router_cache']['incomplete_attempts']) == 5
    assert receipt['router_cache']['successful_decisions'] == 30851
    from skillnet_cohort.seed_queue import verify_sources
    plan = read_json(f/'recovery-v1/plan.json'); verify_sources(plan)
    assert len(plan['source_sha256']) == 114


def test_quiescent_preflight_blocks_new_interruptions_before_spawning(bundle, monkeypatch, tmp_path):
    from skillnet_cohort.first_calls_router_recovery import RouterResumeAssessment
    import skillnet_cohort.first_calls_router_recovery as controller
    memory, router, _, _, auth_path, auth = bundle
    state = request(memory, 30); state.pop('candidate_bundle'); visible = router._visible_input(**state)
    key = digest({'protocol_hash': router.protocol_hash, 'input_hash': digest(visible)})
    router.cache.reserve_many_local([(key, visible)])
    obj = RouterResumeAssessment.__new__(RouterResumeAssessment)
    obj.root = Path(auth['run_root']); obj.audit = tmp_path/'audit'; obj.plan = {'jobs': [{'seed': 404}]}
    obj.recovery = {'router_resume': {'path': str(auth_path)}}; obj.disk = lambda *a: None
    monkeypatch.setattr(controller.subprocess, 'Popen', lambda *a, **k: pytest.fail('No GPU child before cache preflight'))
    with pytest.raises(RouterCacheError, match='before GPU'):
        obj.commands([(['phase2.evaluate', '--root', obj.root, '--update', 0], 'utility', 0)])
    assert not (obj.audit/'logs').exists()


@pytest.mark.parametrize('seed,expected_module', [(404, 'skillnet_cohort.explicit_router_resume'), (505, 'phase2.evaluate')])
def test_only_seed404_evaluators_receive_resume_adapter(bundle, monkeypatch, tmp_path, seed, expected_module):
    from skillnet_cohort.first_calls_router_recovery import RouterResumeAssessment
    import skillnet_cohort.first_calls_router_recovery as controller
    from skillnet_cohort import runtime_watch
    _, router, _, _, auth_path, auth = bundle
    obj = RouterResumeAssessment.__new__(RouterResumeAssessment)
    obj.root = Path(auth['run_root']); obj.audit = tmp_path/'audit'; obj.plan = {'jobs': [{'seed': seed}]}
    obj.recovery = {'router_resume': {'path': str(auth_path)}}; obj.disk = lambda *a: None
    if seed == 505:
        obj.root = tmp_path/'seed-505'; obj.root.mkdir()
        _, clean, _ = runtime(obj.root, budget=100)
        write_new_json(obj.root/'protocol.json', {'runtime': {'cache_path': str(clean.cache.path)}})
    calls = []
    class Child:
        pid = 1234567; returncode = 0
        def poll(self): return 0
        def wait(self, timeout=None): return 0
    def spawn(argv, **kw):
        calls.append((argv, kw)); return Child()
    class Watch:
        def __init__(self, *a): pass
        def tick(self, *a, **kw): pass
    monkeypatch.setattr(controller.subprocess, 'Popen', spawn)
    monkeypatch.setattr(runtime_watch, 'StageWatch', Watch)
    obj.commands([(['phase2.evaluate', '--root', obj.root, '--update', 0, '--shard', 0, '--shards', 8], 'utility', 0)])
    assert len(calls) == 1 and calls[0][0][4] == expected_module
    assert ('--router-resume-authorization' in calls[0][0]) == (seed == 404)
    assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES'] == '0'


def test_workflow_class_adapter_is_always_restored(bundle, monkeypatch):
    from skillnet_cohort import first_calls_recovery, first_calls_router_recovery as controller
    _, _, _, _, auth_path, _ = bundle
    prior = auth_path.parent/'previous.json'; write_new_json(prior, {})
    path = auth_path.parent/'plan.json'; plan = read_json(path)
    plan['previous_attempt'] = {'path': str(prior), 'sha256': file_hash(prior)}
    path.write_text(json.dumps(plan))
    original = first_calls_recovery.RecoveryAssessment
    def run(path):
        assert first_calls_recovery.RecoveryAssessment is controller.RouterResumeAssessment
        raise RuntimeError('intentional test exit')
    monkeypatch.setattr(first_calls_recovery, 'run', run)
    with pytest.raises(RuntimeError, match='intentional'):
        controller.run(path)
    assert first_calls_recovery.RecoveryAssessment is original


def test_evaluator_wrapper_preserves_original_arguments(monkeypatch):
    import sys
    from skillnet_cohort import explicit_router_resume as wrapper
    from phase2 import evaluate
    events = []
    monkeypatch.setattr(wrapper, 'install', lambda p, r, u: events.append((str(p), str(r), u)))
    rest = ['--root', '/test/run', '--update', '0', '--shard', '5', '--shards', '8']
    monkeypatch.setattr(sys, 'argv', ['wrapper', '--router-resume-authorization', '/test/auth.json', *rest])
    monkeypatch.setattr(evaluate, 'main', lambda: events.append(sys.argv[1:]))
    wrapper.main()
    assert events == [('/test/auth.json', '/test/run', 0), rest]
