import json
from types import SimpleNamespace as NS

import pytest

from phase3.api import APIConfig, JSONClient
from phase3.bank import Bank, Skill
from phase3.common import ProtocolError, digest, strict_json
from phase3.editor import choose_evidence, propose
from phase3.editor_model_handoff import migrate_unused_ledger
from phase3.evolution import evolve, paired_effect
from phase3.readout import WindowIdentity
from phase3.routing import BranchRouter


def bank():
    return Bank('test', [Skill(sid, sid, 'description', f'### {sid}\n\nGuidance') for sid in ('a', 'b', 'c')], source='0'*64)


def operation(kind, b, ids=()):
    return {'op': kind, 'targets': [{'skill_id': sid, 'version_sha256': b.active_versions[sid]} for sid in ids],
            'skill': None if kind in ('NOOP', 'DELETE') else {'name': 'new', 'description': 'generic guidance', 'body': 'Reusable action guidance.'},
            'rationale': 'observed issue', 'evidence_ids': ['e']}


@pytest.mark.parametrize('kind,ids,units,size', [('ADD', (), 1, 4), ('MODIFY', ('a',), 1, 3),
    ('DELETE', ('a',), 1, 2), ('MERGE', ('a','b'), 3, 2), ('NOOP', (), 0, 3)])
def test_operations_and_immutable_snapshots(tmp_path, kind, ids, units, size):
    b = bank()
    before = b.manifest_sha256
    result, changes = b.apply({'operations': [operation(kind, b, ids)]}, event_id='u5', evidence_ids={'e'})
    assert b.manifest_sha256 == before and len(result) == size and changes['mutation_units'] == units
    path = result.save(tmp_path)
    assert Bank.load(path, result.manifest_sha256).record() == result.record()
    if kind != 'NOOP':
        assert result.parent == before
    if kind == 'MODIFY':
        assert result.active_versions['a'] != b.active_versions['a'] and len(result.inactive) == 1


def test_budgets_tombstones_and_stale_edits():
    b = bank()
    with pytest.raises(ProtocolError):
        b.apply({'operations': [operation('MERGE', b, ('a','b','c'))]}, event_id='u5', evidence_ids={'e'})
    op = operation('MODIFY', b, ('a',))
    changed, _ = b.apply({'operations': [op]}, event_id='u5', evidence_ids={'e'})
    with pytest.raises(ProtocolError, match='Stale'):
        changed.apply({'operations': [op]}, event_id='u10', evidence_ids={'e'})
    deleted, _ = b.apply({'operations': [operation('DELETE', b, ('a',))]}, event_id='u5', evidence_ids={'e'})
    assert 'a' in deleted.retired_ids and len(deleted.inactive) == 1
    rollback = deleted.rollback_to(b, event_id='rollback-u5')
    assert rollback.active_versions == b.active_versions and rollback.manifest_sha256 != b.manifest_sha256


class FakeClient:
    def __init__(self, value):
        self.value, self.calls = value, []
        self.chat = NS(completions=NS(create=self.create))
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.value, Exception):
            raise self.value
        return NS(choices=[NS(finish_reason='stop', message=NS(content=json.dumps(self.value), refusal=None, tool_calls=None))],
            usage=NS(prompt_tokens=300, completion_tokens=20, total_tokens=320, prompt_tokens_details=None,
                     completion_tokens_details=None), model=kwargs['model'], id='test', _request_id='test', system_fingerprint=None)


def api(tmp_path, value, stage='editor', budget=5, model=None):
    fake = FakeClient(value)
    config = APIConfig(stage, model or ('gpt-5.5' if stage == 'editor' else 'gpt-5.4-mini'),
                       100000, 8192 if stage == 'editor' else 128, budget)
    return JSONClient(config, tmp_path / f'{stage}.sqlite3', client=fake, token_counter=lambda _: 50), fake


def episode(b):
    return {'trajectory_id': 'e', 'game_id': 'train/game', 'split': 'train', 'task': 'Put the object', 'success': False,
            'bank_sha256': b.manifest_sha256, 'global_update': 1,
            'steps': [{'step_index': 0, 'observation': 'here', 'action': 'look',
            'next_observation': 'here', 'selected_skill_id': 'a', 'skill_version_sha256': b.active_versions['a'], 'is_action_valid': True}]}


def test_editor_sees_only_selected_skills_and_complete_associated_trajectories(tmp_path):
    b = bank()
    raw = []
    for number in range(8):
        item = episode(b)
        item['trajectory_id'] = f'e{number}'
        item['steps'] = [{**item['steps'][0], 'step_index': step,
            'selected_skill_id': 'b' if step == 17 else 'a',
            'observation': f'location {step}: ' + 'long observation ' * 12,
            'next_observation': f'next {step}: ' + 'long consequence ' * 12}
            for step in range(50)]
        raw.append(item)
    original = digest(raw)
    evidence = choose_evidence(raw, selector='reward_sign_balance', priority_ids=['b'],
                               source_update=1, allowed_games={'train/game'})
    noop = operation('NOOP', b)
    noop['evidence_ids'] = ['e0']
    client, fake = api(tmp_path, {'operations': [noop]})
    result = propose(b, api=client, event_id='u5', evidence=evidence, priority_ids=['b'])
    assert digest(raw) == original and result['input']['evidence'] == evidence
    assert len(result['input']['candidate_skills']) == 1
    assert result['input']['candidate_skills'][0]['skill_id'] == 'b'
    assert 'bank' not in result['input'] and 'a' not in result['input']['priority_targets']
    assert all(len(item['steps']) == 50 for item in result['input']['evidence'])
    assert len(fake.calls) == 1


def test_failure_evidence_uses_only_old_policy_initial_batch_failures():
    b = bank()
    one = episode(b)
    one['steps'] = [{**one['steps'][0], 'step_index': index,
                     'observation': 'o' * 300} for index in range(8)]
    two = episode(b)
    two['trajectory_id'] = 'e2'
    two['split'] = 'valid_seen'
    earlier = episode(b)
    earlier['trajectory_id'] = 'e3'
    earlier['global_update'] = 2
    success = episode(b)
    success['trajectory_id'] = 'e4'
    success['success'] = True
    evidence = choose_evidence([one, two, earlier, success], selector='failure_driven', priority_ids=[],
                               source_update=1, allowed_games={'train/game'})
    assert {item['evidence_id'] for item in evidence} == {'e'}
    complete = next(item for item in evidence if item['evidence_id'] == 'e')
    assert len(complete['steps']) == 8
    assert all(len(step['observation']) == 300 for step in complete['steps'])
    assert all('selected_skill_id' in step and 'next_observation' in step for step in complete['steps'])


def test_readout_evidence_reverse_selects_all_matching_endpoint_failures():
    b = bank()
    rows = []
    for eid, calls, success in [('e1', 1, False), ('e2', 3, False), ('e3', 3, False), ('e4', 5, True)]:
        item = episode(b)
        item['trajectory_id'], item['success'] = eid, success
        item['steps'] = [{**item['steps'][0], 'step_index': i} for i in range(calls)]
        rows.append(item)
    evidence = choose_evidence(rows, selector='reward_sign_balance', priority_ids=['a'],
                               source_update=1, allowed_games={'train/game'})
    assert [item['evidence_id'] for item in evidence] == ['e1', 'e2', 'e3']
    assert all(not item['success'] for item in evidence)


def test_readout_evidence_has_no_eight_trajectory_cap():
    b = bank()
    rows = []
    for index in range(12):
        sid, eid, calls = ('a', f'e{index:02d}', 50)
        item = episode(b)
        item['trajectory_id'] = eid
        item['steps'] = [{**item['steps'][0], 'selected_skill_id': sid, 'step_index': index}
                         for index in range(calls)]
        rows.append(item)
    evidence = choose_evidence(rows, selector='reward_sign_balance', priority_ids=['a'],
                               source_update=1, allowed_games={'train/game'})
    assert len(evidence) == 12
    assert all(not item['success'] for item in evidence)


@pytest.mark.parametrize('model', ['o3', 'gpt-5.5'])
def test_editor_payload_cache_and_no_temperature(tmp_path, model):
    b = bank()
    client, fake = api(tmp_path, {'operations': [operation('NOOP', b)]}, model=model)
    failed = episode(b)
    evidence = choose_evidence([failed], selector='failure_driven', priority_ids=[],
                               source_update=1, allowed_games={'train/game'})
    first = propose(b, api=client, event_id='u5', evidence=evidence, priority_ids=['a'])
    second = propose(b, api=client, event_id='u5', evidence=evidence, priority_ids=['a'])
    assert len(fake.calls) == 1 and fake.calls[0]['model'] == model
    assert 'temperature' not in fake.calls[0] and fake.calls[0]['reasoning_effort'] == 'medium'
    assert first['accounting']['api_calls'] == 1 and second['accounting']['api_calls'] == 0
    assert second['accounting']['latency_seconds'] == 0
    assert 'branch' not in first['input'] and 'selector' not in first['input']


def test_api_failure_does_not_retry_or_log_exception_text(tmp_path):
    client, fake = api(tmp_path, RuntimeError('sensitive-private-header'))
    args = dict(identity={}, system='x', payload={}, schema={}, validate=lambda _: None)
    for _ in range(2):
        with pytest.raises(ProtocolError):
            client.request(**args)
    assert len(fake.calls) == 1 and b'sensitive-private-header' not in client.path.read_bytes()


def test_editor_local_input_cap_not_enforced_but_router_cap_is(tmp_path):
    editor, editor_fake = api(tmp_path, {}, stage='editor')
    editor.token_counter = lambda _: 200000
    result, accounting = editor.request(identity={'x': 1}, system='x', payload={}, schema={}, validate=lambda _: None)
    assert result == {} and accounting['input_token_estimate'] == 200256
    assert len(editor_fake.calls) == 1
    router, router_fake = api(tmp_path, {'skill_id': 'a'}, stage='router')
    router.token_counter = lambda _: 200000
    with pytest.raises(ProtocolError, match='Router input estimate'):
        router.request(identity={'x': 1}, system='x', payload={}, schema={}, validate=lambda _: None)
    assert len(router_fake.calls) == 0


def test_unused_o3_ledger_can_switch_with_history(tmp_path):
    old, _ = api(tmp_path, {}, model='o3')
    receipt = migrate_unused_ledger(old.path)
    assert receipt['from_model'] == 'o3' and receipt['to_model'] == 'gpt-5.5'
    config = APIConfig('editor', 'gpt-5.5', 100000, 8192, 5)
    replacement = JSONClient(config, old.path, client=FakeClient({}), token_counter=lambda _: 50)
    with replacement.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM profile_migrations').fetchone()[0] == 1
        assert strict_json(db.execute('SELECT old_value FROM profile_migrations').fetchone()[0])['model'] == 'o3'


def test_used_o3_ledger_cannot_switch(tmp_path):
    old, _ = api(tmp_path, {}, model='o3')
    old.request(identity={'x': 1}, system='x', payload={}, schema={}, validate=lambda _: None)
    with pytest.raises(ProtocolError, match='after an API reservation'):
        migrate_unused_ledger(old.path)


def test_api_cap_and_corrupt_cache(tmp_path):
    client, fake = api(tmp_path, {}, budget=1)
    args = dict(system='x', payload={}, schema={}, validate=lambda _: None)
    client.request(identity={'x':1}, **args)
    with pytest.raises(ProtocolError, match='exhausted'):
        client.request(identity={'x':2}, **args)
    with client.connection() as db:
        result = strict_json(db.execute('SELECT result FROM attempts').fetchone()[0])
        result['value'] = {'changed': True}
        db.execute('UPDATE attempts SET result=?', (json.dumps(result),))
    with pytest.raises(ProtocolError, match='Changed'):
        client.request(identity={'x':1}, **args)
    assert len(fake.calls) == 1


def test_router_complete_catalog_and_changed_bank_invalidate_cache(tmp_path):
    b = bank()
    client, fake = api(tmp_path, {'skill_id':'a'}, stage='router')
    first = BranchRouter(b, client)
    visible = dict(task_description='task', current_observation='obs', admissible_actions=['look'], history=[], step_index=0)
    out = first.route(first.memory.retrieve('task'), **visible)
    assert out['skill_version_sha256'] == b.active_versions['a']
    changed, _ = b.apply({'operations': [operation('MODIFY', b, ('a',))]}, event_id='u5', evidence_ids={'e'})
    second = BranchRouter(changed, client)
    second.route(second.memory.retrieve('task'), **visible)
    assert len(fake.calls) == 2
    candidates = second.memory.retrieve('task')
    candidates['candidate_skill_ids'].pop()
    with pytest.raises(ProtocolError):
        second.route(candidates, **visible)


@pytest.mark.parametrize('after,accepted', [(True,True), (False,False)])
def test_evolve_pair_gate_and_resume(tmp_path, after, accepted):
    b = bank()
    client, fake = api(tmp_path, {'operations': [operation('MODIFY', b, ('a',))]})
    def evaluate(candidate):
        return [{'game_id':'gate', 'eval_seed':1, 'success':True if candidate is b else after,
                 'policy_sha256':'2'*64, 'bank_sha256':candidate.manifest_sha256}]
    failed_validation = episode(b)
    identity = WindowIdentity('test', b.manifest_sha256, '1'*64, '2'*64, 0, 5)
    kwargs = dict(output=tmp_path / 'event', event_id='u5', selector='failure_driven', api=client, episodes=[failed_validation],
        evidence_games={'train/game'}, gate_games={'gate'}, max_evidence_trajectories=8,
        tolerance_pp=0, evaluate=evaluate, identity=identity)
    selected, record = evolve(b, **kwargs)
    assert record['accepted'] is accepted and record['rollback_count'] == 0
    assert (selected.manifest_sha256 != b.manifest_sha256) is accepted
    assert record['candidate_count'] == len(b)
    assert record['source_artifact'] == 'source-v13.json'
    assert record['editor_protocol'] == 'same_old_policy_batch_as_readout_v13'
    source = strict_json((tmp_path / 'event' / 'source-v13.json').read_text())
    assert source['sampling_policy_update'] == 0
    assert source['readout_batch_update'] == source['evidence_global_update'] == 1
    assert '"bank"' in fake.calls[0]['messages'][1]['content']
    assert '"candidate_skills"' not in fake.calls[0]['messages'][1]['content']
    assert evolve(b, **kwargs)[1] == record and len(fake.calls) == 1
    kwargs['episodes'][0]['bank_sha256'] = '9'*64
    with pytest.raises(ProtocolError):
        evolve(b, **kwargs)


def test_evolve_rejects_postupdate_validation_and_later_training_evidence_before_editor(tmp_path):
    b = bank()
    client, fake = api(tmp_path, {'operations': [operation('NOOP', b)]})
    identity = WindowIdentity('test', b.manifest_sha256, '1'*64, '2'*64, 0, 5)
    kwargs = dict(output=tmp_path / 'event', event_id='u5', selector='failure_driven', api=client,
                  evidence_games={'train/game'}, gate_games={'gate'}, max_evidence_trajectories=8,
                  tolerance_pp=0, evaluate=lambda _: [], identity=identity)
    for split, update in [('valid_seen', 1), ('train', 2)]:
        item = episode(b)
        item['split'], item['global_update'] = split, update
        with pytest.raises(ProtocolError, match='old-policy initial-batch training'):
            evolve(b, episodes=[item], **kwargs)
    assert fake.calls == []
    assert not (tmp_path / 'event' / 'source-v13.json').exists()


def test_gate_pairing_and_no_unseen_evidence():
    row = {'game_id':'g', 'eval_seed':1, 'success':True}
    with pytest.raises(ProtocolError):
        paired_effect([row], [{**row, 'eval_seed':2}])
    e = episode(bank())
    e['split'] = 'valid_unseen'
    with pytest.raises(ProtocolError):
        choose_evidence([e], selector='failure_driven', priority_ids=[], source_update=1,
                        allowed_games={'train/game'})
