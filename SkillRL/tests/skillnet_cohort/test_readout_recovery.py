"""CPU-only regressions: no model API, GPU, optimizer or environment rollout."""
import json
import sys
import time

import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from skillnet_cohort.common import file_hash, read_json, write_new_bytes, write_new_json


def test_fp32_master_delta_deduplicates_tied_head_and_rejects_lossy_export(tmp_path):
    from skillnet_cohort.parameter_delta import compare_state
    old = {'model.embed_tokens.weight': torch.ones(2, 2), 'lm_head.weight': torch.ones(2, 2),
           'model.norm.weight': torch.ones(2)}
    new = {k: v+1 for k, v in old.items()}
    save_file(new, tmp_path/'model.safetensors')
    result = compare_state(old, tmp_path)
    assert result['unique_parameters'] == 6
    assert result['delta_l2'] == pytest.approx(6**.5)
    assert result['relative_delta_l2'] == 1
    with pytest.raises(ValueError, match='FP32'):
        compare_state({k: v.bfloat16() for k, v in old.items()}, tmp_path)
    new['lm_head.weight'][0, 0] += 1
    save_file(new, tmp_path/'model.safetensors')
    with pytest.raises(ValueError, match='tied'):
        compare_state(old, tmp_path)


@pytest.fixture
def aggregate_fixture(tmp_path, monkeypatch):
    from phase2 import aggregate
    from phase2.protocol import signal_directory
    config = {'capture_scope': 'window_start_old_only_v1',
              'window_direction': 'start_batch_endpoint_projection', 'parent_update': 0}
    write_new_json(tmp_path/'protocol.json', config)
    write_new_json(tmp_path/'batches/u0001/manifest.json', {'row_count': 1})
    write_new_json(tmp_path/'old_logprobs/u0001/row-000000.pt', {})
    out = signal_directory(tmp_path, 5, 0)
    write_new_json(out/'shard-0.json', {'max_decisions': None, 'start_update': 0,
        'direction_batch_update': 1, 'repeat_forward_noise': [0.]})
    pd.DataFrame([{'decision_id': 'one', 'control': 'null', 'response_token_offset': 0}]).to_parquet(out/'tokens-shard-0.parquet')
    pd.DataFrame([{'decision_id': 'one'}]).to_parquet(out/'decisions-shard-0.parquet')
    for update in range(1, 6):
        write_new_bytes(tmp_path/'optimizer_steps'/f'u{update:04d}-rank0.jsonl',
            (json.dumps({'adam_step_before': update-1, 'adam_step_after': update})+'\n').encode())
    monkeypatch.setattr('phase2.audit.audit', lambda *a: None)
    monkeypatch.setattr('phase2.window_evidence.validate_shards', lambda *a: {'status': 'PASS'})
    monkeypatch.setattr('skillnet_cohort.parameter_delta.registered_initial_delta', lambda *a: {'delta_l2': 1.})
    monkeypatch.setattr(aggregate, 'aggregate_features', lambda t, *a: (pd.DataFrame([{
        'phase': 'all', 'control': 'null', 'skill_id': 'skill', 'supported': True,
        'P_int': -.1, 'D_contribution': .1, 'gate_coverage': 1.}]), t))
    monkeypatch.setattr(sys, 'argv', ['aggregate', '--root', str(tmp_path), '--update', '5',
                                     '--start-update', '0', '--shards', '1'])
    return aggregate, out


def test_window_aggregate_accepts_old_only_and_preserves_committed_files(aggregate_fixture):
    aggregate, out = aggregate_fixture
    aggregate.main()
    committed = read_json(out/'committed.json')
    assert committed['live_start_batch_rows'] == 1 and not committed['gold_read']
    assert 'live_old_and_new_rows' not in committed
    assert committed['readout_kind'] == 'start_batch_endpoint_projection'
    before = file_hash(out/'committed.json')
    aggregate.main()  # A completed aggregate is not overwritten.
    assert file_hash(out/'committed.json') == before


def test_window_aggregate_refuses_unregistered_projection(aggregate_fixture, tmp_path):
    aggregate, out = aggregate_fixture
    config = read_json(tmp_path/'protocol.json')
    config.pop('window_direction')
    (tmp_path/'protocol.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='registered window direction'):
        aggregate.main()
    assert not (out/'committed.json').exists()


def test_single_update_still_requires_live_new(tmp_path, monkeypatch):
    from phase2 import aggregate
    from phase2.protocol import signal_directory
    write_new_json(tmp_path/'protocol.json', {})
    write_new_json(tmp_path/'batches/u0001/manifest.json', {'row_count': 1})
    write_new_json(tmp_path/'old_logprobs/u0001/row-000000.pt', {})
    out = signal_directory(tmp_path, 1)
    write_new_json(out/'shard-0.json', {'max_decisions': None})
    pd.DataFrame([{'decision_id': 'one', 'control': 'null', 'response_token_offset': 0}]).to_parquet(out/'tokens-shard-0.parquet')
    pd.DataFrame([{'decision_id': 'one'}]).to_parquet(out/'decisions-shard-0.parquet')
    monkeypatch.setattr('phase2.audit.audit', lambda *a: None)
    monkeypatch.setattr(sys, 'argv', ['aggregate', '--root', str(tmp_path), '--update', '1', '--shards', '1'])
    with pytest.raises(ValueError, match='Missing new live logit rows'):
        aggregate.main()
    assert not (out/'committed.json').exists()


def test_readout_recovery_starts_at_aggregate_and_never_repeats_old_work(tmp_path, monkeypatch):
    from skillnet_cohort.run import Pipeline
    p = Pipeline.__new__(Pipeline)
    p.root, p.audit_root = tmp_path/'source', tmp_path/'attempt'
    p.preparation, p.authorization = tmp_path/'preparation.json', tmp_path/'permit.json'
    for path in (p.preparation, p.authorization, p.root/'metrics/u0005.json', p.root/'launch.json'):
        write_new_json(path, {})
    p.post_training = None
    p.recovery = p.readout_recovery = {'windows': [str(p.root/'windows/u0000-u0005-valid_unseen')]}
    p.router_backend, p.deadline = 'skillrl_embedding_state', None
    p.spec = {'training': {'iterations': 5}, 'seed': 404, 'readout': {},
              'evaluation': {'splits': ['valid_seen', 'valid_unseen'], 'utility_splits': ['valid_unseen']}}
    p.permit = {'gpu_ids': list(range(8))}
    p.budget = lambda: {'paid_router_cost': 0}
    events = []
    p.command = lambda args, label, **kw: events.append(('command', args[0]))
    p.parallel = lambda module, window, update, *a: events.append(('parallel', module, update))
    p.evaluate = lambda update, split, purpose, *a: events.append(('evaluate', update, split, purpose))
    monkeypatch.setattr('phase1.watch_qwen35_checkpoints.validate_full_checkpoint', lambda p: {'world_size': 8})
    monkeypatch.setattr('skillnet_cohort.support.build_support', lambda *a: pytest.fail('repeated support'))
    monkeypatch.setattr('skillnet_cohort.support.register_window', lambda *a: pytest.fail('repeated window registration'))
    monkeypatch.setattr('skillnet_cohort.window_storage.seal_window', lambda *a: events.append(('seal',)))
    monkeypatch.setattr('skillnet_cohort.window_storage.reclaim_proven_rows', lambda *a: {'deleted': [], 'retained': []})
    p.run()
    assert events == [('command', 'phase2.aggregate'), ('command', 'phase2.window_forecast'),
        ('parallel', 'phase2.evaluate', 5), ('command', 'phase2.window_report'), ('seal',),
        ('evaluate', 5, 'valid_seen', 'performance'), ('evaluate', 5, 'valid_unseen', 'performance')]
    assert read_json(p.root/'complete.json')['endpoint'] == 5
    with pytest.raises(FileExistsError, match='already attempted'):
        p.run()


def test_replacement_permit_binds_original_without_modifying_it(tmp_path, monkeypatch):
    from skillnet_cohort.common import require_authorization
    prep, old, new = tmp_path/'prep/manifest.json', tmp_path/'old.json', tmp_path/'new.json'
    write_new_json(prep, {})
    write_new_json(prep.parent/'spec.json', {})
    write_new_json(old, {'approved': False})
    old_sha = file_hash(old)
    permit = {'approved': True, 'preparation_sha256': file_hash(prep), 'run_root': str(tmp_path/'run'),
        'operations': ['readout'], 'readout_recovery': {'original_authorization': str(old),
            'original_authorization_sha256': old_sha, 'source_root': str(tmp_path/'run')}}
    write_new_json(new, permit)
    monkeypatch.setenv('SKILLNET_PHASE12_RECOVERY_AUTHORIZATION', str(new))
    assert require_authorization(old, prep, 'readout') == permit
    assert file_hash(old) == old_sha
    with pytest.raises(PermissionError, match='replacement'):
        require_authorization(old, prep, 'training')
    foreign = tmp_path/'foreign.json'
    write_new_json(foreign, {'approved': False})
    with pytest.raises(PermissionError):
        require_authorization(foreign, prep, 'readout')
    old.write_text('{}')
    with pytest.raises(PermissionError, match='replacement'):
        require_authorization(old, prep, 'readout')


def test_unlimited_time_still_requires_bound_plan_and_storage(tmp_path):
    from skillnet_cohort.day_budget import validate_admission
    from skillnet_cohort.seed_queue import can_start
    prep, admission, plan_path = tmp_path/'prep.json', tmp_path/'admission.json', tmp_path/'plan.json'
    write_new_json(prep, {})
    spec = {'router_profile_sha256': 'router', 'budget_profile': {'settings': {
        'hard_limit_seconds': 108000, 'finish_reserve_seconds': 10800}}}
    evidence = {'status': 'PASS', 'preparation_sha256': file_hash(prep),
        'router_profile_sha256': 'router', 'projected_total_upper_seconds': 200000,
        'projected_peak_run_bytes': 1000, **{key: True for key in ('batch_router_validated',
            'native_eight_gpu_microbatch_validated', 'exact_capture_validated', 'sharded_evaluation_validated')}}
    write_new_json(admission, evidence)
    limits = {'maximum_run_bytes': 2000, 'minimum_free_bytes': 1000}
    plan = {'approved': True, 'hard_limit_seconds': None, 'storage': limits,
        'jobs': [{'preparation_sha256': file_hash(prep), 'run_root': str(tmp_path/'run')}],
        'runtime_limit_override': {'unlimited_wallclock_user_authorized': True, 'storage_limits_unchanged': True}}
    write_new_json(plan_path, plan)
    permit = {'unlimited_wallclock': True, 'shared_queue_plan': str(plan_path),
        'shared_queue_plan_sha256': file_hash(plan_path), 'run_root': str(tmp_path/'run'),
        'budget_started_unix': time.time()-120000, 'budget_deadline_unix': None,
        'budget_admission': {'path': str(admission), 'sha256': file_hash(admission)}, 'storage': dict(limits)}
    validate_admission(permit, spec, prep)
    assert can_start(None, 200000, 10800)
    permit['storage']['minimum_free_bytes'] = 0
    with pytest.raises(PermissionError, match='storage protections'):
        validate_admission(permit, spec, prep)
    permit['storage'] = dict(limits)
    permit.pop('shared_queue_plan')
    with pytest.raises(PermissionError, match='amended queue'):
        validate_admission(permit, spec, prep)


def test_readout_recovery_cumulative_runtime_not_double_counted(tmp_path):
    from skillnet_cohort.readout_recovery import consumed_seconds
    prior = tmp_path/'recovery-v3'
    write_new_json(prior/'seed-404-exit.json', {'exit_code': 1, 'completed': False, 'elapsed_seconds': 48595.2})
    write_new_json(prior/'seed-404/supervisor_launch.json', {'started_unix': 110.8})
    write_new_json(prior/'queue_launch.json', {'actual_attempt_started_unix': 100.})
    assert consumed_seconds(tmp_path) == pytest.approx(48606.)


@pytest.fixture
def tiny_actual_shard(tmp_path, monkeypatch):
    from phase2 import window_evidence as module
    monkeypatch.setattr(module, 'validate_extended', lambda *a: None)
    monkeypatch.setattr(module, 'controls_by_skill', lambda *a: {'skill': {}})
    config = {'capture_scope': 'window_start_old_only_v1',
        'window_direction': 'start_batch_endpoint_projection',
        'windows': [{'start': 0, 'end': 5}], 'evaluation': {'shards': 1}}
    write_new_json(tmp_path/'protocol.json', config)
    meta = {'decision_id': 'decision', 'trajectory_id': 'trajectory',
            'info': {'selected_skill_id': 'skill', 'extra.gamefile': 'game'}}
    batch = {'metadata': [meta], 'tensors': {
        'phase2_actual_loss_mask': torch.tensor([[1, 0, 1]]),
        'responses': torch.tensor([[10, 0, 12]]), 'advantages': torch.tensor([[.5, 0, .5]])}}
    b = tmp_path/'batches/u0001'
    b.mkdir(parents=True)
    torch.save(batch, b/'training_batch.pt')
    write_new_json(b/'manifest.json', {'row_count': 1, 'batch_sha256': file_hash(b/'training_batch.pt')})
    write_new_json(tmp_path/'old_logprobs/u0001/row-000000.pt', {})
    out = tmp_path/'window_signals/u0000-u0005'; out.mkdir(parents=True)
    records, decisions = [], []
    for control in ('placebo', 'null'):
        record = {'decision_id': 'decision', 'control': control, 'global_update': 5, 'start_update': 0,
            'direction_batch_update': 1, 'window_horizon': 5, 'row_index': 0,
            'trajectory_id': 'trajectory', 'game_id': 'game', 'skill_id': 'skill'}
        decisions.append({**record, 'token_count': 2})
        records += [{**record, 'response_token_offset': i, 'action_token_id': 10+i, 'advantage': .5} for i in (0, 2)]
    pd.DataFrame(records).to_parquet(out/'tokens-shard-0.parquet')
    pd.DataFrame(decisions).to_parquet(out/'decisions-shard-0.parquet')
    write_new_json(out/'shard-0.json', {'shard': 0, 'update': 5, 'start_update': 0,
        'direction_batch_update': 1, 'max_decisions': None, 'readout_kind': 'start_batch_endpoint_projection',
        'decisions': 1, 'tokens_with_controls': 4, 'repeat_forward_noise': [0.],
        'tokens_sha256': file_hash(out/'tokens-shard-0.parquet')})
    witness = {k: torch.ones(2, 2) for k in ('old_original_live', 'end_original_replay',
        'old_placebo', 'new_placebo', 'old_null', 'new_null')}
    witness.update(metadata=meta, positions=torch.tensor([0, 2]), token_ids=torch.tensor([10, 12]),
                   advantage=torch.tensor([.5, .5]))
    torch.save(witness, out/'witness-shard-0.pt')
    return module, out


def test_actual_shard_validator_requires_complete_registered_evidence(tiny_actual_shard, tmp_path):
    module, out = tiny_actual_shard
    result = module.validate_shards(tmp_path, 0, 5, 1)
    assert result['eligible_decisions'] == 1 and result['new_live_rows_required'] is False
    with pytest.raises(ValueError, match='frozen window'):
        module.validate_shards(tmp_path, 0, 5, 2)
    tokens = pd.read_parquet(out/'tokens-shard-0.parquet')
    tokens.loc[0, 'advantage'] = -.5
    tokens.to_parquet(out/'tokens-shard-0.parquet')
    with pytest.raises(ValueError, match='measurement shard'):
        module.validate_shards(tmp_path, 0, 5, 1)
    manifest = read_json(out/'shard-0.json')
    manifest['tokens_sha256'] = file_hash(out/'tokens-shard-0.parquet')
    (out/'shard-0.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='actions/advantages'):
        module.validate_shards(tmp_path, 0, 5, 1)


def test_actual_shard_validator_rejects_old_style_live_new_witness(tiny_actual_shard, tmp_path):
    module, out = tiny_actual_shard
    witness = torch.load(out/'witness-shard-0.pt', weights_only=False)
    witness['new_original_live'] = witness.pop('end_original_replay')
    torch.save(witness, out/'witness-shard-0.pt')
    with pytest.raises(ValueError, match='witness'):
        module.validate_shards(tmp_path, 0, 5, 1)
