import copy
import json
from pathlib import Path
import time

import pytest

from skillnet_cohort.common import REPO, digest, file_hash, load_preparation, read_json, write_new_bytes, write_new_json
from skillnet_cohort.day_budget import PROFILE, OFFLOAD_PROFILE, apply_budget, load_budget, select_gold_skills, validate_admission, workload
from skillnet_cohort.evaluate import collect_results, execute_jobs, finalize_shards, job_plan, partition_plan, result_path
from skillnet_cohort.runtime import router_registration
from skillnet_cohort.training import configuration
from skillnet_cohort.segmented_training import block_config


BASE = REPO / 'docs/experiments/skillrl-embedding-router-v1/preparation-s404-8gpu/manifest.json'


@pytest.fixture
def preparation(tmp_path):
    spec = read_json(BASE.parent / 'spec.json')
    spec.update(router_registration('skillrl_embedding_state_batch', model_path='/not-loaded', device='cpu'))
    spec = apply_budget(spec, PROFILE)
    root = tmp_path / 'prepared'
    manifest = read_json(BASE)
    for asset in manifest['assets']:
        if asset['path'] == 'spec.json':
            write_new_json(root / asset['path'], spec)
        else:
            write_new_bytes(root / asset['path'], (BASE.parent / asset['path']).read_bytes())
        asset['sha256'] = file_hash(root / asset['path'])
    write_new_json(root / 'manifest.json', manifest)
    return root / 'manifest.json'


def test_confirmed_scope_has_exact_workload_and_keeps_recording(preparation, tmp_path):
    load_preparation(preparation)
    spec = read_json(preparation.parent / 'spec.json')
    count = workload(spec, read_json(preparation.parent / 'games.json'))
    assert count == dict(training_trajectories=640, monitor_trajectories_upper=128,
        performance_episodes=548, anchor_source_episodes=134, utility_continuations_upper=2592,
        local_router_calls_upper=202100, full_vocab_old_new_bytes_per_response_token=1986560,
        requires_measured_runtime_and_storage_admission=True)
    cfg = configuration(preparation, tmp_path / 'run')
    assert cfg.trainer.total_training_steps == 5
    assert cfg.actor_rollout_ref.rollout.micro_batch_size == 2
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert cfg.data.train_batch_size * cfg.env.rollout.n == 128
    assert cfg.phase2.enabled and cfg.phase2.progress_journal
    assert cfg.env.max_steps == 50 and cfg.data.max_response_length == 512
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6
    assert cfg.env.skills_only_memory.step_routing.backend == 'skillrl_embedding_state_batch'
    with pytest.raises(ValueError, match='Unregistered'):
        block_config(preparation, tmp_path / 'run', {'gpu_ids': list(range(8))}, 5)
    with pytest.raises(ValueError, match='Checkpoint'):
        job_plan(preparation, tmp_path / 'model', 10, 'valid_seen')
    with pytest.raises(ValueError, match='not registered'):
        job_plan(preparation, tmp_path / 'model', 0, 'valid_seen', 'anchors')


def test_gold_selection_does_not_depend_on_order_or_scores():
    settings = load_budget(PROFILE)
    ids = [f'skill-{i}' for i in range(37)]
    first = select_gold_skills(ids, settings)
    assert len(first) == 12 and first == select_gold_skills(ids[::-1], settings)
    assert set(select_gold_skills(ids[:3], settings)) == set(ids[:3])
    assert select_gold_skills(ids, {}) == sorted(ids)


def fake(job):
    return {'success': False, 'steps': [{'step_index': 0, 'projected_action': 'look'}]}


def test_full_evaluation_is_exact_union_of_eight_shards(preparation, tmp_path):
    plan = job_plan(preparation, tmp_path / 'model', 0, 'valid_unseen')
    output = tmp_path / 'evaluation'
    for shard in range(8):
        execute_jobs(partition_plan(plan, shard, 8), output / 'shards' / f'{shard:02d}', fake)
    result = finalize_shards(plan, output, 8)
    assert result['unique_games'] == result['episodes'] == 134
    rows, missing = collect_results(read_json(output / 'plan.json'), output)
    assert len(rows) == 134 and not missing
    assert result_path(output, rows[0]['job']['job_id']).is_file()
    # Collection is idempotent, but no episode is rerun.
    assert finalize_shards(plan, output, 8) == result


def test_incomplete_partition_never_claims_full_traversal(preparation, tmp_path):
    plan = job_plan(preparation, tmp_path / 'model', 0, 'valid_seen')
    output = tmp_path / 'partial'
    for shard in range(7):
        execute_jobs(partition_plan(plan, shard, 8), output / 'shards' / f'{shard:02d}', fake)
    with pytest.raises(FileNotFoundError):
        finalize_shards(plan, output, 8)
    assert not (output / 'completion.json').exists()


def test_deadline_alone_does_not_authorize_30h_run(preparation, tmp_path):
    spec = read_json(preparation.parent / 'spec.json')
    with pytest.raises(PermissionError, match='admission'):
        validate_admission({}, spec, preparation)
    evidence = {'status': 'PASS', 'preparation_sha256': file_hash(preparation),
        'router_profile_sha256': spec['router_profile_sha256'],
        'batch_router_validated': True, 'native_eight_gpu_microbatch_validated': True,
        'exact_capture_validated': True, 'sharded_evaluation_validated': True,
        'projected_total_upper_seconds': 50000, 'projected_peak_run_bytes': 1000000}
    path = tmp_path / 'synthetic-admission.json'
    write_new_json(path, evidence)
    now = time.time()
    permit = {'budget_admission': {'path': str(path), 'sha256': file_hash(path)},
        'storage': {'maximum_run_bytes': 2000000}, 'budget_started_unix': now - 10,
        'budget_deadline_unix': now + 90000}
    validate_admission(permit, spec, preparation)
    permit['budget_deadline_unix'] = now + 1000
    with pytest.raises(PermissionError, match='insufficient'):
        validate_admission(permit, spec, preparation)
    # A later authorized stage need not reserve the full original estimate
    # again. It still cannot execute after the unchanged absolute deadline.
    validate_admission(permit, spec, preparation, require_full_budget=False)
    permit['budget_deadline_unix'] = now - 1
    with pytest.raises(PermissionError, match='expired'):
        validate_admission(permit, spec, preparation, require_full_budget=False)


def test_optimizer_offload_is_versioned_without_changing_scientific_scope(preparation, tmp_path):
    before = load_budget(PROFILE)
    after = load_budget(OFFLOAD_PROFILE)
    assert {key for key in set(before) | set(after) if before.get(key) != after.get(key)} == {
        'profile_id', 'actor_optimizer_offload'}
    spec = apply_budget(read_json(preparation.parent / 'spec.json'), OFFLOAD_PROFILE)
    root = tmp_path / 'offload-preparation'
    manifest = read_json(preparation)
    for asset in manifest['assets']:
        if asset['path'] == 'spec.json':
            write_new_json(root / asset['path'], spec)
        else:
            write_new_bytes(root / asset['path'], (preparation.parent / asset['path']).read_bytes())
        asset['sha256'] = file_hash(root / asset['path'])
    write_new_json(root / 'manifest.json', manifest)
    cfg = configuration(root / 'manifest.json', tmp_path / 'new-run')
    assert cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload is True
    assert cfg.actor_rollout_ref.actor.fsdp_config.param_offload is False
    assert cfg.actor_rollout_ref.ref.fsdp_config.param_offload is True
    assert cfg.actor_rollout_ref.rollout.micro_batch_size == 2
    assert cfg.trainer.total_training_steps == 5
    assert configuration(preparation, tmp_path / 'old-run').actor_rollout_ref.actor.fsdp_config.optimizer_offload is False


def test_partial_rollout_is_journaled_without_inventing_advantage(tmp_path):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from verl import DataProto
    from phase2.capture import attach_decisions
    batch = DataProto(TensorDict({'responses': torch.tensor([[11,12,0], [21,0,0]]),
        'attention_mask': torch.tensor([[1,1,1,0], [1,1,0,0]])}, batch_size=[2]),
        non_tensor_batch={'traj_uid': np.array(['a','b'], dtype=object),
            'uid': np.array(['group','group'], dtype=object),
            'active_masks': np.array([True, False]), 'rewards': np.array([1.,0.])})
    attach_decisions(batch, [{'projected_action':'look'},{'projected_action':'look'}],
        run_id='synthetic-test', update=1, step=0, journal_root=tmp_path)
    saved = read_json(tmp_path / 'rollout_progress/u0001/step-0000.json')
    assert not saved['complete_rollout'] and len(saved['records']) == 1
    row = saved['records'][0]
    assert row['response_token_ids'] == [11,12] and row['reward'] == 1
    assert row['advantage_status'] == 'pending_complete_group_rollout'
    assert 'advantage' not in row


def test_hard_deadline_is_a_stop_not_a_completion():
    from skillnet_cohort.run import Pipeline
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.deadline = time.time() - 1
    with pytest.raises(TimeoutError):
        pipeline.check_deadline()
    pipeline.deadline = None
    pipeline.check_deadline()


def test_every_budget_execution_operation_requires_measured_admission(preparation, tmp_path):
    from skillnet_cohort.common import require_authorization
    permit = tmp_path / 'not-ready.json'
    write_new_json(permit, {'approved': True, 'preparation_sha256': file_hash(preparation),
        'operations': ['training','evaluation','readout','exports'], 'router_max_local_calls': 202100})
    for operation in ('training','evaluation','readout','exports'):
        with pytest.raises(PermissionError, match='admission'):
            require_authorization(permit, preparation, operation)
