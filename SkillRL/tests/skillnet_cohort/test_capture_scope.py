import copy

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from phase2.capture import attach_decisions, capture_old_logits, mark_batch
from skillnet_cohort.capture_scope import WINDOW_START, archive_rollout_summary, capture_post, full_capture
from skillnet_cohort.common import REPO, file_hash, read_json, write_new_bytes, write_new_json
from skillnet_cohort.day_budget import INDEPENDENT_PROFILE, apply_budget
from skillnet_cohort.lossless_tensor import load, shuffled_profile
from skillnet_cohort.training import configuration
from verl import DataProto


def test_fixed_start_batches_not_cumulative_or_each_update():
    settings = {'capture_scope': WINDOW_START, 'capture_updates': [1, 6, 11]}
    assert [u for u in range(1, 16) if full_capture(settings, u)] == [1, 6, 11]
    assert not capture_post(settings)
    assert full_capture({}, 2) and capture_post({})
    with pytest.raises(ValueError):
        full_capture({'capture_scope': 'unknown'}, 1)
    with pytest.raises(ValueError):
        full_capture({'capture_scope': WINDOW_START}, 1)


@pytest.mark.parametrize('seed', [404, 505, 606])
def test_seed_preparation_keeps_training_and_eval_scope(seed, tmp_path):
    source = REPO / 'docs/experiments/phase12-vllm-v1/preparation-s404-8gpu'
    spec = read_json(source / 'spec.json')
    original = copy.deepcopy(spec)
    spec['seed'] = seed
    spec = apply_budget(spec, INDEPENDENT_PROFILE)
    manifest = read_json(source / 'manifest.json')
    for row in manifest['assets']:
        if row['path'] == 'spec.json':
            write_new_json(tmp_path / row['path'], spec)
        else:
            write_new_bytes(tmp_path / row['path'], (source / row['path']).read_bytes())
        row['sha256'] = file_hash(tmp_path / row['path'])
    write_new_json(tmp_path / 'manifest.json', manifest)
    cfg = configuration(tmp_path / 'manifest.json', tmp_path / 'run')
    assert cfg.actor_rollout_ref.rollout.seed == cfg.actor_rollout_ref.cohort_seed == seed
    assert cfg.env.seed == seed
    assert cfg.phase2.capture_updates == [1] and not capture_post(cfg.phase2)
    assert cfg.data.train_batch_size * cfg.env.rollout.n == 128
    assert cfg.trainer.total_training_steps == 5
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6
    assert spec['evaluation'] == original['evaluation']
    assert spec['inference_profile'] == original['inference_profile']


def batch():
    return DataProto(TensorDict({'responses': torch.tensor([[1, 0]]),
        'attention_mask': torch.tensor([[1, 1, 1]])}, batch_size=[1]),
        non_tensor_batch={'phase2_decision_id': np.array(['d'], dtype=object)})


def test_middle_batch_keeps_optimizer_row_identity_but_skips_disk_admission(tmp_path, monkeypatch):
    write_new_json(tmp_path / 'resource_limits.json', {'vocab_size': 4})
    calls = []
    monkeypatch.setattr('skillnet_cohort.runtime.admit_capture', lambda *a, **k: calls.append(k))
    b = batch()
    mark_batch(b, root=str(tmp_path), update=2, full_vocab=False, copies=1)
    assert not calls and b.batch['phase2_row_index'].tolist() == [0]
    assert not b.meta_info['phase2_capture']['full_vocab']
    # Must not touch tensor arguments or create old/new directories.
    capture_old_logits(None, None, None, b.meta_info['phase2_capture'])
    assert not (tmp_path / 'old_logprobs').exists()
    mark_batch(b, root=str(tmp_path), update=1, copies=1)
    assert calls == [{'rows': 1, 'copies': 1}]


def test_capture_compressed_row_roundtrips_actual_chosen_probabilities(tmp_path):
    write_new_json(tmp_path / 'resource_limits.json', {
        'full_vocab_compression': shuffled_profile(),
        'full_vocab_max_encoded_bytes_per_token': 5000, 'full_vocab_max_encoded_overhead_bytes': 10000,
        'minimum_free_bytes': 1, 'maximum_run_bytes': 1024**3, 'checkpoint_reserve_bytes': 1})
    logits = torch.tensor([[[1., 2., 0.], [2., 1., 0.]]])
    chosen = logits.log_softmax(-1).gather(-1, torch.tensor([[[1], [0]]])).squeeze(-1)
    b = {'responses': torch.tensor([[1, 0]]), 'attention_mask': torch.ones(1, 3),
         'phase2_row_index': torch.tensor([0])}
    capture_old_logits(logits, chosen, b, {'root': str(tmp_path), 'update': 1})
    saved = load(tmp_path / 'old_logprobs/u0001/row-000000.pt')
    assert torch.equal(saved['log_probs'], logits[0].log_softmax(-1))
    assert torch.equal(saved['chosen_log_probs'], chosen[0])


def test_compact_journal_and_summary_do_not_keep_middle_states(tmp_path):
    b = batch()
    b.non_tensor_batch.update(traj_uid=np.array(['traj'], dtype=object), uid=np.array(['group'], dtype=object),
                              active_masks=np.array([True]), rewards=np.array([0.]))
    attach_decisions(b, [{'selected_skill_id': 's', 'observation': 'secret state'}], run_id='test',
                     update=2, step=0, journal_root=tmp_path, full_journal=False)
    record = read_json(tmp_path / 'rollout_progress/u0002/step-0000.json')['records'][0]
    assert 'info' not in record and 'response_token_ids' not in record and record['response_tokens'] == 2
    archive_rollout_summary(tmp_path, 2, True, [[{'selected_skill_id': 's'}]], [0.], [1], ['traj'])
    summary = read_json(tmp_path / 'rollout_summaries/u0002-train.json')
    assert summary['trajectories'][0]['skill_selection_counts'] == {'s': 1}


def test_compressed_admission_counts_one_copy_and_metadata(tmp_path, monkeypatch):
    from skillnet_cohort.runtime import admit_capture
    write_new_json(tmp_path / 'resource_limits.json', {
        'vocab_size': 4, 'full_vocab_compression': shuffled_profile(),
        'full_vocab_max_encoded_bytes_per_token': 10, 'full_vocab_max_encoded_overhead_bytes': 20,
        'minimum_free_bytes': 1, 'maximum_run_bytes': 1024**3, 'checkpoint_reserve_bytes': 100})
    calls = []
    monkeypatch.setattr('skillnet_cohort.runtime.disk_gate', lambda root, required, **k: calls.append(required))
    admit_capture(tmp_path, 100, rows=3, copies=1)
    admit_capture(tmp_path, 100, rows=3, copies=2)
    assert calls == [1160, 2220]
