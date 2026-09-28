import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from skillnet_cohort.common import file_hash, read_json, write_new_bytes, write_new_json
from skillnet_cohort.rollout_recovery import reconstruct, records, load_for_training, save_pre_forward, verify_old_reuse


class Tokenizer:
    pad_token_id = 0
    def apply_chat_template(self, chat, **kwargs):
        return chat[0]['content']
    def __call__(self, prompt, **kwargs):
        return {'input_ids': torch.tensor([[3, 4]]), 'attention_mask': torch.ones((1, 2), dtype=torch.long)}


def fixture(tmp_path):
    config = {'env': {'max_steps': 1}, 'data': {'max_prompt_length': 4, 'apply_chat_template_kwargs': {}},
        'actor_rollout_ref': {'rollout': {'response_length': 3, 'log_prob_micro_batch_size_per_gpu': 1},
            'ref': {'log_prob_micro_batch_size_per_gpu': 1},
            'actor': {'use_kl_loss': True, 'ppo_micro_batch_size_per_gpu': 1}},
        'trainer': {'n_gpus_per_node': 8, 'nnodes': 1}, 'algorithm': {'use_kl_in_reward': False},
        'skillnet_cohort': {'seed': 404}}
    write_new_json(tmp_path/'segments/u0000-u0005.json', config)
    write_new_json(tmp_path/'stopped.json', {'stage': 'train-u0000-u0005'})
    rows = [{'trajectory_id': str(i), 'group_id': str(i//8), 'decision_id': f'd{i}', 'environment_step': 0,
        'global_update': 1, 'response_token_ids': [2, 1], 'response_tokens': 2, 'reward': float(i%2),
        'info': {'extra.gamefile': f'game{i//8}', 'won': bool(i%2), 'is_action_valid': True,
                 'prompt_text': f'prompt{i}', 'observation': 'obs'}} for i in range(128)]
    write_new_json(tmp_path/'rollout_progress/u0001/step-0000.json', {'step': 0, 'update': 1, 'records': rows})
    write_new_json(tmp_path/'rollout_summaries/u0001-train.json', {'trajectories': [
        {'trajectory_id': str(i), 'environment_steps': 1, 'episode_return': float(i%2), 'game_id': f'game{i//8}'}
        for i in range(128)]})


def test_reconstruct_preserves_original_samples_groups_and_mask(tmp_path):
    fixture(tmp_path)
    np.random.seed(80)
    before = np.random.get_state()
    batch, audit = reconstruct(tmp_path, Tokenizer())
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert len(batch) == audit['raw_rows'] == 128
    assert batch.batch['responses'].shape == (128, 3)
    assert (batch.batch['responses'] == torch.tensor([2, 1, 0])).all()
    assert (batch.batch['attention_mask'][:, -3:] == torch.tensor([1, 1, 0])).all()
    assert (batch.batch['position_ids'][0] == torch.tensor([0, 0, 0, 1, 2, 3, 4])).all()
    assert 'rollout_log_probs' not in batch.batch
    assert set(batch.non_tensor_batch['phase2_decision_id']) == {f'd{i}' for i in range(128)}
    for i, meta in enumerate(batch.non_tensor_batch['phase2_metadata']):
        value = json.loads(meta)
        assert value['trajectory_id'] == batch.non_tensor_batch['traj_uid'][i]
        assert value['group_id'] == batch.non_tensor_batch['uid'][i]
    save_pre_forward(batch, tmp_path, 1)
    saved = torch.load(tmp_path/'pre_forward_batches/u0001.pt', weights_only=False)
    assert torch.equal(saved['tensors']['input_ids'], batch.batch['input_ids'])


def test_recovery_rejects_any_optimizer_boundary_or_incomplete_rollout(tmp_path):
    fixture(tmp_path)
    (tmp_path/'optimizer_steps').mkdir()
    with pytest.raises(ValueError, match='pre-optimizer'):
        records(tmp_path)


def test_recovery_old_row_reuse_checks_bits_and_never_overwrites(tmp_path):
    from phase2.capture import capture_old_logits
    from skillnet_cohort.lossless_tensor import load
    logits = torch.tensor([[[1., 2., 0.], [2., 1., 0.]]])
    chosen = logits.log_softmax(-1).gather(-1, torch.tensor([[[1], [0]]])).squeeze(-1)
    batch = {'responses': torch.tensor([[1, 0]]), 'attention_mask': torch.ones(1, 3), 'phase2_row_index': torch.tensor([0])}
    capture = {'root': str(tmp_path), 'update': 1}
    capture_old_logits(logits, chosen, batch, capture)
    path = tmp_path/'old_logprobs/u0001/row-000000.pt'
    original = file_hash(path)
    manifest = tmp_path/'recovery-manifest.json'
    write_new_json(manifest, {'matched_original_old_rows': [{'row': 0, 'sha256': original}]})
    capture.update(reuse_verified_old={'manifest': str(manifest), 'sha256': file_hash(manifest)},
                   progress_root=str(tmp_path/'recovery'))
    capture_old_logits(logits, chosen, batch, capture)
    assert file_hash(path) == original
    assert (tmp_path/'recovery/reused-old-rank0.jsonl').is_file()
    changed = load(path)
    changed['log_probs'][0, 0] += 1
    with pytest.raises(ValueError, match='differs'):
        verify_old_reuse(path, changed, capture)
    assert file_hash(path) == original


def test_reconstructed_batch_hash_and_game_iterator_are_checked(tmp_path):
    manifest = tmp_path/'manifest.json'
    write_new_bytes(tmp_path/'rollout_batch.pt', b'not a batch')
    write_new_json(manifest, {'batch_sha256': file_hash(tmp_path/'rollout_batch.pt'), 'initial_game_ids': ['expected']})
    envs = SimpleNamespace(envs=SimpleNamespace(reset=lambda: (None, None, [{'extra.gamefile': 'wrong'}])))
    with pytest.raises(ValueError, match='iterator'):
        load_for_training({'manifest': str(manifest), 'sha256': file_hash(manifest)}, envs)
