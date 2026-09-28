"""CPU-only native interfaces, including the last partial PPO minibatch."""
from copy import deepcopy
import io
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import action_validity_array, apply_invalid_action_penalty, compute_advantage
from verl.trainer.ppo.metric_utils import compute_data_metrics
from skillnet_cohort.common import file_hash, read_json, write_new_bytes, write_new_json


def batch_fixture(values):
    n = len(values)
    mask = torch.tensor([[1, 1, 1, 1, 0]]*n)
    tensors = {'prompts': torch.ones(n, 2, dtype=torch.long), 'responses': torch.ones(n, 3, dtype=torch.long),
        'input_ids': torch.ones(n, 5, dtype=torch.long), 'attention_mask': mask,
        'position_ids': torch.arange(5).expand(n, -1), 'response_mask': mask[:, -3:],
        'token_level_scores': torch.zeros(n, 3), 'step_rewards': torch.ones(n, 1)}
    non = {'is_action_valid': np.asarray(values, dtype=object),
        'uid': np.asarray(['g']*n, dtype=object), 'traj_uid': np.asarray([str(i) for i in range(n)], dtype=object),
        'episode_rewards': np.asarray([float(i % 2) for i in range(n)], dtype=object),
        'episode_lengths': np.asarray([1]*n, dtype=object), 'tool_callings': np.asarray([0]*n, dtype=object),
        'success_rate': np.asarray([.5]*n, dtype=object)}
    return DataProto.from_dict(tensors, non, {'temperature': 1.})


@pytest.mark.parametrize('values', [[True, False], [np.bool_(True), np.bool_(False)],
    [1, 0], [1., 0.], [[True], [False]]])
def test_validity_scalar_types_preserve_exact_penalty(values):
    batch = batch_fixture(values)
    result, metrics = apply_invalid_action_penalty(batch, .1)
    expected = torch.tensor([[0., 0., 0.], [0., -.1, 0.]])
    assert torch.equal(result.batch['token_level_scores'], expected)
    assert torch.equal(result.batch['step_rewards'], torch.tensor([[1.], [.9]]))
    assert metrics == {'episode/valid_action_ratio': .5}


@pytest.mark.parametrize('values', [[None], ['True'], [2], [-1], [float('nan')], [float('inf')]])
def test_rejects_invalid_metadata_before_forwards(values):
    with pytest.raises(ValueError, match='validity'):
        action_validity_array(values, 1)


def test_empty_response_rejected_without_mutation():
    batch = batch_fixture([True, False])
    batch.batch['attention_mask'][1, -3:] = 0
    before = batch.batch['token_level_scores'].clone()
    with pytest.raises(ValueError, match='nonempty'):
        apply_invalid_action_penalty(batch, .1)
    assert torch.equal(before, batch.batch['token_level_scores'])


def test_grpo_and_metrics_accept_object_python_scalars():
    batch, _ = apply_invalid_action_penalty(batch_fixture([True, False]*4), .1)
    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
    compute_advantage(batch, 'grpo')
    assert torch.isfinite(batch.batch['advantages']).all()
    metrics = compute_data_metrics(batch, use_critic=False)
    assert all(isinstance(v, float) and np.isfinite(v) for v in metrics.values())


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_complete_old_cache_native_dtype_witness(tmp_path, dtype):
    from skillnet_cohort.recovery_forward import load_old_with_witness
    batch = batch_fixture([True, False]*8)
    batch.meta_info['phase2_capture'] = {'full_vocab': True, 'reuse_verified_old': {'dummy': True}}
    values = torch.full((16, 3), -0.25)
    cache = tmp_path/'chosen.pt'
    stream = io.BytesIO()
    torch.save({'trainer_chosen_log_probs_fp32': values, 'mask': batch.batch['response_mask'].bool()}, stream)
    write_new_bytes(cache, stream.getvalue())
    manifest = tmp_path/'manifest.json'
    write_new_json(manifest, {'old_probability_cache': str(cache), 'old_probability_cache_sha256': file_hash(cache),
                             'complete_old_rows': 16})
    binding = {'manifest': str(manifest), 'sha256': file_hash(manifest), 'attempt_dir': str(tmp_path/'attempt')}
    def forward(witness):
        assert len(witness) == 8 and witness.meta_info['phase2_capture']['full_vocab'] is False
        assert 'reuse_verified_old' not in witness.meta_info['phase2_capture']
        return DataProto.from_dict({'old_log_probs': torch.full((8, 3), -.25, dtype=dtype)})
    result = load_old_with_witness(batch, binding, SimpleNamespace(world_size=8, compute_log_prob=forward))
    assert result.batch['old_log_probs'].dtype == dtype
    assert torch.equal(result.batch['old_log_probs'].float(), values)
    assert batch.meta_info['phase2_capture']['full_vocab'] is True
    assert read_json(tmp_path/'attempt/old-reuse-witness.json')['witness_bitwise_match'] is True


def test_native_cpu_optimizer_multiple_and_partial_minibatches(tmp_path, monkeypatch):
    from verl.workers.actor import dp_actor
    module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(.1)
    config = OmegaConf.create({'use_remove_padding': False, 'use_fused_kernels': False,
        'ulysses_sequence_parallel_size': 1, 'use_torch_compile': False,
        'use_dynamic_bsz': False, 'use_kl_loss': True, 'kl_loss_type': 'low_var_kl', 'kl_loss_coef': .01,
        'ppo_mini_batch_size': 16, 'ppo_micro_batch_size_per_gpu': 1, 'ppo_epochs': 1,
        'clip_ratio': .2, 'clip_ratio_low': None, 'clip_ratio_high': None, 'clip_ratio_c': 3.,
        'entropy_coeff': .001, 'loss_agg_mode': 'token-mean', 'policy_loss': {'loss_mode': 'vanilla'}, 'grad_clip': 1.})
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-6)
    actor = dp_actor.DataParallelPPOActor(config, module, optimizer)
    monkeypatch.setattr(dp_actor, 'get_torch_device', lambda: SimpleNamespace(current_device=lambda: 'cpu'))
    def forward(micro_batch, temperature, calculate_entropy):
        scores = module.weight.reshape(1, 1).expand_as(micro_batch['responses'])
        return scores.square()+.5 if calculate_entropy else None, scores-.35
    monkeypatch.setattr(actor, '_forward_micro_batch', forward)
    batch = batch_fixture([True]*33)
    batch.meta_info['phase2_capture'] = {'root': str(tmp_path), 'update': 1}
    batch.batch['phase2_row_index'] = torch.arange(33)
    batch.batch['old_log_probs'] = torch.full((33, 3), -.25)
    batch.batch['ref_log_prob'] = torch.full((33, 3), -.25)
    batch.batch['advantages'] = torch.ones(33, 3)
    before = module.weight.detach().clone()
    result = actor.update_policy(batch)
    assert len(result['actor/grad_norm']) == 3
    assert all(torch.isfinite(torch.tensor(v)).all() for v in result.values())
    assert not torch.equal(module.weight, before)
    import json
    rows = [json.loads(x) for x in (tmp_path/'optimizer_steps/u0001-rank0.jsonl').read_text().splitlines()]
    assert [r['adam_step_after'] for r in rows] == [1, 2, 3]
    assert [len(r['batch_row_indices']) for r in rows] == [16, 16, 1]
