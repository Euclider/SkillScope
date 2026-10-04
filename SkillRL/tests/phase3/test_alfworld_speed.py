import copy

import pytest
import torch
from omegaconf import OmegaConf

from alfworld_rl_speed.runtime import apply_options, choose_options, probe_indices


def test_selects_a_full_global_minibatch_with_lengths_and_advantages():
    masks = torch.arange(512)[None, :] < (torch.arange(256) % 31 + 1)[:, None]
    advantages = (torch.arange(256) % 3 - 1)[:, None].expand(-1, 512).float()
    indices = probe_indices(masks, advantages, 128)
    assert len(indices) == len(set(indices)) == 128
    assert {int(torch.sign(advantages[i, 0])) for i in indices} == {-1, 0, 1}
    assert min(int(masks[i].sum()) for i in indices) == 1
    assert max(int(masks[i].sum()) for i in indices) == 31
    assert indices == probe_indices(masks, advantages, 128)


def result(**changes):
    base = dict(passed=True, world_size=8, rows_per_rank=16,
                max_logprob_error=0., max_entropy_error=0.,
                grad_relative_l2=.01, grad_cosine=.9999,
                optimizer_boundaries=1, peak_gib=25., total_gib=31.8,
                seconds=20., finite=True)
    base.update(changes)
    return {'ranks': [copy.deepcopy(base) for _ in range(8)]}


def test_fail_closed_and_per_option_acceptance():
    records = {'suffix': result(), 'reference_gpu_root': result(),
               'no_sync': result(peak_gib=31.)}
    assert choose_options(records) == {'response_logits_only': True,
                                      'reference_gpu_root': True, 'accumulate_no_sync': False}
    records['suffix']['ranks'][7]['grad_relative_l2'] = .1
    with pytest.raises(ValueError, match='suffix'):
        choose_options(records)


def test_partial_or_nonfinite_probe_is_not_an_acceptance():
    with pytest.raises(ValueError):
        choose_options({'suffix': {'ranks': result()['ranks'][:7]}})
    with pytest.raises(ValueError):
        choose_options({'suffix': result(grad_relative_l2=float('nan'))})


def test_options_change_only_performance_fields():
    original = dict(actor_rollout_ref=dict(actor=dict(ppo_mini_batch_size=128,
        ppo_micro_batch_size_per_gpu=1, loss_agg_mode='token-mean', ppo_epochs=1),
        ref=dict(log_prob_micro_batch_size_per_gpu=1, fsdp_config=dict(param_offload=True)),
        rollout=dict(name='vllm_v1', response_length=512)), data=dict(train_batch_size=16),
        env=dict(rollout=dict(n=8)))
    cfg = OmegaConf.create(original)
    apply_options(cfg, dict(response_logits_only=True, reference_gpu_root=False, accumulate_no_sync=False))
    assert cfg.actor_rollout_ref.ref.fsdp_config.param_offload
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 128
    assert cfg.actor_rollout_ref.actor.loss_agg_mode == 'token-mean'
    assert cfg.actor_rollout_ref.rollout.name == 'vllm_v1'
    assert not cfg.actor_rollout_ref.actor.trim_common_padding
