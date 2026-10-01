"""Preserve response alignment and the old microbatch-one objective."""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty


class CausalToy(torch.nn.Module):
    # Actual differentiable causal model; masked padding contributes nothing.
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(31, 8)
        self.head = torch.nn.Linear(8, 31)
        self.observed = None

    def forward(self, input_ids, attention_mask, position_ids, use_cache=False, logits_to_keep=0):
        self.observed = (input_ids.shape, position_ids.clone(), logits_to_keep)
        x = (self.embed(input_ids) * attention_mask[..., None]).cumsum(1)
        x = x + position_ids[..., None] / 100
        return SimpleNamespace(logits=self.head(x[:, -logits_to_keep:] if logits_to_keep else x))


def actor(model, **options):
    config = OmegaConf.create(dict(use_torch_compile=False, ulysses_sequence_parallel_size=1, **options))
    obj = DataParallelPPOActor(config, model)
    obj.device_name = 'cpu'
    return obj


def batch():
    ids = torch.tensor([[0, 0, 0, 2, 3, 4, 5, 0], [0, 0, 1, 2, 3, 6, 7, 8]])
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1]])
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    return dict(input_ids=ids, attention_mask=mask, position_ids=pos, responses=ids[:, -3:])


@pytest.mark.parametrize('trim', [False, True])
def test_response_logits_keep_preserves_scores_entropy_and_gradients(trim):
    torch.manual_seed(5)
    model = CausalToy()
    data = batch()
    mask = data['attention_mask'][:, -3:]
    base = actor(model)
    ent0, lp0 = base._forward_micro_batch(data, 1., True)
    loss0 = ((lp0 + .01 * ent0) * mask).sum()
    loss0.backward()
    grads0 = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    fast = actor(model, response_logits_only=True, trim_common_padding=trim)
    ent1, lp1 = fast._forward_micro_batch(data, 1., True)
    ((lp1 + .01 * ent1) * mask).sum().backward()
    assert model.observed[2] == 4  # Three labels need four final logits.
    assert model.observed[0][1] == (6 if trim else 8)
    assert torch.equal(model.observed[1], data['position_ids'][:, 2:] if trim else data['position_ids'])
    torch.testing.assert_close(lp0, lp1)
    torch.testing.assert_close(ent0, ent1)
    for p, old in zip(model.parameters(), grads0):
        torch.testing.assert_close(p.grad, old, atol=.04, rtol=.01)
    assert data['input_ids'].shape == (2, 8)  # Caller/captured training batch stays untouched.


def test_sequence_mean_matches_accumulated_microbatch_one_ppo_entropy_kl():
    torch.manual_seed(17)
    old = torch.randn(4, 5)
    ref = old + .1
    adv = torch.randn(4, 5)
    mask = torch.arange(5)[None, :] < torch.tensor([1, 2, 4, 5])[:, None]
    lp = (old + .2 * torch.randn(4, 5)).requires_grad_()
    entropy = torch.randn(4, 5, requires_grad=True)
    def objective(logp, ent, prev, reference, advantages, valid, mode):
        pg, *_ = compute_policy_loss(prev, logp, advantages, valid, .2, loss_agg_mode=mode)
        return pg - .001 * agg_loss(ent, valid, mode) + .01 * agg_loss(kl_penalty(logp, reference, 'low_var_kl'), valid, mode)
    base = sum(objective(lp[i:i+1], entropy[i:i+1], old[i:i+1], ref[i:i+1], adv[i:i+1], mask[i:i+1], 'token-mean') for i in range(4)) / 4
    grad0 = torch.autograd.grad(base, (lp, entropy), retain_graph=True)
    merged = objective(lp, entropy, old, ref, adv, mask, 'seq-mean-token-mean')
    torch.testing.assert_close(base, merged)
    for a, b in zip(grad0, torch.autograd.grad(merged, (lp, entropy))):
        torch.testing.assert_close(a, b)


def test_trim_rejects_unsupported_remove_padding_combination():
    with pytest.raises(ValueError, match='padded'):
        actor(CausalToy(), trim_common_padding=True, use_remove_padding=True)


def test_sync_context_reduces_only_nonfinal_accumulation():
    from contextlib import contextmanager
    from verl.workers.actor.padded_forward import gradient_sync_context
    class Model:
        syncing = True
        @contextmanager
        def no_sync(self):
            self.syncing = False
            try:
                yield
            finally:
                self.syncing = True
    model = Model()
    for index in range(4):
        with gradient_sync_context(model, enabled=True, last=index == 3):
            assert model.syncing == (index == 3)
        assert model.syncing
    with gradient_sync_context(model, enabled=False, last=False):
        assert model.syncing


def test_forward_optimization_keeps_eight_optimizer_boundaries(monkeypatch):
    from verl import DataProto
    from verl.workers.actor import dp_actor
    monkeypatch.setattr(dp_actor, 'get_torch_device', lambda: SimpleNamespace(current_device=lambda:'cpu'))
    model=CausalToy()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-6)
    obj=actor(model,response_logits_only=True,ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1,ppo_epochs=1,use_dynamic_bsz=False,
        clip_ratio=.2,clip_ratio_low=.2,clip_ratio_high=.2,entropy_coeff=.001,
        loss_agg_mode='token-mean',use_kl_loss=True,kl_loss_type='low_var_kl',
        kl_loss_coef=.01,policy_loss={'loss_mode':'vanilla'},grad_clip=1.)
    obj.actor_optimizer=opt
    data={k:v.repeat(8,1) for k,v in batch().items()}
    with torch.no_grad(): _,lp=obj._forward_micro_batch(data,1.)
    data.update(old_log_probs=lp,ref_log_prob=lp.clone(),advantages=torch.ones_like(lp))
    metrics=obj.update_policy(DataProto.from_dict(tensors=data,meta_info={'temperature':1.}))
    assert metrics['actor/optimizer_steps']==8
    assert len(metrics['actor/grad_norm'])==8
    assert {int(state['step'].item()) for state in opt.state.values()}=={8}
