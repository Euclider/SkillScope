from types import SimpleNamespace

import pytest
import torch


class IndexedCausalToy(torch.nn.Module):
    """Real causal computation; selects hidden positions before its output head."""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 8)
        self.head = torch.nn.Linear(8, 17)

    def forward(self, input_ids, attention_mask, position_ids, use_cache, logits_to_keep):
        hidden = self.embedding(input_ids).cumsum(1)
        idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return SimpleNamespace(logits=self.head(hidden[:, idx]))


@pytest.mark.parametrize('mask', [[True, True, False, False], [True, False, True, False]])
def test_indexed_head_matches_dense_valid_values_and_parameter_gradients(mask):
    from verl.workers.actor.active_logits import response_position_indices, scatter_response_values
    torch.manual_seed(12)
    model = IndexedCausalToy().double()
    ids = torch.tensor([[0, 0, 3, 4, 5, 6, 7, 8]])
    attention = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0]])
    positions = torch.arange(8)[None]
    loss_mask = torch.tensor([mask])
    inputs = dict(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False)
    dense = model(**inputs, logits_to_keep=5).logits[:, :-1]
    dense_lp = dense.log_softmax(-1)
    dense_loss = -dense_lp[loss_mask].sum()
    dense_loss.backward()
    reference = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    idx, active = response_position_indices(ids.shape[-1], loss_mask)
    sparse = model(**inputs, logits_to_keep=idx).logits
    sparse_lp = sparse.log_softmax(-1)
    restored = scatter_response_values(sparse_lp.sum(-1), active, 4)
    assert torch.allclose(sparse_lp[0], dense_lp[loss_mask], atol=1e-12)
    assert torch.equal(restored[~loss_mask], torch.zeros_like(restored[~loss_mask]))
    (-restored.sum()).backward()
    for expected, parameter in zip(reference, model.parameters()):
        assert torch.allclose(expected, parameter.grad, atol=1e-12)
    # An inactive response still influences a later active prediction.
    if not mask[1] and mask[2]:
        changed = ids.clone(); changed[0, 5] = 9
        assert not torch.equal(model(**{**inputs, 'input_ids': changed}, logits_to_keep=idx).logits, sparse)


def test_empty_mask_preserves_a_differentiable_zero_and_rejects_batching():
    from verl.workers.actor.active_logits import response_position_indices, scatter_response_values
    idx, active = response_position_indices(8, torch.zeros(1, 4, dtype=torch.bool))
    value = torch.tensor([[2.]], requires_grad=True)
    out = scatter_response_values(value, active, 4)
    assert out.shape == (1, 4) and not out.any()
    out.sum().backward()
    assert value.grad.item() == 0
    with pytest.raises(ValueError):
        response_position_indices(8, torch.ones(2, 4, dtype=torch.bool))


def test_native_actor_active_head_preserves_valid_entropy_logprob_and_gradients():
    from omegaconf import OmegaConf
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    torch.manual_seed(12)
    model = IndexedCausalToy()
    ids = torch.tensor([[0, 0, 3, 4, 5, 6, 7, 8]])
    attention = torch.tensor([[0, 0, 1, 1, 1, 0, 1, 0]])
    data = {'input_ids': ids, 'attention_mask': attention, 'position_ids': torch.arange(8)[None],
            'responses': ids[:, -4:]}
    config = OmegaConf.create({'use_torch_compile': False, 'ulysses_sequence_parallel_size': 1,
                              'response_logits_only': True})
    actor = DataParallelPPOActor(config, model);actor.device_name = 'cpu'
    mask = attention[:, -4:].bool()
    entropy, lp = actor._forward_micro_batch(data, 1., True)
    ((lp + .001*entropy)*mask).sum().backward()
    expected = [p.grad.clone() for p in model.parameters()]
    model.zero_grad();actor.active_response_logits_only = True
    ent2, lp2 = actor._forward_micro_batch(data, 1., True)
    ((lp2 + .001*ent2)*mask).sum().backward()
    torch.testing.assert_close(lp[mask], lp2[mask])
    torch.testing.assert_close(entropy[mask], ent2[mask])
    for ref, parameter in zip(expected, model.parameters()):
        torch.testing.assert_close(ref, parameter.grad, atol=.01, rtol=.01)
