import pytest
import torch

from skillnet_cohort.capture_preflight import synthetic_batch


def test_probe_has_exact128_row_layout_without_storing512_tokens_per_row():
    rows = []
    for rank in range(8):
        batch = synthetic_batch(rank, 8, 'cpu')
        assert batch.batch['input_ids'].shape == (16, 4608)
        assert batch.batch['responses'].shape == (16, 512)
        assert batch.batch['attention_mask'][:, :4096].all()
        assert torch.equal(batch.batch['attention_mask'][:, -512:].sum(1), torch.full((16,), 16))
        rows += batch.batch['phase2_row_index'].tolist()
    assert rows == list(range(128))


def test_probe_does_not_silently_change_world_size():
    with pytest.raises(ValueError, match='eight'):
        synthetic_batch(0, 4, 'cpu')
