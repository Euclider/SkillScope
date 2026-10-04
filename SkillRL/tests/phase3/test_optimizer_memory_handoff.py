"""Successful actor RPC must hand unused pages to the router process."""
from types import SimpleNamespace
import pytest
import torch


def test_handoff_runs_after_success_and_preserves_output(monkeypatch):
    from verl.utils import fsdp_utils
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.AdamW([parameter])
    parameter.square().sum().backward()
    optimizer.step()
    expected = {k: v.clone() for k, v in optimizer.state[parameter].items()}
    calls = []
    def synchronize():
        assert all(v.device.type == 'cpu' for v in optimizer.state[parameter].values())
        calls.append('synchronize')
    monkeypatch.setattr(fsdp_utils, 'get_torch_device', lambda: SimpleNamespace(
        is_available=lambda: True, synchronize=synchronize,
        empty_cache=lambda: calls.append('release')))
    output = object()
    @fsdp_utils.release_cuda_cache_after
    def update():
        calls.append('update')
        return output
    assert update() is output
    assert calls == ['update', 'synchronize', 'release']
    for key, value in expected.items():
        assert torch.equal(optimizer.state[parameter][key], value)


def test_handoff_preserves_original_error(monkeypatch):
    from verl.utils import fsdp_utils
    monkeypatch.setattr(fsdp_utils, 'get_torch_device', lambda: (_ for _ in ()).throw(
        AssertionError('Do not mask the original exception')))
    @fsdp_utils.release_cuda_cache_after
    def update():
        raise ValueError('original')
    with pytest.raises(ValueError, match='original'):
        update()
