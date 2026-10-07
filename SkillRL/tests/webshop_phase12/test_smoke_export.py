import pytest
import torch


def test_model_only_checkpoint_requires_explicit_export_and_cannot_resume(monkeypatch):
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    monkeypatch.setattr(torch.distributed,'get_rank',lambda:0)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:1)
    args=dict(model=torch.nn.Linear(2,2),optimizer=None,lr_scheduler=None,
              processing_class=object(),checkpoint_contents=['model'])
    with pytest.raises((ValueError,AssertionError)):FSDPCheckpointManager(**args)
    manager=FSDPCheckpointManager(**args,export_model_only=True)
    assert manager.export_model_only
    with pytest.raises(ValueError,match='resume'):manager.load_checkpoint('/unused')


def test_full_native_checkpoint_remains_default(monkeypatch):
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    monkeypatch.setattr(torch.distributed,'get_rank',lambda:0)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:1)
    manager=FSDPCheckpointManager(model=torch.nn.Linear(2,2),optimizer=None,lr_scheduler=None,processing_class=object())
    assert manager.checkpoint_contents==['model','optimizer','extra']
    assert not manager.export_model_only
