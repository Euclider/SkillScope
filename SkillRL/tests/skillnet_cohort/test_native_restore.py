"""Real two-rank CPU/Gloo restore tests; no GPU or ALFWorld experiment."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

from skillnet_cohort.native_restore import restore_parameters


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn(8))
        self.first = torch.nn.Linear(8, 8)
        self.second = torch.nn.Linear(8, 8)

    def forward(self, value):
        return self.second(self.first(value)) + self.bias


def _restore_worker(rank, directory, layered, mesh_enabled):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{directory}/rendezvous',
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(404)
        mesh = init_device_mesh('cpu', (2,)) if mesh_enabled else None
        options = {'device_id': torch.device('cpu'), 'use_orig_params': False,
                   'device_mesh': mesh}
        module = TinyModel()
        if layered:
            module.first = FSDP(module.first, **options)
            module.second = FSDP(module.second, **options)
        model = FSDP(module, **options)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        model(torch.ones(2, 8)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                  ShardedStateDictConfig(offload_to_cpu=True)):
            state = deepcopy(model.state_dict())
        expected = [p.detach().clone() for p in model.parameters()]
        optimizer_state = deepcopy(optimizer.state_dict())
        with torch.no_grad():
            for param in model.parameters():
                param.add_(100)
        restore_parameters(model, state, rank, 2)
        assert all(torch.equal(old, new) for old, new in zip(expected, model.parameters()))
        optimizer.load_state_dict(optimizer_state)
        actual_optimizer = optimizer.state_dict()
        assert actual_optimizer['param_groups'] == optimizer_state['param_groups']
        for index, saved in optimizer_state['state'].items():
            for key, value in saved.items():
                assert torch.equal(value, actual_optimizer['state'][index][key])
        # The restored model remains differentiable and its Adam state usable.
        model(torch.ones(2, 8)).sum().backward()
        optimizer.step()
        assert all(int(state['step']) == 2 for state in optimizer.state.values())
        with pytest.raises(ValueError, match='topology'):
            restore_parameters(model, state, rank, 4)
        if layered:
            invalid = dict(state)
            invalid['unknown.weight'] = next(iter(state.values()))
            with pytest.raises(RuntimeError, match='Unexpected key'):
                restore_parameters(model, invalid, rank, 2)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('layered', [False, True], ids=['root-flat', 'layer-wrapped'])
@pytest.mark.parametrize('mesh_enabled', [False, True], ids=['sharded-tensor', 'dtensor'])
def test_native_restore_roundtrip_cpu_two_ranks(tmp_path, layered, mesh_enabled):
    torch.multiprocessing.start_processes(_restore_worker,
        args=(str(tmp_path), layered, mesh_enabled), nprocs=2, join=True, start_method='spawn')
