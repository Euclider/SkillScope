"""TP=1 V1 bridge using public LLM.apply_model, never executor internals.

The engine remains awake across environment steps. Any colocated native model
forward/update/checkpoint suspends it first. We synchronize only after actor
updates/restores, not at every state. No partial/stale load is accepted.
"""
from weakref import WeakSet

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

from .base import BaseShardingManager

_LOCAL_MANAGERS = WeakSet()


def suspend_vllm_rollouts():
    for manager in list(_LOCAL_MANAGERS):
        manager.suspend()


def load_complete_weights(model, weights):
    loaded = set(model.load_weights(weights))
    expected = set(dict(model.named_parameters()))
    missing = expected - loaded
    if missing:
        raise RuntimeError(f'Incomplete actor -> vLLM sync: {sorted(missing)[:12]}')
    return {'loaded_parameters': len(loaded), 'expected_parameters': len(expected)}


class FSDPVLLMV1ShardingManager(BaseShardingManager):
    def __init__(self, module, inference_engine, *, offload_param=False):
        if not isinstance(module, FSDP):
            raise ValueError('V1 profile requires native FSDP1; do not silently change training strategy')
        self.module, self.inference_engine = module, inference_engine
        self.offload_param = offload_param
        self.awake = False
        self.dirty = True
        self.weight_version = 0
        self.sync_count = 0
        self.last_sync = None
        self.inference_engine.sleep(level=1)
        _LOCAL_MANAGERS.add(self)

    def mark_dirty(self):
        self.dirty = True
        self.weight_version += 1

    def suspend(self):
        if self.awake:
            self.inference_engine.sleep(level=1)
            self.awake = False
            torch.cuda.empty_cache()

    def __enter__(self):
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
        if self.awake and not self.dirty:
            return self
        self.suspend()
        torch.cuda.empty_cache()
        # Native FSDP RNG must not depend on inference engine initialization or
        # scheduling. Sampling itself is handled by vLLM's request generators.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.inference_engine.wake_up(tags=['weights'])
            self.awake = True
            try:
                if self.dirty:
                    if self.offload_param:
                        load_fsdp_model_to_gpu(self.module)
                    # Keep only sharded source tensors + one gathered tensor,
                    # never a full additional FP32 4B model on each GPU.
                    with FSDP.state_dict_type(self.module, StateDictType.SHARDED_STATE_DICT,
                                              ShardedStateDictConfig(offload_to_cpu=False)):
                        params = self.module.state_dict()

                    def weights():
                        for name, value in params.items():
                            if not hasattr(value, 'full_tensor'):
                                raise TypeError('Expected DTensor from device-mesh FSDP state_dict')
                            yield name, value.full_tensor().detach()

                    self.last_sync = self.inference_engine.apply_model(
                        lambda model: load_complete_weights(model, weights()))
                    del params
                    if self.offload_param:
                        offload_fsdp_model_to_cpu(self.module)
                    if self.inference_engine.reset_prefix_cache() is False:
                        raise RuntimeError('Could not invalidate inference caches after weight update')
                    self.dirty = False
                    self.sync_count += 1
                torch.cuda.empty_cache()
                self.inference_engine.wake_up(tags=['kv_cache'])
            except BaseException:
                self.dirty = True
                self.suspend()
                raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.suspend()
        self.module.train()
        return False
