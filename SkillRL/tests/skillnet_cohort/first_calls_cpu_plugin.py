"""Test-only GPU telemetry isolation; never imported by training/evaluation.

One legacy CPU optimizer test mocks its computation device but leaves its GPU
memory logger enabled. With CUDA hidden, torch.cpu has no memory_allocated API.
Disable only that observational logger in that one test. The real native CPU
optimizer, losses, gradients and all assertions still execute unchanged.
"""
import pytest


@pytest.fixture(autouse=True)
def isolate_legacy_cpu_memory_logger(request, monkeypatch):
    if request.node.nodeid == 'tests/skillnet_cohort/test_reward_recovery.py::test_native_cpu_optimizer_multiple_and_partial_minibatches':
        from verl.utils.debug import performance
        monkeypatch.setattr(performance, '_get_current_mem_info', lambda *a, **k: ('CPU-test-telemetry-unavailable',)*4)
