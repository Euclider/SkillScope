import pytest

from phase1.preflight import check_cuda_devices


def test_phase1_default_still_requires_two_gpus():
    assert check_cuda_devices(True,1)["status"]=="fail"
    assert check_cuda_devices(True,2)["status"]=="pass"


@pytest.mark.parametrize("world",range(1,9))
def test_phase2_can_request_actual_allocation(world):
    assert check_cuda_devices(True,world,world)["status"]=="pass"
    assert check_cuda_devices(True,world-1,world)["status"]=="fail"
    assert check_cuda_devices(False,world,world)["status"]=="fail"


def test_invalid_gpu_minimum_not_accepted():
    for minimum in (0,-1,9):
        with pytest.raises(ValueError):check_cuda_devices(True,8,minimum)
