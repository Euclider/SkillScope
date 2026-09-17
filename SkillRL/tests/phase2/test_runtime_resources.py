from phase2.watch_training_resources import parse_snapshot


def test_processes_mapped_to_physical_devices_not_local_ray_rank():
    result=parse_snapshot("3, GPU-a, 30000, 50000, 80000, 92\n6, GPU-b, 13, 79987, 80000, 0\n",
                          "123, GPU-a, 29000\n456, GPU-b, N/A\n")
    assert [row["gpu"] for row in result["compute_processes"]]==[3,6]
    assert result["compute_processes"][1]["used_mib"] is None
    assert result["gpus"][0]["free_mib"]==50000


def test_idle_snapshot_can_have_no_compute_processes():
    result=parse_snapshot("3, GPU-a, 13, 79987, 80000, 0\n", "")
    assert result["compute_processes"]==[]
