import json

import pytest

from scripts.watch_sra_logicbench_gpus import (
    eligible_gpus,
    launch_blockers,
    parse_gpu_snapshot,
    stable_gpu_choice,
    watch,
)


def test_process_blocks_low_utilization_gpu():
    gpu_lines = "\n".join((
        "0, GPU-a, 16, 81920, 0",
        "1, GPU-b, 16, 81920, 0",
        "2, GPU-c, 2048, 81920, 0",
        "3, GPU-d, 16, 81920, 40",
    ))
    snapshot = parse_gpu_snapshot(gpu_lines, "GPU-a, 12345\n")
    assert eligible_gpus(snapshot, max_used_mib=512, max_utilization=10) == [1]


def test_stable_choice_requires_same_two_cards_across_polls():
    history = [[1, 2], [2, 3], [1, 2, 3]]
    assert stable_gpu_choice(history, count=2, required_polls=3) is None
    history[0] = [2, 3]
    history[-1] = [2, 3]
    assert stable_gpu_choice(history, count=2, required_polls=3) == [2, 3]


def test_launch_blockers_refuse_unready_protocol_and_low_disk(tmp_path):
    run_root = tmp_path / "run"
    receipt = tmp_path / "preflight.json"
    manifest = {
        "schema_version": "skillscope.logicbench_gpu_queue.v1",
        "runnable_experiment": False,
        "command": ["/usr/bin/true"],
        "run_root": str(run_root),
        "preflight_receipt": str(receipt),
        "minimum_free_bytes": 200 * 2**30,
    }
    assert "experiment_not_ready" in launch_blockers(manifest, 300 * 2**30)
    manifest["runnable_experiment"] = True
    assert "preflight_receipt_missing" in launch_blockers(manifest, 300 * 2**30)
    receipt.write_text(json.dumps({"status": "PASS", "two_gpu_training": True,
                                   "logicbench_phase12_pipeline": True}))
    assert "insufficient_disk" in launch_blockers(manifest, 59 * 2**30)
    assert launch_blockers(manifest, 300 * 2**30) == []
    manifest["required_gpu_count"] = 4
    assert "preflight_receipt_invalid" in launch_blockers(manifest, 300 * 2**30)
    receipt.write_text(json.dumps({"status": "PASS", "training_gpu_count": 4,
                                   "logicbench_phase12_pipeline": True}))
    assert launch_blockers(manifest, 300 * 2**30) == []
    run_root.mkdir()
    (run_root / "launch.json").write_text("{}")
    assert "run_root_not_empty" in launch_blockers(manifest, 300 * 2**30)


def test_malformed_gpu_snapshot_is_rejected():
    with pytest.raises(ValueError, match="GPU snapshot"):
        parse_gpu_snapshot("0, GPU-a, bad, 81920, 0", "")


def test_watch_launches_once_after_stable_two_gpu_preflight(tmp_path, monkeypatch):
    import scripts.watch_sra_logicbench_gpus as watcher

    receipt = tmp_path / "preflight.json"
    receipt.write_text(json.dumps({"status": "PASS", "two_gpu_training": True,
                                   "logicbench_phase12_pipeline": True}))
    run_root = tmp_path / "run"
    manifest = tmp_path / "queue.json"
    manifest.write_text(json.dumps({
        "schema_version": "skillscope.logicbench_gpu_queue.v1",
        "runnable_experiment": True,
        "command": ["/usr/bin/true"], "run_root": str(run_root),
        "preflight_receipt": str(receipt), "minimum_free_bytes": 1,
    }))
    gpus = [{"index": index, "used_mib": 16, "total_mib": 81920,
             "utilization": 0, "compute_pids": []} for index in (1, 2)]
    monkeypatch.setattr(watcher, "gpu_snapshot", lambda: gpus)
    monkeypatch.setattr(watcher.time, "sleep", lambda _: None)
    result = watch(manifest, tmp_path / "state", stable_polls=2,
                   poll_seconds=0.01)
    assert result["state"] == "finished"
    assert json.loads((run_root / "launch.json").read_text())["gpu_ids"] == [1, 2]
    assert len((tmp_path / "state/history.jsonl").read_text().splitlines()) == 2
