import json
from types import SimpleNamespace

import pytest

from phase2 import evaluation_resources as module


def gpu(used=13, util=0):
    return {"gpu": 2, "used_mib": used, "total_mib": 81920, "utilization": util}


def test_three_useful_measurement_workers_fit_with_headroom():
    assert module.extra_slots(gpu(), set(), set(), 23552, 8192, 3) == 3
    assert module.extra_slots(gpu(200), {10}, set(), 23552, 8192, 3) == 2
    assert module.extra_slots(gpu(40000), {10, 11, 12}, {10, 11, 12}, 23552, 8192, 3) == 0


def test_foreign_occupied_gpu_never_gets_additional_workers():
    assert module.extra_slots(gpu(20000), set(), {91}, 14336, 8192, 4) == 0
    assert module.extra_slots(gpu(20000), {10}, {10, 91}, 14336, 8192, 4) == 0
    assert module.extra_slots(gpu(10000), set(), set(), 14336, 8192, 4) == 0
    assert module.extra_slots(gpu(13, 100), set(), set(), 14336, 8192, 4) == 0


def test_large_actual_use_overrides_nominal_worker_budget():
    assert module.extra_slots(gpu(68000), {10}, {10}, 23552, 8192, 3) == 0
    assert module.extra_slots(gpu(16000), {10}, {10}, 14336, 8192, 4) == 3


def test_only_oom_is_retryable_not_alignment_failure():
    assert module.retryable("torch.OutOfMemoryError: CUDA out of memory")
    assert not module.retryable("ValueError: token alignment failed")
    assert not module.retryable("token alignment failed; out of memory")


def test_eval_disk_admission_does_not_require_another_training_checkpoint(tmp_path, monkeypatch):
    policy = {"disk_reserve_gib": 200, "remaining_write_allowance_gib": 20}
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: SimpleNamespace(free=244*2**30))
    module.check_disk(tmp_path, policy)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: SimpleNamespace(free=219*2**30))
    with pytest.raises(RuntimeError): module.check_disk(tmp_path, policy)


def test_resource_amendment_is_bound_to_protocol(tmp_path):
    assert module.load_policy(tmp_path) is None
    (tmp_path/"protocol.json").write_text("{}")
    (tmp_path/module.AMENDMENT).write_text(json.dumps({"root": str(tmp_path), "protocol_sha256": "changed"}))
    with pytest.raises(ValueError, match="frozen protocol"): module.load_policy(tmp_path)


def test_authorized_disk_waiver_allows_evaluation_below_old_budget(tmp_path, monkeypatch):
    policy = {"disk_budget_waived": True, "minimum_write_safety_mib": 256}
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: SimpleNamespace(free=2*2**30))
    module.check_disk(tmp_path, policy)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: SimpleNamespace(free=128*2**20))
    with pytest.raises(RuntimeError, match="actual write space"): module.check_disk(tmp_path, policy)


def test_waiver_retains_original_policy_and_requires_bound_authority(tmp_path):
    from phase1.archive import sha256_file
    (tmp_path/"protocol.json").write_text("{}")
    policy = {"root": str(tmp_path), "protocol_sha256": sha256_file(tmp_path/"protocol.json"), "disk_reserve_gib": 200}
    original = tmp_path/module.AMENDMENT; original.write_text(json.dumps(policy))
    waiver = {**policy, "superseded_resource_sha256": sha256_file(original), "user_authorized": True,
              "minimum_write_safety_mib": 256}
    (tmp_path/module.DISK_WAIVER).write_text(json.dumps(waiver))
    assert module.load_policy(tmp_path)["disk_budget_waived"]
    assert json.loads(original.read_text()) == policy
    (tmp_path/module.DISK_WAIVER).write_text(json.dumps({**waiver, "user_authorized": False}))
    with pytest.raises(ValueError, match="authorized"): module.load_policy(tmp_path)


def test_multiworker_scheduler_runs_each_shard_once_and_waits_for_completion(tmp_path, monkeypatch):
    (tmp_path/"logs").mkdir()
    policy = {"disk_reserve_gib": 200, "remaining_write_allowance_gib": 20, "gpu_headroom_mib": 8192,
              "oom_cooldown_seconds": 0, "poll_seconds": 0,
              "stages": {"measure": {"worker_budget_mib": 23552, "max_workers_per_gpu": 3}}}
    monkeypatch.setattr(module, "load_policy", lambda root: policy)
    monkeypatch.setattr(module, "check_disk", lambda *a: None)
    monkeypatch.setattr(module, "gpu_snapshot", lambda: [gpu()])
    monkeypatch.setattr(module, "process_inventory", lambda: {2: set()})
    monkeypatch.setattr(module.time, "sleep", lambda *a: None)
    complete, launched = set(), []
    class Process:
        def __init__(self, args, **kwargs):
            self.shard = int(args[args.index("--shard")+1]); self.pid = 100+self.shard
            self.polls = 0; launched.append(self.shard)
        def poll(self):
            self.polls += 1
            if self.polls < 2: return None
            complete.add(self.shard); return 0
    monkeypatch.setattr(module.subprocess, "Popen", Process)
    p = SimpleNamespace(root=tmp_path, repo=tmp_path, env={}, config={"evaluation": {"shards": 8}},
                        shard_complete=lambda m, u, s, start=None: s in complete,
                        status=lambda *a, **k: None, refresh_report=lambda: None)
    module.run_parallel(p, "phase2.measure", 40)
    assert sorted(launched) == list(range(8)) and len(complete) == 8


def test_committed_checkpoint_eval_resumes_below_training_disk_threshold(tmp_path, monkeypatch):
    from phase2 import run_extended as runner
    p = runner.ExtendedPipeline.__new__(runner.ExtendedPipeline)
    p.root = tmp_path; p.extended = True; p.repo = tmp_path
    p.config = {"parent_update": 35, "post_updates": [40], "windows": [], "evaluation": {"shards": 8}}
    (tmp_path/"models/u0035").mkdir(parents=True)
    p.parallel = lambda *a: None; p.command = lambda *a: None
    p.status = lambda *a, **k: None; p.refresh_report = lambda: None
    p.train = lambda *a: pytest.fail("U40 already committed; never retrain")
    monkeypatch.setattr(runner, "read_committed_step", lambda p: 40)
    monkeypatch.setattr(runner, "validate_full_checkpoint", lambda p: None)
    monkeypatch.setattr(module, "load_policy", lambda p: {"disk_reserve_gib": 200, "remaining_write_allowance_gib": 20})
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: SimpleNamespace(free=244*2**30))
    p.run()
    assert (tmp_path/"completed_updates.jsonl").exists()
