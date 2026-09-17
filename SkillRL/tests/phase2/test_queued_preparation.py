import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import phase2.queued_preparation as runner
from phase1.archive import sha256_file


def config(tmp_path):
    value = json.loads((runner.REPO/"phase2/config/ranking_preparation_u35_v1.json").read_text())
    (tmp_path/"games.txt").write_text("a\nb\n")
    (tmp_path/"bank.json").write_text(json.dumps({"task_specific_skills": {"clean": []}}))
    (tmp_path/"model").mkdir()
    (tmp_path/"model/model.safetensors").write_bytes(b"fake")
    value.update(game_ids_file="games.txt", game_count=2, skill_bank="bank.json", source_model=str(tmp_path/"model"), shards=2)
    return value


def test_preparation_queue_has_disjoint_seeds_and_cannot_enable_training(tmp_path):
    value = config(tmp_path)
    assert runner.validate_queue(value, tmp_path) == ["a", "b"]
    value["scope"]["launch_new_rl"] = True
    with pytest.raises(ValueError, match="cannot train"):
        runner.validate_queue(value, tmp_path)


def test_queue_rejects_seed_reuse_or_false_idle_rule(tmp_path):
    value = config(tmp_path)
    value["coverage_seeds"][0] = value["evaluation"]["gold_seeds"][0]
    with pytest.raises(ValueError, match="distinct"):
        runner.validate_queue(value, tmp_path)
    value["coverage_seeds"] = [991]
    value["resources"]["idle_used_mib_below"] = 50000
    with pytest.raises(ValueError, match="conservative"):
        runner.validate_queue(value, tmp_path)


def test_coverage_command_is_full_bank_natural_router_not_forced_skill(tmp_path):
    value = config(tmp_path)
    command = runner.coverage_command(value, tmp_path, 1)
    assert command[command.index("--conditions")+1] == "full_bank"
    assert "--step-routing" in command
    assert command[command.index("--game-shard-index")+1] == "1"
    assert command[command.index("--max-new-tokens")+1] == "64"
    assert not any("train" in arg for arg in command)


def test_source_mutation_prevents_resuming_queue(tmp_path):
    source = tmp_path/"source.py"
    source.write_text("old")
    queue = tmp_path/"queue.json"
    queue.write_text("{}")
    (tmp_path/"queue_manifest.json").write_text(json.dumps({"queue_sha256": sha256_file(queue),
        "source_sha256": {str(source): sha256_file(source)}, "input_stats": {}}))
    runner.verify_manifest(tmp_path)
    source.write_text("new")
    with pytest.raises(ValueError, match="source changed"):
        runner.verify_manifest(tmp_path)


def test_baseline_schema_cannot_open_a_future_checkpoint(tmp_path):
    value = {"schema_version": runner.BASELINE_SCHEMA, "status": "frozen", "parent_update": 35}
    with pytest.raises(ValueError, match="pre-update endpoint"):
        runner.validate_baseline(value, tmp_path, 36)


def test_completed_coverage_checks_archive_identity_and_files(tmp_path, monkeypatch):
    value = config(tmp_path)
    value["coverage_seeds"] = [991]
    monkeypatch.setattr(runner, "validate_queue", lambda *_: ["a", "b"])
    import phase1.eval_skill_margin as evaluator
    monkeypatch.setattr(evaluator, "resolve_game_file", lambda game: tmp_path/game)
    directory = tmp_path/"coverage"
    directory.mkdir()
    trajectory = tmp_path/"trajectory.json"
    trajectory.write_text("{}")
    row = {"game_id": str(tmp_path/"a"), "eval_seed": 991, "checkpoint_id": "model", "context_id": "clean",
           "skill_condition": "full_bank", "trajectory_path": str(trajectory)}
    (directory/"shard-0.jsonl").write_text(json.dumps(row)+"\n")
    assert runner.coverage_complete(value, tmp_path, 0)
    row["skill_condition"] = "minus_skill"
    (directory/"shard-0.jsonl").write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError, match="frozen source"):
        runner.coverage_complete(value, tmp_path, 0)


def test_scheduler_needs_two_idle_polls_and_only_starts_one_worker_per_gpu(tmp_path, monkeypatch):
    queue = runner.PreparationQueue.__new__(runner.PreparationQueue)
    queue.root = tmp_path
    queue.config = {"shards": 2, "resources": {"minimum_disk_free_gib": 250, "maximum_workers": 8, "poll_seconds": 30}}
    queue.env = {}
    (tmp_path/"logs").mkdir()
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "verify_manifest", lambda *_: None)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=400*2**30))
    sleeps, launches, complete, states = [], [], set(), []
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: sleeps.append(seconds))
    queue.status = lambda stage, **kw: states.append((stage, kw))
    monkeypatch.setattr(runner, "gpu_snapshot", lambda: [{"gpu": 1, "used_mib": 20, "utilization": 0},
                                                        {"gpu": 2, "used_mib": 50000, "utilization": 0}])
    class Process:
        def __init__(self, command, **kwargs):
            self.shard = int(command[0]); self.pid = 100+self.shard; self.returncode = None
            launches.append((self.shard, len(sleeps), kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
            assert kwargs["pass_fds"]
        def poll(self):
            self.returncode = 0; complete.add(self.shard); return 0
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    queue.schedule("coverage", lambda s: [str(s)], lambda s: s in complete)
    assert launches == [(0, 1, "1"), (1, 2, "1")]
    assert states[0][0] == "waiting_for_idle_gpu"
    assert complete == {0, 1}


def test_low_disk_keeps_queue_waiting_even_when_gpu_is_idle(tmp_path, monkeypatch):
    queue = runner.PreparationQueue.__new__(runner.PreparationQueue)
    queue.root = tmp_path; queue.config = {"shards": 1, "resources": {"minimum_disk_free_gib": 250, "maximum_workers": 8, "poll_seconds": 30}}
    queue.env = {}; stages = []
    queue.status = lambda stage, **kw: stages.append(stage)
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=200*2**30))
    monkeypatch.setattr(runner, "gpu_snapshot", lambda: [{"gpu": 0, "used_mib": 0, "utilization": 0}])
    def stop(_): raise RuntimeError("test stop")
    monkeypatch.setattr(runner.time, "sleep", stop)
    with pytest.raises(RuntimeError, match="test stop"):
        queue.schedule("coverage", lambda _: pytest.fail("must not launch"), lambda _: False)
    assert stages == ["waiting_for_disk_reserve"]


def test_baseline_summary_does_not_report_a_delta_or_count_repeats_as_anchors():
    import pandas as pd
    from phase2.preparation_report import summarize_margins
    from phase2.utilities import margins
    rows = []
    for seed in (1, 2, 3, 4):
        for arm in ("original", "placebo", "null"):
            rows.append({"update": 35, "purpose": "gold", "skill_id": "cle_003", "context_id": "clean",
                         "anchor_id": "a", "game_id": "g", "phase": "middle", "trigger_step": 7,
                         "continuation_seed": seed, "arm": arm, "success": arm == "original"})
    table = summarize_margins(margins(pd.DataFrame(rows)), 50)
    assert table.anchor_count.eq(1).all()
    assert table.continuation_repeats.eq(4).all()
    assert table.semantic_or_total_margin.eq(1).all()
    assert "delta_utility" not in table
