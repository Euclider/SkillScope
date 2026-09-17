from types import SimpleNamespace

import pytest

from phase2.resource_watch import idle_devices,startup_oom_retryable


OOM="actor_rollout_init_model() torch.OutOfMemoryError: CUDA out of memory"


def test_occupied_idle_utilization_is_not_a_free_gpu():
    assert idle_devices([{"gpu":0,"used_mib":28000,"utilization":0},
                         {"gpu":1,"used_mib":13,"utilization":0},
                         {"gpu":2,"used_mib":13,"utilization":100}])==[1]


def test_retry_only_pre_training_oom(tmp_path):
    assert startup_oom_retryable(OOM,tmp_path,tmp_path,34)
    assert not startup_oom_retryable("actor_rollout_init_model() ValueError",tmp_path,tmp_path,34)
    for marker in ("Training Progress:","actor_rollout_update_actor","actor_rollout_generate_sequences"):
        assert not startup_oom_retryable(OOM+marker,tmp_path,tmp_path,34)


@pytest.mark.parametrize("path",["batches/u0034/a.pt","old_logprobs/u0034/a.pt",
    "optimizer_steps/u0034-rank0.jsonl","checkpoints/global_step_34/data.pt",
    "artifacts/trajectories/phase2-s303-fast-u34/a.json",
    "artifacts/training_steps/phase2-s303-fast-u34/step-000034.json"])
def test_no_automatic_retry_if_any_update_evidence_exists(tmp_path,path):
    evidence=tmp_path/path;evidence.parent.mkdir(parents=True);evidence.touch()
    assert not startup_oom_retryable(OOM,tmp_path,tmp_path,34)


def test_monitor_requires_two_consecutive_free_samples(tmp_path,monkeypatch):
    import phase2.run_fast as runner
    pipeline=runner.Pipeline.__new__(runner.Pipeline)
    pipeline.root=tmp_path;pipeline.update=34
    pipeline.status=lambda *args,**kwargs:None
    pipeline.refresh_report=lambda:None
    polls=iter([[{"gpu":0,"used_mib":13,"utilization":0}],
                [{"gpu":0,"used_mib":30000,"utilization":0}],
                [{"gpu":1,"used_mib":13,"utilization":0}],
                [{"gpu":1,"used_mib":13,"utilization":0}]])
    monkeypatch.setattr(runner,"gpu_snapshot",lambda:next(polls))
    monkeypatch.setattr(runner.shutil,"disk_usage",lambda _:SimpleNamespace(free=500*2**30))
    monkeypatch.setattr(runner.time,"sleep",lambda _:None)
    assert pipeline.wait_training_gpus()==[1]
    assert len((tmp_path/"gpu_watch.jsonl").read_text().splitlines())==4


def test_startup_oom_is_archived_then_retried_without_touching_parent(tmp_path,monkeypatch):
    import json
    import phase2.run_fast as runner
    p=runner.Pipeline.__new__(runner.Pipeline)
    p.root=tmp_path;p.repo=tmp_path
    p.status=lambda *args,**kwargs:None;p.refresh_report=lambda:None
    parent=tmp_path/"checkpoints/global_step_33/data.pt"
    parent.parent.mkdir(parents=True);parent.write_text("immutable parent")
    log=tmp_path/"attempt.log";log.write_text("older attempt\n"+OOM)
    calls=[]
    def attempt(update):
        calls.append(update)
        if len(calls)==1:raise runner.CommandFailure("train-u34",1,log,len("older attempt\n"))
        return "resumed"
    p._train_attempt=attempt
    monkeypatch.setattr(runner.time,"sleep",lambda _:None)
    assert p.train(34)=="resumed"
    assert calls==[34,34]
    archive=next((tmp_path/"attempts").iterdir())
    assert (archive/"attempt.log").read_text()==OOM
    assert json.loads((archive/"failure.json").read_text())["startup_oom_retryable"]
    assert parent.read_text()=="immutable parent"


def test_training_attempts_have_separate_manifests_and_stable_experiment_ids(tmp_path,monkeypatch):
    import json
    import phase2.run_fast as runner
    p=runner.Pipeline.__new__(runner.Pipeline)
    p.root=tmp_path;p.repo=tmp_path;p.config={};p.env={}
    p.wait_training_gpus=lambda:[0,1,2,3,4,6]
    p.free_gpus=lambda:[0,1,2,3,4,6]
    monkeypatch.setattr(runner,"validate_full_checkpoint",lambda _: {"world_size":6})
    commands=[]
    p.command=lambda command,name,env:commands.append((command,name,env))
    p._train_attempt(34);p._train_attempt(34)
    assert len(commands)==2
    assert commands[0][0]==commands[1][0]==["bash","phase2/run_training_update.sh","34"]
    assert commands[0][2]["PHASE1_MANIFEST_OUTPUT_DIR"]!=commands[1][2]["PHASE1_MANIFEST_OUTPUT_DIR"]
    assert commands[0][2]["PHASE2_RESUME_CHECKPOINT"]==str(tmp_path/"checkpoints/global_step_33")
    assert commands[0][2]["PHASE2_GPU_IDS"]=="0,1,2,3,4,6"
    allocation=json.loads((tmp_path/"allocations/u0034.json").read_text())
    assert allocation["launch_manifest_dir"]==commands[1][2]["PHASE1_MANIFEST_OUTPUT_DIR"]


def test_manual_restart_also_refuses_unarchived_training_evidence(tmp_path):
    import phase2.run_fast as runner
    p=runner.Pipeline.__new__(runner.Pipeline)
    p.root=tmp_path;p.repo=tmp_path
    evidence=tmp_path/"batches/u0034/training_batch.pt"
    evidence.parent.mkdir(parents=True);evidence.write_bytes(b"old evidence")
    with pytest.raises(RuntimeError,match="uncommitted evidence"):
        p._train_attempt(34)
    assert evidence.read_bytes()==b"old evidence"
