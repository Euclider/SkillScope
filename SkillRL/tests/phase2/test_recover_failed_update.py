import json

import pytest

from phase1.archive import atomic_write_json,sha256_file
from phase2.recover_failed_update import recover


def fixture(tmp_path,monkeypatch):
    import phase2.recover_failed_update as recovery
    repo=tmp_path/"repo";root=repo/"artifacts/phase2/test"
    attempt=root/"attempts/failure"
    atomic_write_json(attempt/"failure.json",{"update":34})
    (attempt/"attempt.log").write_text("backward: CUDA out of memory")
    batch=root/"batches/u0034";batch.mkdir(parents=True)
    (batch/"training_batch.pt").write_bytes(b"frozen-batch")
    atomic_write_json(batch/"manifest.json",{"batch_sha256":sha256_file(batch/"training_batch.pt"),
                                            "row_count":3,"unique_decisions":2,"loss_tokens":10})
    atomic_write_json(root/"old_logprobs/u0034/row-0.json",{"old":True})
    atomic_write_json(root/"elastic_restore_audits/u0034/rank-0.json",{"checksums_match":True})
    atomic_write_json(root/"forward_progress/rank-0.json",{"global_update":34})
    atomic_write_json(root/"forward_progress/rank-7.json",{"global_update":31})
    run="phase2-s303-fast-u34"
    atomic_write_json(repo/f"artifacts/trajectories/{run}/abc.json",{"trajectory_id":"abc"})
    index=repo/"artifacts/trajectories/index.jsonl"
    index.write_text(json.dumps({"run_id":"unrelated","trajectory_id":"keep"})+"\n"+
                     json.dumps({"run_id":run,"trajectory_id":"abc"})+"\n")
    parent=root/"checkpoints/global_step_33/data.pt"
    parent.parent.mkdir(parents=True);parent.write_bytes(b"immutable-parent")
    monkeypatch.setattr(recovery,"validate_full_checkpoint",lambda _: {"world_size":4})
    return repo,root,attempt,index,parent


def test_failed_evidence_preserved_and_only_target_index_rows_removed(tmp_path,monkeypatch):
    repo,root,attempt,index,parent=fixture(tmp_path,monkeypatch)
    original=index.read_bytes()
    result=recover(root,repo,34,attempt)
    assert result["status"]=="archived_ready_for_fresh_retry"
    assert result["trajectory_count"]==1
    assert parent.read_bytes()==b"immutable-parent"
    assert not (root/"batches/u0034").exists()
    assert (attempt/"evidence/batches/u0034/training_batch.pt").read_bytes()==b"frozen-batch"
    assert (attempt/"evidence/artifacts/trajectories/phase2-s303-fast-u34/abc.json").exists()
    assert (attempt/"evidence/trajectory-index-before.jsonl").read_bytes()==original
    assert json.loads(index.read_text())["run_id"]=="unrelated"
    assert (root/"forward_progress/rank-7.json").exists()
    with pytest.raises(ValueError,match="already started"):
        recover(root,repo,34,attempt)


@pytest.mark.parametrize("path",["optimizer_steps/u0034-rank0.jsonl","checkpoints/global_step_34/data.pt",
                                 "predictions/u0034.json","signals/u0034/committed.json"])
def test_recovery_rejects_post_update_evidence(tmp_path,monkeypatch,path):
    repo,root,attempt,index,parent=fixture(tmp_path,monkeypatch)
    target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text("exists")
    original=index.read_bytes()
    with pytest.raises(ValueError,match="separate recovery audit"):
        recover(root,repo,34,attempt)
    assert index.read_bytes()==original
    assert not (attempt/"evidence").exists()


@pytest.mark.parametrize("world,allowed",[(4,True),(2,False)])
def test_missing_elastic_audit_only_allowed_for_native_same_world_restore(tmp_path,monkeypatch,world,allowed):
    import shutil
    repo,root,attempt,index,parent=fixture(tmp_path,monkeypatch)
    shutil.rmtree(root/"elastic_restore_audits/u0034")
    atomic_write_json(attempt/"u0034.json",{"world_size":world,"resume_checkpoint":str(parent.parent)})
    if allowed:
        assert recover(root,repo,34,attempt)["restore_audit"]=="native_same_world_checkpoint"
    else:
        with pytest.raises(ValueError,match="Missing elastic restore audit"):
            recover(root,repo,34,attempt)
