"""Explicit recovery of an OOM attempt with no completed optimizer step.

Never called by the automatic monitor. All evidence is retained in the existing
failure archive before its active paths are cleared for a user-requested retry.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase1.watch_qwen35_checkpoints import validate_full_checkpoint


def recover(root, repo, update, attempt):
    root,repo,attempt=Path(root).resolve(),Path(repo).resolve(),Path(attempt).resolve()
    if attempt.parent!=root/"attempts":raise ValueError("Expected an exact existing attempt under root/attempts")
    failure=json.loads((attempt/"failure.json").read_text())
    if failure["update"]!=update:raise ValueError("Failure/update mismatch")
    if "out of memory" not in (attempt/"attempt.log").read_text().lower():
        raise ValueError("This recovery helper only handles reviewed OOM failures")
    archive=attempt/"evidence"
    if archive.exists():raise ValueError("Recovery already started; inspect its journal before resuming")
    with (root/"pipeline.lock").open("a") as pipeline_lock:
        fcntl.flock(pipeline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent=root/"checkpoints"/f"global_step_{update-1}"
        parent_info=validate_full_checkpoint(parent)
        forbidden=[root/"checkpoints"/f"global_step_{update}",
                   root/"models"/f"u{update:04d}",root/"new_logprobs"/f"u{update:04d}",
                   root/"signals"/f"u{update:04d}",root/"evaluations"/f"u{update:04d}",
                   root/"predictions"/f"u{update:04d}.json",
                   repo/"artifacts/training_steps"/f"phase2-s303-fast-u{update}"]
        forbidden+=list((root/"optimizer_steps").glob(f"u{update:04d}-rank*.jsonl"))
        if any(p.exists() for p in forbidden):
            raise ValueError("Optimizer/post-update evidence exists; requires a separate recovery audit")
        run_id=f"phase2-s303-fast-u{update}"
        trajectories=repo/"artifacts/trajectories"/run_id
        batch=root/"batches"/f"u{update:04d}"
        manifest=json.loads((batch/"manifest.json").read_text())
        if sha256_file(batch/"training_batch.pt")!=manifest["batch_sha256"]:
            raise ValueError("Training batch checksum mismatch")
        paths=[batch,root/"old_logprobs"/f"u{update:04d}",trajectories]
        restore_audit=root/"elastic_restore_audits"/f"u{update:04d}"
        if restore_audit.exists():
            paths.append(restore_audit)
        else:
            allocation=json.loads((attempt/f"u{update:04d}.json").read_text())
            if (allocation["world_size"]!=parent_info["world_size"] or
                    Path(allocation["resume_checkpoint"]).resolve()!=parent):
                raise ValueError("Missing elastic restore audit for a non-native resume")
        paths += [p for p in (root/"forward_progress").glob("rank-*.json")
                  if json.loads(p.read_text())["global_update"]==update]
        if not all(p.exists() for p in paths):raise ValueError("Expected evidence missing")
        moves=[(p,archive/(p.relative_to(repo) if p==trajectories else p.relative_to(root))) for p in paths]
        index=repo/"artifacts/trajectories/index.jsonl"
        with index.open("r+",encoding="utf-8") as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            original=handle.read()
            lines=original.splitlines(keepends=True)
            selected=[line for line in lines if json.loads(line)["run_id"]==run_id]
            ids={json.loads(line)["trajectory_id"] for line in selected}
            if len(ids)!=len(selected) or ids!={p.stem for p in trajectories.glob("*.json")}:
                raise ValueError("Trajectory index/archive mismatch")
            archive.mkdir()
            # Full byte-preserving backup also permits recovery from interruption
            # while rewriting the small, shared index under its normal file lock.
            with (archive/"trajectory-index-before.jsonl").open("w",encoding="utf-8") as backup:
                backup.write(original);backup.flush();os.fsync(backup.fileno())
            with (archive/"trajectory-index-failed-attempt.jsonl").open("w",encoding="utf-8") as backup:
                backup.writelines(selected);backup.flush();os.fsync(backup.fileno())
            journal={"created_at":utc_now(),"update":update,"parent_checkpoint":str(parent),
                     "status":"archiving","batch_rows":manifest["row_count"],
                     "unique_decisions":manifest["unique_decisions"],"loss_tokens":manifest["loss_tokens"],
                     "trajectory_count":len(ids),"completed_optimizer_steps":0,
                     "restore_audit":"elastic" if restore_audit.exists() else "native_same_world_checkpoint",
                     "moves":[{"source":str(s),"archive":str(d)} for s,d in moves]}
            atomic_write_json(archive/"recovery.json",journal)
            for source,target in moves:
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(source),str(target))
            handle.seek(0)
            handle.writelines(line for line in lines if json.loads(line)["run_id"]!=run_id)
            handle.truncate();handle.flush();os.fsync(handle.fileno())
            journal.update(status="archived_ready_for_fresh_retry",completed_at=utc_now(),
                           active_index_sha256=sha256_file(index))
            atomic_write_json(archive/"recovery.json",journal)
        return journal


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--update",type=int,required=True)
    parser.add_argument("--attempt",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(recover(args.root,Path(__file__).resolve().parents[1],args.update,args.attempt),indent=2))


if __name__=="__main__":main()
