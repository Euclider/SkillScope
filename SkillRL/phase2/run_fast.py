"""Durable, sequential training -> signals -> locked forecast -> gold pipeline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from phase1.archive import append_jsonl_idempotent, atomic_write_json, utc_now
from phase1.watch_qwen35_checkpoints import read_committed_step, validate_full_checkpoint
from phase2.resource_watch import gpu_snapshot, idle_devices, startup_oom_retryable, training_evidence
from phase2.protocol import parent_update, signal_directory, training_run_id, validate_extended


class CommandFailure(RuntimeError):
    def __init__(self,name,code,logpath,offset):
        super().__init__(f"{name} exited {code}; see {logpath}")
        self.logpath=logpath;self.offset=offset


class Pipeline:
    def __init__(self,root):
        self.root=root
        self.repo=Path(__file__).resolve().parents[1]
        self.config=json.loads((root/"protocol.json").read_text())
        self.extended=self.config.get("schema_version","").startswith("phase2.extended.")
        if self.extended:validate_extended(self.config,self.repo)
        policy=json.loads((self.repo/"phase2/config/elastic_resource_policy_v1.json").read_text())
        policy_path=root/"resource_policy.json"
        if policy_path.exists() and json.loads(policy_path.read_text())!=policy:
            raise ValueError("Resource policy changed without a separately archived amendment")
        atomic_write_json(policy_path,policy)
        for source,target in (("signed_analysis_amendment_v2.json","analysis_amendment_v2.json"),
                              ("resource_watch_amendment_v2.json","resource_watch_amendment_v2.json")):
            if self.extended:continue  # A new frozen protocol replaces pilot-specific amendments.
            value=json.loads((self.repo/"phase2/config"/source).read_text())
            destination=root/target
            if target=="analysis_amendment_v2.json" and not destination.exists():
                for update in value["heldout_updates_not_yet_evaluated_at_amendment"]:
                    if list((root/"evaluations"/f"u{update:04d}").glob("shard-*.jsonl")):
                        raise ValueError("Cannot register this amendment after heldout gold has opened")
            if destination.exists() and json.loads(destination.read_text())!=value:
                raise ValueError(f"Immutable amendment changed: {destination}")
            atomic_write_json(destination,value)
        self.env=os.environ.copy()
        self.env.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",TOKENIZERS_PARALLELISM="false",
                        ALFWORLD_DATA="/home/wangyifan/skill-RL/data/alfworld",
                        HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
        self.stage="starting"
        self.update=parent_update(self.config)+1

    def status(self,stage=None,**extra):
        if stage:self.stage=stage
        atomic_write_json(self.root/"status.json",{"updated_at":utc_now(),"stage":self.stage,
                                                  "global_update":self.update,"pipeline_pid":os.getpid(),**extra})

    def refresh_report(self):
        if self.extended:
            # Never write a new cohort into the historical fast-v1 report/metrics.
            return
        with (self.root/"logs/report.log").open("a") as log:
            subprocess.run([sys.executable,"-m","phase2.report","--root",str(self.root)],cwd=self.repo,env=self.env,stdout=log,stderr=subprocess.STDOUT)

    def command(self,command,name,env=None):
        logpath=self.root/"logs"/f"{name}.log"
        offset=logpath.stat().st_size if logpath.exists() else 0
        self.status(name)
        print(f"[{utc_now()}] {name}",flush=True)
        with logpath.open("a") as log:
            proc=subprocess.Popen(command,cwd=self.repo,env=env or self.env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                self.status(name,worker_pid=proc.pid,log=str(logpath))
                time.sleep(15)
            if proc.returncode:
                raise CommandFailure(name,proc.returncode,logpath,offset)

    def free_gpus(self):
        return idle_devices(gpu_snapshot())

    def wait_training_gpus(self):
        last_report=0.
        previous=set()
        while True:
            if shutil.disk_usage(self.root).free<250*2**30:
                raise RuntimeError("Free disk below 250GiB reserve while waiting for GPUs")
            snapshot=gpu_snapshot()
            available=set(idle_devices(snapshot))
            stable=sorted(available&previous)
            record={"at":utc_now(),"update":self.update,"gpus":snapshot,
                    "idle_gpus":sorted(available),"confirmed_idle_gpus":stable}
            append_jsonl_idempotent(self.root/"gpu_watch.jsonl",[record],unique_fields=("at",))
            atomic_write_json(self.root/"gpu_watch_latest.json",record)
            if stable:return stable
            self.status("waiting_for_any_free_training_gpu",
                        waiting_stage=f"training-u{self.update}",gpu_snapshot=snapshot,
                        idle_candidates=sorted(available),poll_seconds=30)
            if time.monotonic()-last_report>300:
                self.refresh_report()
                last_report=time.monotonic()
            previous=available
            time.sleep(30)

    def train(self,update):
        while True:
            try:
                return self._train_attempt(update)
            except CommandFailure as error:
                with error.logpath.open("rb") as source:
                    source.seek(error.offset)
                    output=source.read().decode("utf-8",errors="replace")
                retry=startup_oom_retryable(output,self.root,self.repo,update)
                directory=self.root/"attempts"/f"u{update:04d}-failed-{time.time_ns()}"
                directory.mkdir(parents=True)
                (directory/"attempt.log").write_text(output)
                allocation_file=self.root/"allocations"/f"u{update:04d}.json"
                allocation=json.loads(allocation_file.read_text()) if allocation_file.exists() else {}
                manifest_directory=Path(allocation.get("launch_manifest_dir",self.repo/"artifacts"))
                for file in (allocation_file,self.root/"status.json",manifest_directory/"preflight-step-router.json",
                             manifest_directory/"manifests"/f"{training_run_id(getattr(self,'config',{}),update)}.json"):
                    if file.exists():shutil.copy2(file,directory/file.name)
                atomic_write_json(directory/"failure.json",{"at":utc_now(),"update":update,
                    "error":str(error),"log_offset":error.offset,"startup_oom_retryable":retry})
                if not retry:raise
                print(f"[{utc_now()}] Archived initialization OOM for U{update}; no new training evidence; returning to GPU watch",flush=True)
                self.status("waiting_after_startup_oom",failure_archive=str(directory))
                self.refresh_report()
                time.sleep(30)

    def _train_attempt(self,update):
        existing=training_evidence(self.root,self.repo,update)
        if existing:
            raise RuntimeError(f"U{update} has uncommitted evidence; archive/audit the failed attempt before retrying: {existing}")
        source=(Path(self.config["parent_checkpoint"]) if update==parent_update(self.config)+1 else
                self.root/"checkpoints"/f"global_step_{update-1}")
        source_world=validate_full_checkpoint(source)["world_size"]
        while True:
            gpus=self.wait_training_gpus()
            world=len(gpus)
            resume=source
            scratch_parent=self.root/"elastic_resume"
            if getattr(self,"extended",False) and self.config.get("storage"):
                from phase2.staged_storage import admit, load_amendment
                amendment=load_amendment(self.root,update)
                settings=self.config["storage"]
                allowance=settings.get("minimum_next_update_allowance_gib",100)+(50 if world!=source_world else 0)
                free_gib=shutil.disk_usage(self.root).free/2**30
                if amendment:
                    admit(self.root,self.config,update,world=world,source_world=source_world)
                    scratch_parent=Path(amendment["scratch_parent"])
                elif free_gib-allowance<settings.get("projected_transient_reserve_gib",200):
                    raise RuntimeError(f"Projected storage reserve insufficient: free={free_gib:.1f}GiB, next-update allowance={allowance}GiB")
            if world!=source_world:
                resume=scratch_parent/f"u{update-1:04d}-w{world}"/f"global_step_{update-1}"
                self.command([sys.executable,"-u","-m","phase2.elastic_checkpoint",
                              "--checkpoint",str(source),"--model",str(self.root/"models"/f"u{update-1:04d}"),
                              "--target",str(resume),"--world-size",str(world)],f"elastic-u{update-1}-w{world}")
            if set(gpus)<=set(self.free_gpus()):break
            print(f"[{utc_now()}] GPU availability changed during conversion; selecting remaining free devices",flush=True)
        train=self.config.get("training",{})
        allocation={"created_at":utc_now(),"global_update":update,"physical_gpus":gpus,
                    "world_size":world,"source_world_size":source_world,"source_checkpoint":str(source),
                    "resume_checkpoint":str(resume),"training_rollouts":train.get("games_per_update",8)*train.get("rollouts_per_game",4),"learning_rate":train.get("lr",1e-6),
                    "optimizer_global_decision_batch":32,"optimizer_local_slots":(32+world-1)//world,
                    "optimizer_backend":"cpu_adamw" if world<=2 else "gpu_adamw",
                    "selection_rule":"all currently free GPUs, frozen for this update; no in-update resize"}
        manifest_dir=self.root/"launch_manifests"/f"u{update:04d}-{time.time_ns()}"
        allocation["launch_manifest_dir"]=str(manifest_dir)
        atomic_write_json(self.root/"allocations"/f"u{update:04d}.json",allocation)
        append_jsonl_idempotent(self.root/"allocation_history.jsonl",[allocation],unique_fields=("created_at",))
        env={**self.env,"PHASE2_GPU_IDS":",".join(map(str,gpus)),"PHASE2_RESUME_CHECKPOINT":str(resume),
             "PHASE1_MANIFEST_OUTPUT_DIR":str(manifest_dir)}
        command=([sys.executable,"-m","phase2.launch_training","--protocol",str(self.root/"protocol.json"),"--update",str(update)]
                 if getattr(self,"extended",False) else ["bash","phase2/run_training_update.sh",str(update)])
        self.command(command,f"train-u{update}",env=env)

    def shard_complete(self,module,update,shard,start_update=None):
        if module=="phase2.evaluate":
            directory=self.root/"evaluations"/f"u{update:04d}"
            marker=directory/f"shard-{shard}-complete.json"
            if not marker.exists():return False
            metadata=json.loads(marker.read_text())
            if metadata["max_jobs"] is not None:return False
            index=directory/f"shard-{shard}.jsonl"
            rows=[json.loads(line) for line in index.read_text().splitlines() if line.strip()]
            if len(rows)!=metadata["jobs"] or len({x["trajectory_id"] for x in rows})!=len(rows):
                raise RuntimeError(f"Completed evaluation shard has an inconsistent index: {marker}")
            return True
        if module=="phase2.measure":
            directory=signal_directory(self.root,update,start_update)
            marker=directory/f"shard-{shard}.json"
            if not marker.exists():return False
            metadata=json.loads(marker.read_text())
            return metadata["max_decisions"] is None and (directory/f"tokens-shard-{shard}.parquet").exists() and (directory/f"decisions-shard-{shard}.parquet").exists()
        raise ValueError(f"No completion contract for {module}")

    def parallel(self,module,update,start_update=None):
        from phase2.evaluation_resources import load_policy, run_parallel
        if load_policy(self.root) is not None:
            return run_parallel(self,module,update,start_update)
        name=f"{module.rsplit('.',1)[-1]}-u{update}"+(f"-from{start_update}" if start_update is not None else "")
        shards=self.config.get("evaluation",{}).get("shards",8)
        pending=[shard for shard in range(shards) if not self.shard_complete(module,update,shard,start_update)]
        if not pending:
            print(f"[{utc_now()}] {name}: all eight logical shards already complete; skipping",flush=True)
            return
        self.status(name)
        print(f"[{utc_now()}] {name}: scheduling {len(pending)} logical shards on currently free GPUs",flush=True)
        processes={}
        last_report=0.
        try:
            while pending or processes:
                for gpu,(proc,log,shard) in list(processes.items()):
                    code=proc.poll()
                    if code is None:continue
                    log.close()
                    del processes[gpu]
                    if code or not self.shard_complete(module,update,shard,start_update):
                        raise RuntimeError(f"{name} shard {shard} failed (exit {code}); see shard log")
                memory=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
                free=[]
                for line in memory.splitlines():
                    gpu,used=(int(x.strip()) for x in line.split(","))
                    if 0<=gpu<8 and used<1500 and gpu not in processes:free.append(gpu)
                for gpu in free:
                    if not pending:break
                    shard=pending.pop(0)
                    env={**self.env,"CUDA_VISIBLE_DEVICES":str(gpu)}
                    log=(self.root/"logs"/f"{name}-shard{shard}.log").open("a")
                    args=[sys.executable,"-u","-m",module,"--root",str(self.root),"--update",str(update),"--shard",str(shard),"--shards",str(shards)]
                    if start_update is not None:args += ["--start-update",str(start_update)]
                    proc=subprocess.Popen(args,cwd=self.repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    processes[gpu]=(proc,log,shard)
                    print(f"[{utc_now()}] {name} shard {shard} -> GPU {gpu}, PID {proc.pid}",flush=True)
                self.status(name if processes else f"waiting_for_free_gpu-{name}",
                            evaluation_update=update,pending_shards=pending,
                            workers=[{"gpu":gpu,"shard":shard,"pid":proc.pid,"returncode":proc.poll()} for gpu,(proc,_,shard) in processes.items()])
                if time.monotonic()-last_report>300:
                    self.refresh_report()
                    last_report=time.monotonic()
                if pending or processes:time.sleep(15)
        finally:
            for proc,log,_ in processes.values():
                if proc.poll() is None:proc.terminate()
                log.close()
        self.refresh_report()

    def run(self,adopt_pid=None):
        if self.extended:raise ValueError("Use phase2.run_extended for a window cohort")
        donefile=self.root/"completed_updates.jsonl"
        completed={json.loads(line)["global_update"] for line in donefile.read_text().splitlines() if line.strip()} if donefile.exists() else set()
        for update in self.config["post_updates"]:
            self.update=update
            if update in completed:
                validate_full_checkpoint(self.root/"checkpoints"/f"global_step_{update}")
                if not (self.root/"signals"/f"u{update:04d}"/"committed.json").exists() or not all(self.shard_complete("phase2.evaluate",update,shard) for shard in range(8)):
                    raise RuntimeError(f"Completed update {update} has missing evidence")
                print(f"[{utc_now()}] Update {update} fully archived; resuming next unfinished update",flush=True)
                continue
            if shutil.disk_usage(self.root).free < 250*2**30:
                raise RuntimeError("Free disk below the preregistered 250GiB reserve; stopped at a recovery boundary")
            checkpoint=self.root/"checkpoints"/f"global_step_{update}"
            if update==31 and adopt_pid:
                self.status("training-u31-adopted",worker_pid=adopt_pid)
                try:
                    process=psutil.Process(adopt_pid)
                    while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
                        self.status("training-u31-adopted",worker_pid=adopt_pid)
                        time.sleep(15)
                except psutil.NoSuchProcess:pass
                if (read_committed_step(self.root/"checkpoints") or 0)<31:
                    raise RuntimeError("Adopted U31 training exited before a committed checkpoint; inspect train-u31.log")
            elif (read_committed_step(self.root/"checkpoints") or 0)<update:
                self.train(update)
            validate_full_checkpoint(checkpoint)
            self.command([sys.executable,"-u","-m","phase2.export_model","--checkpoint",str(checkpoint),"--target",str(self.root/"models"/f"u{update:04d}")],f"export-u{update}")
            if not (self.root/"signals"/f"u{update:04d}"/"committed.json").exists():
                self.parallel("phase2.measure",update)
                self.command([sys.executable,"-m","phase2.aggregate","--root",str(self.root),"--update",str(update)],f"aggregate-u{update}")
            self.refresh_report()
            if update==31:
                self.parallel("phase2.evaluate",30)
            self.command([sys.executable,"-m","phase2.forecast","--root",str(self.root),"--update",str(update)],f"forecast-u{update}")
            self.parallel("phase2.evaluate",update)
            self.refresh_report()
            append_jsonl_idempotent(self.root/"completed_updates.jsonl",[{"global_update":update,"completed_at":utc_now()}],unique_fields=("global_update",))
        self.status("core_fast_cohort_complete",completed=True)
        self.refresh_report()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--adopt-training-pid",type=int)
    a=p.parse_args()
    import fcntl
    lock=(a.root/"pipeline.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pipeline=Pipeline(a.root)
    try:pipeline.run(a.adopt_training_pid)
    except Exception as error:
        pipeline.status("stopped_on_error",error=str(error))
        pipeline.refresh_report()
        raise


if __name__=="__main__":main()
