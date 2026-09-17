"""New-cohort runner: every-update evidence, wider-window labels and locked probes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess

from phase1.archive import append_jsonl_idempotent, atomic_write_json, utc_now
from phase1.watch_qwen35_checkpoints import read_committed_step, validate_full_checkpoint
from phase2.protocol import parent_update, signal_directory
from phase2.run_fast import Pipeline


class ExtendedPipeline(Pipeline):
    def refresh_report(self):
        with (self.root/"logs/report-extended.log").open("a") as log:
            subprocess.run([sys.executable,"-m","phase2.window_report","--root",str(self.root)],
                           cwd=self.repo,env=self.env,stdout=log,stderr=subprocess.STDOUT)

    def run(self):
        if not self.extended:raise ValueError("Historical cohorts must use their original runner")
        parent=parent_update(self.config)
        if self.config.get("baseline_source"):
            self.command([sys.executable,"-m","phase2.import_baseline","--root",str(self.root)],"import-matched-baseline")
        base=self.root/"models"/f"u{parent:04d}"
        if not base.exists():
            self.command([sys.executable,"-m","phase2.export_model","--checkpoint",self.config["parent_checkpoint"],"--target",str(base)],f"export-u{parent}")
        self.parallel("phase2.evaluate",parent)
        endpoints={u for w in self.config["windows"] for u in (w["start"],w["end"])}
        shards=self.config["evaluation"].get("shards",8)
        donefile=self.root/"completed_updates.jsonl"
        completed={json.loads(line)["global_update"] for line in donefile.read_text().splitlines() if line.strip()} if donefile.exists() else set()
        for update in self.config["post_updates"]:
            self.update=update
            if update in completed:
                # Native optimizer checkpoints may have been intentionally
                # rotated; retain and verify the complete FP32 policy instead.
                if not (self.root/"models"/f"u{update:04d}"/"phase2_export.json").exists() or not (signal_directory(self.root,update)/"committed.json").exists():
                    raise RuntimeError("Completed update is missing retained policy/signal evidence")
                continue
            checkpoint=self.root/"checkpoints"/f"global_step_{update}"
            requires_training=(read_committed_step(self.root/"checkpoints") or 0)<update
            from phase2.evaluation_resources import load_policy, check_disk
            execution=load_policy(self.root)
            if not requires_training and execution is not None:
                check_disk(self.root,execution)
            elif shutil.disk_usage(self.root).free<250*2**30:
                raise RuntimeError("Free disk below 250GiB reserve; stopped at a recovery boundary")
            if requires_training:self.train(update)
            validate_full_checkpoint(checkpoint)
            if self.config.get("storage"):
                from phase2.storage import release_run_conversion, rotate_recovery
                release_run_conversion(self.root,self.config,update)
                rotate_recovery(self.root,self.config,update)
            self.command([sys.executable,"-m","phase2.export_model","--checkpoint",str(checkpoint),"--target",str(self.root/"models"/f"u{update:04d}")],f"export-u{update}")
            if not (signal_directory(self.root,update)/"committed.json").exists():
                self.parallel("phase2.measure",update)
                self.command([sys.executable,"-m","phase2.aggregate","--root",str(self.root),"--update",str(update),"--shards",str(shards)],f"aggregate-u{update}")
            for window in self.config["windows"]:
                if window["end"]!=update:continue
                start=window["start"]
                directory=signal_directory(self.root,update,start)
                if not (directory/"committed.json").exists():
                    self.parallel("phase2.measure",update,start)
                    self.command([sys.executable,"-m","phase2.aggregate","--root",str(self.root),"--start-update",str(start),"--update",str(update),"--shards",str(shards)],f"aggregate-window-{start}-{update}")
                self.command([sys.executable,"-m","phase2.window_forecast","--root",str(self.root),"--start-update",str(start),"--update",str(update)],f"forecast-window-{start}-{update}")
            if update in endpoints:self.parallel("phase2.evaluate",update)
            append_jsonl_idempotent(self.root/"completed_updates.jsonl",[{"global_update":update,"completed_at":utc_now()}],unique_fields=("global_update",))
        self.status("extended_cohort_complete",completed=True)
        self.refresh_report()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--detach",action="store_true")
    a=p.parse_args()
    if a.detach:
        with (a.root/"supervisor.log").open("a") as log:
            process=subprocess.Popen([sys.executable,"-u","-m","phase2.run_extended","--root",str(a.root)],
                                     cwd=Path(__file__).resolve().parents[1],stdout=log,stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,start_new_session=True)
        atomic_write_json(a.root/"supervisor_launch.json",{"at":utc_now(),"pid":process.pid})
        print(f"Extended pipeline PID {process.pid}");return
    with (a.root/"pipeline.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pipeline=ExtendedPipeline(a.root)
        try:pipeline.run()
        except Exception as error:
            pipeline.status("stopped_on_error",error=str(error))
            raise


if __name__=="__main__":main()
