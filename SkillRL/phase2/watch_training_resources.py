"""Read-only runtime GPU evidence; never reserve memory or terminate a job."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import psutil

from phase1.archive import append_jsonl_idempotent,atomic_write_json,utc_now


def parse_snapshot(gpu_csv,process_csv):
    gpus=[];uuid_to_index={}
    for line in gpu_csv.splitlines():
        if not line.strip():continue
        index,uuid,used,free,total,util=[field.strip() for field in line.split(",")]
        row={"gpu":int(index),"uuid":uuid,"used_mib":int(used),"free_mib":int(free),
             "total_mib":int(total),"utilization":int(util)}
        gpus.append(row);uuid_to_index[uuid]=int(index)
    processes=[]
    for line in process_csv.splitlines():
        if not line.strip():continue
        pid,uuid,used=[field.strip() for field in line.split(",")]
        processes.append({"pid":int(pid),"gpu":uuid_to_index.get(uuid),"gpu_uuid":uuid,
                          "used_mib":int(used) if used.isdigit() else None})
    return {"gpus":gpus,"compute_processes":processes}


def snapshot():
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu",
                                 "--format=csv,noheader,nounits"],text=True,timeout=10)
    processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory",
                                       "--format=csv,noheader,nounits"],text=True,timeout=10)
    return parse_snapshot(gpu,processes)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--pipeline-pid",type=int,required=True)
    args=parser.parse_args()
    process=psutil.Process(args.pipeline_pid);born=process.create_time()
    directory=args.root/"runtime_resources"
    output=directory/f"pipeline-{args.pipeline_pid}.jsonl"
    while True:
        try:
            if not process.is_running() or process.status()==psutil.STATUS_ZOMBIE or process.create_time()!=born:break
            status=json.loads((args.root/"status.json").read_text())
            row={"at":utc_now(),"pipeline_pid":args.pipeline_pid,"stage":status.get("stage"),
                 "update":status.get("global_update"),"disk_free_bytes":psutil.disk_usage(str(args.root)).free}
            try:
                row.update(snapshot())
                row["pipeline_descendant_pids"]=[p.pid for p in process.children(recursive=True)]
            except (subprocess.SubprocessError,psutil.Error) as error:row["sampling_error"]=str(error)
            append_jsonl_idempotent(output,[row],unique_fields=("at",))
            atomic_write_json(directory/"latest.json",row)
            time.sleep(15)
        except psutil.NoSuchProcess:break
    atomic_write_json(directory/f"pipeline-{args.pipeline_pid}-stopped.json",{
        "at":utc_now(),"reason":"watched pipeline exited","pipeline_pid":args.pipeline_pid})


if __name__=="__main__":main()
