"""Read-only GPU polling and conservative startup-only OOM retry checks."""
import subprocess
import json
from pathlib import Path
from phase2.protocol import training_run_id


def gpu_snapshot():
    output=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                    "--format=csv,noheader,nounits"],text=True)
    result=[]
    for line in output.splitlines():
        index,used,total,util=[int(value.strip()) for value in line.split(",")]
        if 0<=index<8:
            result.append({"gpu":index,"used_mib":used,"total_mib":total,"utilization":util})
    return result


def idle_devices(snapshot):
    # Low utilization alone does NOT make a GPU occupied by another job free.
    return [row["gpu"] for row in snapshot if row["used_mib"]<1500 and row["utilization"]<10]


def training_evidence(root,repo,update):
    """Active evidence that must not be overwritten by a fresh training attempt."""
    root,repo=Path(root),Path(repo)
    config=json.loads((root/"protocol.json").read_text()) if (root/"protocol.json").exists() else {}
    forbidden=[root/"checkpoints"/f"global_step_{update}"]
    for kind in ("batches","old_logprobs","new_logprobs","signals","evaluations"):
        forbidden.append(root/kind/f"u{update:04d}")
    for kind in ("trajectories","training_steps"):
        forbidden.append(repo/"artifacts"/kind/training_run_id(config,update))
    return [path for path in forbidden if path.exists() and
            (not path.is_dir() or any(path.iterdir()))] + list(
                (root/"optimizer_steps").glob(f"u{update:04d}-rank*.jsonl"))


def startup_oom_retryable(log,root,repo,update):
    """Never retry a computation/evidence-bearing attempt into the same paths."""
    if "out of memory" not in log.lower() or "actor_rollout_init_model" not in log:return False
    if any(marker in log for marker in ("Training Progress:","actor_rollout_update_actor","actor_rollout_generate_sequences")):
        return False
    return not training_evidence(root,repo,update)
