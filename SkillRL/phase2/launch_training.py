"""Protocol-driven launch for NEW cohorts, with explicit dry-run support."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from phase2.protocol import parent_update, training_run_id, validate_extended
from phase1.archive import stable_hash


def launch_spec(config, update, repo, inherited):
    validate_extended(config,repo)
    if update not in config["post_updates"]:raise ValueError("Update outside frozen schedule")
    root=Path(config["root"]).resolve()
    if not (root/"protocol.json").exists() or json.loads((root/"protocol.json").read_text())!=config:
        raise ValueError("Registered immutable protocol is missing or differs from launch config")
    train=config["training"]
    gpus=inherited.get("PHASE2_GPU_IDS","0,1,2,3,4,5,6,7")
    devices=gpus.split(",")
    if len(set(devices))!=len(devices) or any(not x.isdigit() for x in devices):
        raise ValueError("Explicit unique numeric GPU IDs required")
    world=len(devices)
    resume=config["parent_checkpoint"] if update==parent_update(config)+1 else str(root/f"checkpoints/global_step_{update-1}")
    resume=inherited.get("PHASE2_RESUME_CHECKPOINT",resume)
    prefix=training_run_id(config,update)
    env={**inherited,"PHASE2_ROOT":str(root),"PHASE2_ELASTIC_TRAINING":"1",
         "PHASE2_CPU_ADAM":"1" if world<=2 else "0", "PHASE1_MIN_CUDA_DEVICES":str(world),
         "PHASE1_RUN_ID":prefix,"PHASE1_UPDATE_TYPE":"real_rl","PHASE1_RL_SEED":str(train["sampler_seed_base"]+update),
         "PHASE1_TOTAL_EPOCHS":str(max(config["post_updates"])),"PHASE1_SAVE_FREQ":"1","PHASE1_TEST_FREQ":"0",
         "PHASE1_TRAIN_DATA_SIZE":str(train["games_per_update"]),"PHASE1_VAL_DATA_SIZE":str(train.get("validation_games",27)),
         "PHASE1_MODEL_PATH":train["reference"],"PHASE1_DATA_DIR":str(root/"datasets/verl-agent"),
         "PHASE1_RAY_TEMP_DIR":f"/home/wangyifan/ray-p2-{stable_hash(str(root))[:10]}-u{update}",
         "CUDA_VISIBLE_DEVICES":gpus,"OMP_NUM_THREADS":"1","TOKENIZERS_PARALLELISM":"false",
         "ALFWORLD_DATA":"/home/wangyifan/skill-RL/data/alfworld"}
    env.setdefault("PHASE1_MANIFEST_OUTPUT_DIR",str(root/f"launch_manifests/{prefix}-pid{os.getpid()}"))
    args=["bash",str(repo/"examples/grpo_trainer/run_alfworld_phase1_qwen35.sh"),"hf",
          f"trainer.n_gpus_per_node={world}",f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={'true' if world<=2 else 'false'}",
          f"data.max_prompt_length={train['max_prompt_tokens']}",f"data.max_response_length={train['max_response_tokens']}",
          f"actor_rollout_ref.actor.optim.lr={train['lr']}",f"actor_rollout_ref.actor.kl_loss_coef={train['kl_coef']}",
          f"env.rollout.n={train['rollouts_per_game']}",f"env.max_steps={train['max_environment_steps']}",
          f"env.alfworld.task_types={json.dumps(train['task_types'],separators=(',',':'))}",
          f"trainer.total_training_steps={update}","trainer.resume_mode=resume_path",f"trainer.resume_from_path={resume}",
          "trainer.del_local_ckpt_after_load=false","trainer.max_actor_ckpt_to_keep=100",
          f"trainer.default_local_dir={root}/checkpoints","+phase2.enabled=true",f"+phase2.root={root}",
          "+phase2.allow_expanded_rollout_batch=true"]
    return args,env


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--update",type=int,required=True)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    config=json.loads(a.protocol.read_text())
    args,env=launch_spec(config,a.update,repo,os.environ)
    if a.dry_run:
        print(json.dumps({"command":args,"experiment_env":{k:v for k,v in env.items() if k.startswith(("PHASE1_","PHASE2_"))}},indent=2))
        return
    subprocess.run([os.sys.executable,"-m","phase2.provenance","--root",config["root"]],cwd=repo,env=env,check=True)
    os.execvpe(args[0],args,env)


if __name__=="__main__":main()
