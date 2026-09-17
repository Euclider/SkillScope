#!/usr/bin/env bash
set -euo pipefail
PHASE2_UPDATE=${1:?Provide cumulative post-update 31..36}
case "$PHASE2_UPDATE" in 31|32|33|34|35|36) ;; *) exit 2 ;; esac
PHASE2_REPO=/home/wangyifan/skill-RL/SkillRL
PHASE2_ROOT="$PHASE2_REPO/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1"
PHASE2_PREV=$((PHASE2_UPDATE - 1))
if [[ $PHASE2_UPDATE == 31 ]]; then
  PHASE2_RESUME="$PHASE2_REPO/artifacts/checkpoints/qwen35-clean-formal-seed303-u30-resume12-s2-20260903/global_step_30"
else
  PHASE2_RESUME="$PHASE2_ROOT/checkpoints/global_step_$PHASE2_PREV"
fi
PHASE2_RESUME=${PHASE2_RESUME_CHECKPOINT:-$PHASE2_RESUME}
PHASE2_GPU_IDS=${PHASE2_GPU_IDS:-0,1,2,3,4,5,6,7}
IFS=',' read -ra PHASE2_GPU_ARRAY <<< "$PHASE2_GPU_IDS"
PHASE2_GPU_COUNT=${#PHASE2_GPU_ARRAY[@]}
PHASE2_OPTIMIZER_OFFLOAD=false
export PHASE2_CPU_ADAM=0
if (( PHASE2_GPU_COUNT <= 2 )); then
  PHASE2_OPTIMIZER_OFFLOAD=true
  export PHASE2_CPU_ADAM=1
fi
source /home/wangyifan/miniconda3/etc/profile.d/conda.sh
conda activate skill-RL
cd "$PHASE2_REPO"
export ALFWORLD_DATA=/home/wangyifan/skill-RL/data/alfworld
export CUDA_VISIBLE_DEVICES="$PHASE2_GPU_IDS"
export PHASE2_ROOT
export PHASE2_ELASTIC_TRAINING=1
export PHASE1_MIN_CUDA_DEVICES="$PHASE2_GPU_COUNT"
export PHASE1_MANIFEST_OUTPUT_DIR=${PHASE1_MANIFEST_OUTPUT_DIR:-"$PHASE2_ROOT/launch_manifests/u$PHASE2_UPDATE-${EPOCHREALTIME/./}-pid$$"}
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PHASE1_RUN_ID="phase2-s303-fast-u$PHASE2_UPDATE"
export PHASE1_RL_SEED=$((30300 + PHASE2_UPDATE))
export PHASE1_TOTAL_EPOCHS=36
export PHASE1_SAVE_FREQ=1
export PHASE1_TEST_FREQ=0
export PHASE1_TRAIN_DATA_SIZE=8
export PHASE1_VAL_DATA_SIZE=27
export PHASE1_RAY_TEMP_DIR="/home/wangyifan/ray-phase2-s303-u$PHASE2_UPDATE"
python -m phase2.provenance --root "$PHASE2_ROOT"
exec bash examples/grpo_trainer/run_alfworld_phase1_qwen35.sh hf \
  trainer.n_gpus_per_node="$PHASE2_GPU_COUNT" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$PHASE2_OPTIMIZER_OFFLOAD" \
  data.max_prompt_length=2048 data.max_response_length=64 \
  trainer.total_training_steps="$PHASE2_UPDATE" \
  trainer.resume_mode=resume_path trainer.resume_from_path="$PHASE2_RESUME" \
  trainer.del_local_ckpt_after_load=false trainer.max_actor_ckpt_to_keep=100 \
  trainer.default_local_dir="$PHASE2_ROOT/checkpoints" \
  +phase2.enabled=true +phase2.root="$PHASE2_ROOT"
