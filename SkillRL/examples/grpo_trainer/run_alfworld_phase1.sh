#!/usr/bin/env bash
set -euo pipefail

ENGINE=${1:-vllm}
if [[ $# -gt 0 ]]; then shift; fi

PHASE1_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PHASE1_MODEL_PATH=${PHASE1_MODEL_PATH:-/home/wangyifan/model/Qwen2.5-0.5B-Instruct}
PHASE1_RUN_ID=${PHASE1_RUN_ID:?Set PHASE1_RUN_ID to an immutable experiment identifier}
PHASE1_RL_SEED=${PHASE1_RL_SEED:-101}
PHASE1_UPDATE_TYPE=${PHASE1_UPDATE_TYPE:-real_rl}
ALFWORLD_DATA=${ALFWORLD_DATA:?Set ALFWORLD_DATA to the downloaded ALFWorld data directory}

if [[ ${CONDA_DEFAULT_ENV:-} != "skill-RL" ]]; then
  echo "Activate the environment first: conda activate skill-RL" >&2
  exit 2
fi
if [[ ! -d "$PHASE1_MODEL_PATH" ]]; then
  echo "Model directory not found: $PHASE1_MODEL_PATH" >&2
  exit 2
fi

export ALFWORLD_DATA
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false

PHASE1_DATA_DIR="$PHASE1_REPO_ROOT/artifacts/datasets/verl-agent"
PHASE1_OUTPUT_DIR="$PHASE1_REPO_ROOT/artifacts"

python -m phase1.create_run_manifest \
  --run-id "$PHASE1_RUN_ID" \
  --stage rl_training \
  --model "$PHASE1_MODEL_PATH" \
  --skill-bank "$PHASE1_REPO_ROOT/phase1/config/frozen_alfworld_skills.json" \
  --protocol "$PHASE1_REPO_ROOT/phase1/config/phase1_protocol.json" \
  --rl-seed "$PHASE1_RL_SEED" \
  --update-type "$PHASE1_UPDATE_TYPE" \
  --output-dir "$PHASE1_OUTPUT_DIR" \
  --command "PHASE1_RL_SEED=$PHASE1_RL_SEED PHASE1_UPDATE_TYPE=$PHASE1_UPDATE_TYPE $0 $*"

python -m examples.data_preprocess.prepare \
  --mode text \
  --local_dir "$PHASE1_DATA_DIR" \
  --train_data_size 8 \
  --val_data_size 8

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$PHASE1_DATA_DIR/text/train.parquet" \
  data.val_files="$PHASE1_DATA_DIR/text/test.parquet" \
  data.train_batch_size=8 \
  data.val_batch_size=8 \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$PHASE1_MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name="$ENGINE" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=128 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed="$PHASE1_RL_SEED" \
  env.max_steps=30 \
  env.rollout.n=4 \
  env.resources_per_worker.num_cpus=0.5 \
  +env.use_skills_only_memory=True \
  +env.skills_only_memory.skills_json_path="$PHASE1_REPO_ROOT/phase1/config/frozen_alfworld_skills.json" \
  +env.skills_only_memory.retrieval_mode=template \
  +env.skills_only_memory.top_k=1 \
  +env.skills_only_memory.task_specific_top_k=1 \
  +env.skills_only_memory.enable_dynamic_update=False \
  +env.phase1_archive.enabled=True \
  +env.phase1_archive.run_id="$PHASE1_RUN_ID" \
  +env.phase1_archive.output_dir="$PHASE1_OUTPUT_DIR" \
  +phase1_control.type="$PHASE1_UPDATE_TYPE" \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.project_name=skill_rl_phase1 \
  trainer.experiment_name="$PHASE1_RUN_ID" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=5 \
  trainer.test_freq=5 \
  trainer.total_epochs=30 \
  trainer.default_local_dir="$PHASE1_OUTPUT_DIR/checkpoints/$PHASE1_RUN_ID" \
  trainer.val_before_train=False \
  "$@"
