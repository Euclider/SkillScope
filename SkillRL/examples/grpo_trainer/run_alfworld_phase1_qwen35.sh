#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5 compatibility launcher for the text-only ALFWorld experiment.
# HF rollout is the safe default: it generates from the live FSDP policy and
# therefore does not depend on the legacy embedded-vLLM weight-sync internals.
ENGINE=${1:-hf}
if [[ $# -gt 0 ]]; then shift; fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PHASE1_MODEL_PATH=${PHASE1_MODEL_PATH:-/home/wangyifan/model/Qwen3.5-4B}

exec "$SCRIPT_DIR/run_alfworld_phase1_step_router.sh" "$ENGINE" \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  env.alfworld.action_only_prompt=true \
  env.alfworld.task_types=[3] \
  +actor_rollout_ref.model.load_text_only=true \
  +actor_rollout_ref.model.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  "$@"
