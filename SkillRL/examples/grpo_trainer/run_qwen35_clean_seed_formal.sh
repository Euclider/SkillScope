#!/usr/bin/env bash
set -euo pipefail

# Preregistered fixed-milestone run: one seed, 30 updates, 32 rollouts/update.
# Validation is explanatory only and is evaluated at the same 10-update
# milestones; it is never used to select checkpoints.
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <rl-seed: 101|202|303>" >&2
  exit 2
fi

FORMAL_SEED=$1
case "$FORMAL_SEED" in
  101|202|303) ;;
  *) echo "Unregistered RL seed: $FORMAL_SEED" >&2; exit 2 ;;
esac

FORMAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FORMAL_RUN_ID="qwen35-clean-formal-seed${FORMAL_SEED}-u30-20260831"
FORMAL_EXTRA_ARGS=()

if [[ -n ${PHASE1_RESUME_FROM_PATH:-} ]]; then
  if [[ ! -d "$PHASE1_RESUME_FROM_PATH" ]]; then
    echo "Resume checkpoint not found: $PHASE1_RESUME_FROM_PATH" >&2
    exit 2
  fi
  FORMAL_EXTRA_ARGS+=(
    trainer.resume_mode=resume_path
    trainer.resume_from_path="$PHASE1_RESUME_FROM_PATH"
  )
fi

PHASE1_MAX_ACTOR_CKPT_TO_KEEP=${PHASE1_MAX_ACTOR_CKPT_TO_KEEP:-2}
FORMAL_EXTRA_ARGS+=(
  trainer.max_actor_ckpt_to_keep="$PHASE1_MAX_ACTOR_CKPT_TO_KEEP"
)

export PHASE1_RUN_ID=${PHASE1_RUN_ID:-$FORMAL_RUN_ID}
export PHASE1_RL_SEED=$FORMAL_SEED
export PHASE1_TOTAL_EPOCHS=30
export PHASE1_SAVE_FREQ=${PHASE1_SAVE_FREQ:-1}
export PHASE1_TEST_FREQ=10
export PHASE1_TRAIN_DATA_SIZE=8
export PHASE1_VAL_DATA_SIZE=27
export PHASE1_RAY_TEMP_DIR="/home/wangyifan/ray-q35-clean-s${FORMAL_SEED}"
PHASE1_N_GPUS_PER_NODE=${PHASE1_N_GPUS_PER_NODE:-4}

WATCHER_PID=""
if [[ ${PHASE1_ENABLE_CHECKPOINT_WATCHER:-1} == 1 ]]; then
  WATCHER_EXTRA_ARGS=()
  if [[ -n ${PHASE1_EXTERNAL_RESUME_CHECKPOINT_TO_ROTATE:-} ]]; then
    WATCHER_EXTRA_ARGS+=(--external-resume-checkpoint "$PHASE1_EXTERNAL_RESUME_CHECKPOINT_TO_ROTATE")
  fi
  mkdir -p "$FORMAL_ROOT/artifacts/logs" "$FORMAL_ROOT/artifacts/model_only"
  python -u "$FORMAL_ROOT/phase1/watch_qwen35_checkpoints.py" \
    --checkpoint-root "$FORMAL_ROOT/artifacts/checkpoints/$PHASE1_RUN_ID" \
    --model-only-root "$FORMAL_ROOT/artifacts/model_only" \
    --repo-root "$FORMAL_ROOT" \
    --run-id "$PHASE1_RUN_ID" \
    --archive-prefix "qwen35-clean-formal-seed${FORMAL_SEED}-u30" \
    --milestones 10 20 30 \
    --keep-full "$PHASE1_MAX_ACTOR_CKPT_TO_KEEP" \
    "${WATCHER_EXTRA_ARGS[@]}" \
    >"$FORMAL_ROOT/artifacts/logs/${PHASE1_RUN_ID}-checkpoint-watcher.log" 2>&1 &
  WATCHER_PID=$!
fi

set +e
"$FORMAL_ROOT/examples/grpo_trainer/run_alfworld_phase1_qwen35.sh" hf \
  trainer.n_gpus_per_node="$PHASE1_N_GPUS_PER_NODE" \
  data.max_prompt_length=2048 \
  data.max_response_length=64 \
  "${FORMAL_EXTRA_ARGS[@]}"
TRAIN_STATUS=$?
set -e

if [[ -n "$WATCHER_PID" ]]; then
  if [[ $TRAIN_STATUS -eq 0 ]]; then
    wait "$WATCHER_PID"
  else
    kill "$WATCHER_PID" 2>/dev/null || true
    wait "$WATCHER_PID" 2>/dev/null || true
  fi
fi
exit "$TRAIN_STATUS"
