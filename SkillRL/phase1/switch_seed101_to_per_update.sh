#!/usr/bin/env bash
set -euo pipefail

# One-shot operational supervisor for the already-running Seed 101 r3 job.
# It waits for update 5's tracker commit, freezes the process group before the
# next rollout can progress, verifies all FSDP shards, and resumes from update 5
# with per-update checkpointing plus automatic milestone archival.

REPO_ROOT=/home/wangyifan/skill-RL/SkillRL
SOURCE_RUN_ID=qwen35-clean-formal-seed101-u30-r3-20260901
RUN_ID=qwen35-clean-formal-seed101-u30-resume5-s1-20260901
CHECKPOINT_ROOT="$REPO_ROOT/artifacts/checkpoints/$SOURCE_RUN_ID"
RESUME_CHECKPOINT="$CHECKPOINT_ROOT/global_step_5"
TRAIN_LOG="$REPO_ROOT/artifacts/logs/$RUN_ID.log"
SUPERVISOR_LOG="$REPO_ROOT/artifacts/logs/$SOURCE_RUN_ID-per-update-switch.log"
OLD_PROCESS_GROUP=${1:?Usage: $0 OLD_PROCESS_GROUP_ID}

exec >>"$SUPERVISOR_LOG" 2>&1
echo "[$(date -Is)] armed; waiting for committed update 5 in $CHECKPOINT_ROOT"

while true; do
  if ! kill -0 -- "-$OLD_PROCESS_GROUP" 2>/dev/null; then
    echo "[$(date -Is)] original process group exited before update 5 was committed"
    exit 1
  fi
  if [[ -f "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt" ]] &&
     [[ $(<"$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt") == 5 ]]; then
    break
  fi
  sleep 5
done

# Freeze immediately after the trainer's atomic commit marker appears so that
# no appreciable update-6 rollout work can be produced before the handoff.
kill -STOP -- "-$OLD_PROCESS_GROUP"
echo "[$(date -Is)] update 5 commit marker observed; original process group frozen"

set +e
source /home/wangyifan/miniconda3/etc/profile.d/conda.sh
conda activate skill-RL
python - "$RESUME_CHECKPOINT" <<'PY'
from pathlib import Path
import runpy
import sys

module = runpy.run_path("/home/wangyifan/skill-RL/SkillRL/phase1/watch_qwen35_checkpoints.py")
metadata = module["validate_full_checkpoint"](Path(sys.argv[1]))
print(f"verified full checkpoint: {metadata}")
PY
VERIFY_STATUS=$?
set -e

if [[ $VERIFY_STATUS -ne 0 ]]; then
  echo "[$(date -Is)] checkpoint verification failed; allowing original training to continue"
  kill -CONT -- "-$OLD_PROCESS_GROUP" 2>/dev/null || true
  exit "$VERIFY_STATUS"
fi

echo "[$(date -Is)] stopping save_freq=5 process group after verified checkpoint"
kill -TERM -- "-$OLD_PROCESS_GROUP" 2>/dev/null || true
kill -CONT -- "-$OLD_PROCESS_GROUP" 2>/dev/null || true
for _ in $(seq 1 60); do
  if ! kill -0 -- "-$OLD_PROCESS_GROUP" 2>/dev/null; then
    break
  fi
  sleep 2
done
if kill -0 -- "-$OLD_PROCESS_GROUP" 2>/dev/null; then
  echo "[$(date -Is)] process group did not terminate within 120 seconds; forcing only that group"
  kill -KILL -- "-$OLD_PROCESS_GROUP" 2>/dev/null || true
fi

echo "[$(date -Is)] resuming Seed 101 from update 5 with save_freq=1 and keep=2"
setsid bash -lc "
  source /home/wangyifan/miniconda3/etc/profile.d/conda.sh
  conda activate skill-RL
  export ALFWORLD_DATA=/home/wangyifan/skill-RL/data/alfworld
  export CUDA_VISIBLE_DEVICES=4,5,6,7
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PHASE1_RUN_ID=$RUN_ID
  export PHASE1_RESUME_FROM_PATH=$RESUME_CHECKPOINT
  export PHASE1_SAVE_FREQ=1
  export PHASE1_MAX_ACTOR_CKPT_TO_KEEP=2
  export PHASE1_ENABLE_CHECKPOINT_WATCHER=1
  export PHASE1_EXTERNAL_RESUME_CHECKPOINT_TO_ROTATE=$RESUME_CHECKPOINT
  cd $REPO_ROOT
  exec bash examples/grpo_trainer/run_qwen35_clean_seed_formal.sh 101
" >>"$TRAIN_LOG" 2>&1 </dev/null &
NEW_SESSION_PID=$!
echo "$NEW_SESSION_PID" >"$REPO_ROOT/artifacts/manifests/$RUN_ID-per-update-session.pid"
echo "[$(date -Is)] new session pid=$NEW_SESSION_PID"

sleep 20
if ! kill -0 "$NEW_SESSION_PID" 2>/dev/null; then
  echo "[$(date -Is)] resumed session exited during startup"
  exit 1
fi
echo "[$(date -Is)] resumed session alive; handoff complete"
