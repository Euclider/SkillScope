#!/usr/bin/env bash
set -euo pipefail

# One-shot handoff supervisor.  Seed 202 is allowed to start only after Seed
# 101 update 30 and its model-only C3 archive are independently verified and
# the four target GPUs are free.  No performance or Skill-utility result is
# consulted.

REPO_ROOT=/home/wangyifan/skill-RL/SkillRL
SEED101_RUN=qwen35-clean-formal-seed101-u30-resume5-s1-20260901
SEED101_ROOT="$REPO_ROOT/artifacts/checkpoints/$SEED101_RUN"
SEED101_PID_FILE="$REPO_ROOT/artifacts/manifests/$SEED101_RUN-per-update-session.pid"
SEED101_C3="$REPO_ROOT/artifacts/model_only/qwen35-clean-formal-seed101-u30-c3-update30"

SEED202_SOURCE_RUN=qwen35-clean-formal-seed202-u30-20260831
SEED202_SOURCE="$REPO_ROOT/artifacts/checkpoints/$SEED202_SOURCE_RUN/global_step_10"
SEED202_RUN=qwen35-clean-formal-seed202-u30-resume10-s1-20260902
SEED202_LOG="$REPO_ROOT/artifacts/logs/$SEED202_RUN.log"
SUPERVISOR_LOG="$REPO_ROOT/artifacts/logs/seed101-to-seed202-handoff-20260902.log"
TARGET_GPUS=4,5,6,7

exec >>"$SUPERVISOR_LOG" 2>&1
echo "[$(date -Is)] handoff supervisor armed"

source /home/wangyifan/miniconda3/etc/profile.d/conda.sh
conda activate skill-RL

validate_full() {
  python - "$1" <<'PY'
from pathlib import Path
import runpy
import sys

module = runpy.run_path("/home/wangyifan/skill-RL/SkillRL/phase1/watch_qwen35_checkpoints.py")
print(module["validate_full_checkpoint"](Path(sys.argv[1])))
PY
}

validate_model_only() {
  python - "$1" <<'PY'
from pathlib import Path
import runpy
import sys

module = runpy.run_path("/home/wangyifan/skill-RL/SkillRL/phase1/watch_qwen35_checkpoints.py")
print(module["validate_model_only"](Path(sys.argv[1])))
PY
}

seed101_alive() {
  [[ -f "$SEED101_PID_FILE" ]] || return 1
  local pid state
  pid=$(<"$SEED101_PID_FILE")
  state=$(ps -o stat= -p "$pid" 2>/dev/null || true)
  [[ -n "$state" && "$state" != Z* ]]
}

while true; do
  tracker=""
  [[ -f "$SEED101_ROOT/latest_checkpointed_iteration.txt" ]] && tracker=$(<"$SEED101_ROOT/latest_checkpointed_iteration.txt")
  if [[ "$tracker" == 30 && -d "$SEED101_ROOT/global_step_30" && -d "$SEED101_C3" ]]; then
    if validate_full "$SEED101_ROOT/global_step_30" && validate_model_only "$SEED101_C3"; then
      echo "[$(date -Is)] Seed 101 update 30 and C3 verified"
      break
    fi
  fi
  if ! seed101_alive; then
    echo "[$(date -Is)] ERROR: Seed 101 exited before update 30 plus C3 verification; Seed 202 will not start"
    exit 2
  fi
  sleep 20
done

# The formal launcher waits for its checkpoint watcher, so a clean session exit
# confirms that training, model conversion, and final watcher validation ended.
while seed101_alive; do
  sleep 5
done
echo "[$(date -Is)] Seed 101 session exited cleanly after verified C3"

validate_full "$SEED202_SOURCE"
validate_model_only "$REPO_ROOT/artifacts/model_only/qwen35-clean-formal-seed202-u30-c1-update10"
echo "[$(date -Is)] Seed 202 update 10 recovery source and C1 model-only archive verified"

# Seed 101 is complete and its three analysis weights are verified.  Remove its
# two rolling optimizer checkpoints before the next seed to recover about 100GB.
python - "$SEED101_ROOT" "$SEED101_C3" "$REPO_ROOT/artifacts/manifests/$SEED101_RUN-completed-full-checkpoint-cleanup.json" <<'PY'
from datetime import datetime
import json
from pathlib import Path
import runpy
import shutil
import sys

root, c3, manifest = map(Path, sys.argv[1:])
expected = Path("/home/wangyifan/skill-RL/SkillRL/artifacts/checkpoints/qwen35-clean-formal-seed101-u30-resume5-s1-20260901")
if root.resolve() != expected.resolve():
    raise RuntimeError(f"refusing unexpected cleanup target: {root}")
module = runpy.run_path("/home/wangyifan/skill-RL/SkillRL/phase1/watch_qwen35_checkpoints.py")
c3_metadata = module["validate_model_only"](c3)
checkpoints = []
for path in sorted(root.glob("global_step_*")):
    try:
        metadata = module["validate_full_checkpoint"](path)
    except RuntimeError:
        continue
    checkpoints.append({"path": str(path), **metadata})
if not any(item["path"].endswith("global_step_30") for item in checkpoints):
    raise RuntimeError("validated global_step_30 not found before cleanup")
payload = {
    "schema_version": "phase1.completed_seed_checkpoint_cleanup.v1",
    "run_id": "qwen35-clean-formal-seed101-u30-resume5-s1-20260901",
    "reason": "Seed completed and C1/C2/C3 model-only analysis artifacts verified; release rolling optimizer checkpoints before Seed 202.",
    "removed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "removed_full_checkpoints": checkpoints,
    "retained_c3": str(c3),
    "retained_c3_metadata": c3_metadata,
}
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
shutil.rmtree(root)
print(json.dumps(payload, sort_keys=True))
PY
echo "[$(date -Is)] released completed Seed 101 full checkpoints"

while true; do
  free_count=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '
    $1 ~ /^[[:space:]]*[4567][[:space:]]*$/ && $2 + 0 < 5000 {count++}
    END {print count + 0}
  ')
  free_kb=$(df --output=avail -k /home/wangyifan | tail -n 1 | tr -d ' ')
  if [[ "$free_count" == 4 && "$free_kb" -ge 160000000 ]]; then
    break
  fi
  echo "[$(date -Is)] waiting: free target GPUs=$free_count/4, disk_available_kb=$free_kb"
  sleep 20
done

echo "[$(date -Is)] starting Seed 202 from update 10 on GPUs $TARGET_GPUS"
setsid bash -lc "
  source /home/wangyifan/miniconda3/etc/profile.d/conda.sh
  conda activate skill-RL
  export ALFWORLD_DATA=/home/wangyifan/skill-RL/data/alfworld
  export CUDA_VISIBLE_DEVICES=$TARGET_GPUS
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PHASE1_RUN_ID=$SEED202_RUN
  export PHASE1_RESUME_FROM_PATH=$SEED202_SOURCE
  export PHASE1_SAVE_FREQ=1
  export PHASE1_MAX_ACTOR_CKPT_TO_KEEP=2
  export PHASE1_ENABLE_CHECKPOINT_WATCHER=1
  export PHASE1_EXTERNAL_RESUME_CHECKPOINT_TO_ROTATE=$SEED202_SOURCE
  cd $REPO_ROOT
  exec bash examples/grpo_trainer/run_qwen35_clean_seed_formal.sh 202
" >>"$SEED202_LOG" 2>&1 </dev/null &
SEED202_SESSION_PID=$!
echo "$SEED202_SESSION_PID" >"$REPO_ROOT/artifacts/manifests/$SEED202_RUN-session.pid"
echo "[$(date -Is)] Seed 202 session pid=$SEED202_SESSION_PID"

sleep 30
if ! kill -0 "$SEED202_SESSION_PID" 2>/dev/null; then
  echo "[$(date -Is)] ERROR: Seed 202 exited during startup"
  exit 3
fi
echo "[$(date -Is)] Seed 202 startup alive; handoff complete"
