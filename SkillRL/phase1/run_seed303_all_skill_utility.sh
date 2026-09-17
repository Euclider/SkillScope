#!/usr/bin/env bash
set -euo pipefail

# Seed-303 extension of the frozen B0-anchor evaluation. Scientific inputs are
# byte-for-byte reused from the completed seeds101/202 run. Parallel placement
# affects throughput only: each branch resets its RNG from the frozen anchor.

REPO_ROOT=/home/wangyifan/skill-RL/SkillRL
RUN_ID=qwen35-clean-all-skill-seed303-v1
OLD_RUN=qwen35-clean-all-skill-seeds101-202-v1
THREE_SEED_RUN=qwen35-clean-all-skill-seeds101-202-303-v1
ANCHORS="$REPO_ROOT/artifacts/anchors/$OLD_RUN/selected"
PLACEBOS="$REPO_ROOT/artifacts/controls/$OLD_RUN"
SKILL_BANK="$REPO_ROOT/memory_data/alfworld/claude_style_skills.json"
EVAL_DIR="$REPO_ROOT/artifacts/evaluations/$RUN_ID"
METRICS_DIR="$REPO_ROOT/artifacts/metrics/$THREE_SEED_RUN"
OLD_EVAL_DIR="$REPO_ROOT/artifacts/evaluations/$OLD_RUN"
LOG_DIR="$EVAL_DIR/logs"
SUPERVISOR_LOG="$LOG_DIR/supervisor.log"

if [[ ${CONDA_DEFAULT_ENV:-} != skill-RL ]]; then
  echo "Activate conda environment skill-RL before launching" >&2
  exit 2
fi
if [[ -f "$REPO_ROOT/artifacts/manifests/$RUN_ID.json" ]]; then
  echo "Completed immutable run already exists: $RUN_ID" >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$METRICS_DIR"
exec > >(tee -a "$SUPERVISOR_LOG") 2>&1

export ALFWORLD_DATA=/home/wangyifan/skill-RL/data/alfworld
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

skills=(cle_003 cle_004 cle_006 gen_002)
updates=(10 20 30)
pids=()
labels=()
job_index=0

cleanup() {
  local status=$?
  if (( status != 0 )); then
    echo "[$(date -Is)] stopping remaining evaluation workers after failure"
    for pid in "${pids[@]:-}"; do
      kill "$pid" 2>/dev/null || true
    done
  fi
}
trap cleanup EXIT INT TERM

echo "[$(date -Is)] starting Seed303 frozen-anchor evaluation"
for update in "${updates[@]}"; do
  checkpoint_id="seed303-u${update}"
  ordinal=$((update / 10))
  checkpoint="$REPO_ROOT/artifacts/model_only/qwen35-clean-formal-seed303-u30-c${ordinal}-update${update}"
  for skill in "${skills[@]}"; do
    gpu=$((job_index % 8))
    output="$EVAL_DIR/${checkpoint_id}-${skill}.jsonl"
    log="$LOG_DIR/${checkpoint_id}-${skill}.log"
    echo "[$(date -Is)] launch checkpoint=$checkpoint_id skill=$skill gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m phase1.eval_all_first_invocation_utility \
      --checkpoint "$checkpoint" \
      --checkpoint-id "$checkpoint_id" \
      --anchors-dir "$ANCHORS" \
      --placebo-dir "$PLACEBOS" \
      --skill-bank "$SKILL_BANK" \
      --skills "$skill" \
      --max-anchors-per-skill 50 \
      --arms original placebo null \
      --temperature 0.4 \
      --top-p 1.0 \
      --max-new-tokens 64 \
      --history-length 2 \
      --router-general-top-k 12 \
      --run-id "$RUN_ID" \
      --rl-seed 303 \
      --global-update "$update" \
      --output "$output" >"$log" 2>&1 &
    pids+=("$!")
    labels+=("$checkpoint_id/$skill/gpu$gpu")
    job_index=$((job_index + 1))
  done
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[$(date -Is)] complete ${labels[$index]}"
  else
    echo "[$(date -Is)] FAILED ${labels[$index]}"
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 3
fi

python - "$EVAL_DIR" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = {(f"seed303-u{update}", skill) for update in (10, 20, 30) for skill in ("cle_003", "cle_004", "cle_006", "gen_002")}
seen = set()
for path in sorted(root.glob("*.jsonl")):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 150:
        raise RuntimeError(f"expected 150 records in {path}, got {len(rows)}")
    keys = {(row["checkpoint_id"], row["skill_id"]) for row in rows}
    if len(keys) != 1:
        raise RuntimeError(f"mixed identity in {path}: {keys}")
    seen.update(keys)
if seen != expected:
    raise RuntimeError(f"evaluation identity mismatch: missing={sorted(expected-seen)} extra={sorted(seen-expected)}")
print("Seed303 index pre-audit passed: 12 files, 1800 records")
PY

python -m phase1.compute_all_first_invocation_metrics \
  --input "$OLD_EVAL_DIR"/*.jsonl "$EVAL_DIR"/*.jsonl \
  --coverage "$REPO_ROOT/artifacts/anchors/$OLD_RUN/coverage.json" \
  --base-checkpoint-id b0 \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 20260904 \
  --output-dir "$METRICS_DIR"

echo "[$(date -Is)] Seed303 evaluation and three-seed metrics complete"
