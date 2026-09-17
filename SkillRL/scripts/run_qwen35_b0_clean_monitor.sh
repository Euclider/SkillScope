#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ID=${RUN_ID:-qwen35-b0-clean-valid-seen-20260831}
MODEL_PATH=${MODEL_PATH:-/home/wangyifan/model/Qwen3.5-4B}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
ALFWORLD_DATA=${ALFWORLD_DATA:-/home/wangyifan/skill-RL/data/alfworld}
OUTPUT_ROOT="$REPO_ROOT/artifacts/evaluations/$RUN_ID"
LOG_ROOT="$OUTPUT_ROOT/logs"
GAME_LIST="$REPO_ROOT/phase1/config/game_ids/qwen35_clean/valid_seen/clean.txt"
PROTOCOL="$REPO_ROOT/phase1/config/qwen35_clean_seed_generalization_protocol.json"
SKILL_BANK="$REPO_ROOT/memory_data/alfworld/claude_style_skills.json"

if [[ ${CONDA_DEFAULT_ENV:-} != "skill-RL" ]]; then
  echo "Activate the skill-RL environment first" >&2
  exit 2
fi
export ALFWORLD_DATA
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LOG_ROOT"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
SHARD_COUNT=${#GPUS[@]}

python -m phase1.create_run_manifest \
  --run-id "$RUN_ID" \
  --stage qwen35_b0_clean_valid_seen_monitor \
  --model "$MODEL_PATH" \
  --skill-bank "$SKILL_BANK" \
  --protocol "$PROTOCOL" \
  --rl-seed 0 \
  --update-type zero_update \
  --output-dir "$REPO_ROOT/artifacts" \
  --command "27 valid_seen clean games x eval seeds 1101,2202; full bank; 8 deterministic shards"

pids=()
for shard in "${!GPUS[@]}"; do
  output="$OUTPUT_ROOT/shard-${shard}.jsonl"
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" python -m phase1.eval_skill_margin \
    --checkpoint "$MODEL_PATH" \
    --skill-bank "$SKILL_BANK" \
    --skill-id cle_006 \
    --context-id clean \
    --game-ids-file "$GAME_LIST" \
    --game-shard-index "$shard" \
    --game-shard-count "$SHARD_COUNT" \
    --eval-seeds 1101 2202 \
    --conditions full_bank \
    --temperature 0.4 \
    --top-p 1.0 \
    --max-steps 30 \
    --max-new-tokens 32 \
    --history-length 2 \
    --step-routing \
    --router-general-top-k 12 \
    --environment-seed $((31000 + shard * 100)) \
    --rl-seed 0 \
    --update-id B0 \
    --update-type zero_update \
    --split valid_seen \
    --run-id "$RUN_ID" \
    --output "$output" \
    >"$LOG_ROOT/shard-${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ $failed -ne 0 ]]; then
  echo "At least one B0 monitor shard failed; inspect $LOG_ROOT" >&2
  exit 1
fi

python -m phase1.summarize_b0_clean \
  --inputs "$OUTPUT_ROOT"/shard-*.jsonl \
  --combined-index "$OUTPUT_ROOT/rollouts.jsonl" \
  --output "$OUTPUT_ROOT/summary.json"
