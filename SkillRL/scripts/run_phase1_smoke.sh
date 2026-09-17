#!/usr/bin/env bash
set -euo pipefail

PHASE1_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PHASE1_RUN_ID=${PHASE1_RUN_ID:-smoke-0p5b-20260825}
PHASE1_MODEL_PATH=${PHASE1_MODEL_PATH:-/home/wangyifan/model/Qwen2.5-0.5B-Instruct}
PHASE1_GPU_IDS=${PHASE1_GPU_IDS:-0,1,2,3}
ALFWORLD_DATA=${ALFWORLD_DATA:?ALFWORLD_DATA must be set}

if [[ ${CONDA_DEFAULT_ENV:-} != "skill-RL" ]]; then
  echo "Activate the skill-RL Conda environment first" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$PHASE1_GPU_IDS"
if [[ ${#GPUS[@]} -lt 1 ]]; then
  echo "PHASE1_GPU_IDS must contain at least one GPU" >&2
  exit 2
fi

CONTEXTS=(pick_and_place look_at_obj_in_light clean heat cool pick_two)
SKILLS=(pic_001 loo_001 cle_001 hea_001 coo_001 pic_005)
OUTPUT_ROOT="$PHASE1_REPO_ROOT/artifacts/evaluations/$PHASE1_RUN_ID"
ROLLOUTS="$OUTPUT_ROOT/rollouts.jsonl"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

python -m phase1.create_run_manifest \
  --run-id "$PHASE1_RUN_ID" \
  --stage model_action_smoke \
  --model "$PHASE1_MODEL_PATH" \
  --skill-bank "$PHASE1_REPO_ROOT/phase1/config/frozen_alfworld_skills.json" \
  --protocol "$PHASE1_REPO_ROOT/phase1/config/phase1_protocol.json" \
  --rl-seed 0 \
  --update-type zero_update \
  --output-dir "$PHASE1_REPO_ROOT/artifacts" \
  --command "6 contexts x 2 games x eval-seed 11 x full_bank/minus_skill/no_skill; max_steps=30"

run_context() {
  local context=$1
  local skill=$2
  local gpu=$3
  CUDA_VISIBLE_DEVICES="$gpu" python -m phase1.eval_skill_margin \
    --checkpoint "$PHASE1_MODEL_PATH" \
    --skill-bank "$PHASE1_REPO_ROOT/phase1/config/frozen_alfworld_skills.json" \
    --skill-id "$skill" \
    --context-id "$context" \
    --game-ids-file "$PHASE1_REPO_ROOT/phase1/config/game_ids/development/$context.txt" \
    --max-games 2 \
    --eval-seeds 11 \
    --conditions full_bank minus_skill no_skill \
    --temperature 0.4 \
    --top-p 1.0 \
    --max-steps 30 \
    --max-new-tokens 512 \
    --environment-seed 1000 \
    --rl-seed 0 \
    --update-id 0 \
    --update-type zero_update \
    --split development \
    --run-id "$PHASE1_RUN_ID" \
    --output "$ROLLOUTS" \
    >"$LOG_DIR/$context.log" 2>&1
}

for ((start=0; start<${#CONTEXTS[@]}; start+=${#GPUS[@]})); do
  pids=()
  labels=()
  for ((slot=0; slot<${#GPUS[@]} && start+slot<${#CONTEXTS[@]}; slot++)); do
    index=$((start + slot))
    run_context "${CONTEXTS[$index]}" "${SKILLS[$index]}" "${GPUS[$slot]}" &
    pids+=("$!")
    labels+=("${CONTEXTS[$index]}")
  done
  for ((slot=0; slot<${#pids[@]}; slot++)); do
    if ! wait "${pids[$slot]}"; then
      echo "Context failed: ${labels[$slot]}; see $LOG_DIR/${labels[$slot]}.log" >&2
      exit 1
    fi
  done
done

python -m phase1.compute_smoke_metrics \
  --input "$ROLLOUTS" \
  --output-dir "$PHASE1_REPO_ROOT/artifacts/metrics/$PHASE1_RUN_ID" \
  --expected-records 36

for skill in "${SKILLS[@]}"; do
  python -m phase1.capture_probe_states \
    --trajectory-index "$ROLLOUTS" \
    --output "$PHASE1_REPO_ROOT/artifacts/probes/$PHASE1_RUN_ID/states.jsonl" \
    --skill-id "$skill" \
    --max-states-per-game 4
done
