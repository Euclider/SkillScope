# Phase-I minimal validation

This directory implements two ordered validations:

1. a fixed-state action probe comparing pre/post-update policies under
   `full_bank`, `minus_skill`, and `no_skill`;
2. held-out episode-level marginal utility, where the primary contrast is
   `FULL_BANK - MINUS_SKILL`.

No experiment should start until this command passes:

```bash
conda activate skill-RL
python -m phase1.preflight
```

## Frozen inputs

- Skill Bank: `phase1/config/frozen_alfworld_skills.json`
- Locked protocol: `phase1/config/phase1_protocol.json`
- Model hashes: `phase1/config/model_manifest.json`
- Game manifests: `phase1/config/game_ids/manifest.json`
- Smoke model: `/home/wangyifan/model/Qwen2.5-0.5B-Instruct`
- Fallback model: `/home/wangyifan/model/Qwen2.5-1.5B-Instruct`
- ALFWorld data: `/home/wangyifan/skill-RL/data/alfworld`

The six contexts are `pick_and_place`, `look_at_obj_in_light`, `clean`,
`heat`, `cool`, and `pick_two`. The frozen bank contains one general skill
and exactly one task-specific skill for each context.

`phase1/config/phase1_protocol.json` is the source of truth for seeds,
checkpoint cadence, conditions, minimum support, capability gates, and Phase-I
go/no-go thresholds. Any deviation must use a new protocol version and run ID.

## Training launcher

```bash
export PHASE1_RUN_ID=smoke-seed-101
export PHASE1_RL_SEED=101
bash examples/grpo_trainer/run_alfworld_phase1.sh vllm
```

The launcher creates an immutable run manifest, saves checkpoints every five
updates, and writes a full JSON trajectory per episode. It requires two GPUs.
For the shuffled-reward control, pass:

```bash
PHASE1_UPDATE_TYPE=shuffled_reward \
bash examples/grpo_trainer/run_alfworld_phase1.sh vllm \
  +phase1_control.seed=1000 \
  +phase1_control.mapping_output=artifacts/controls/shuffled_reward_mapping.jsonl
```

`phase1/create_random_parameter_control.py` builds the matched-norm random
parameter checkpoint. Re-evaluating an unchanged checkpoint with
`--update-type zero_update` supplies the zero-update control.

## Matched evaluator

Example for one eligible pair:

```bash
python -m phase1.eval_skill_margin \
  --checkpoint /path/to/checkpoint \
  --skill-bank phase1/config/frozen_alfworld_skills.json \
  --skill-id cle_001 \
  --context-id clean \
  --game-ids-file phase1/config/game_ids/valid_seen/clean.txt \
  --eval-seeds 11 22 33 \
  --conditions full_bank minus_skill no_skill \
  --run-id eval-clean-checkpoint-0 \
  --output artifacts/evaluations/rollouts.jsonl
```

Each completed unique key is skipped on rerun. Full trajectories include
every prompt, observation, admissible action set, raw completion, projected
action, reward, and per-step retrieved/injected/disabled Skill IDs.
Training additionally writes one `training_step.v1` record per RL step with
per-trajectory reward/return/advantage summaries and optimizer metrics.

Measure the exact parameter delta between saved checkpoints with:

```bash
python -m phase1.measure_checkpoint_delta \
  --pre-checkpoint /path/to/checkpoint-0 \
  --post-checkpoint /path/to/checkpoint-5 \
  --output artifacts/checkpoint_deltas/0-to-5.json
```

## Fixed-state probe

First capture states from archived full-bank trajectories, then compare
checkpoints with `phase1.action_probe`. The probe reports greedy action flips
and a constrained distribution over the state's admissible actions. Use
`phase1.compute_action_probe_metrics` for checkpoint transition summaries.

## Statistics

```bash
python -m phase1.compute_phase1_metrics \
  --input artifacts/evaluations/rollouts.jsonl \
  --output-dir artifacts/metrics
```

Bootstrap resampling is clustered by game. Outputs include margins,
checkpoint transitions, strict CI-based flip labels, abstentions, cross-seed
stability, and comparisons with available negative controls.

## Step-routed real-RL validation (v2)

The earlier 7-Skill protocol and its smoke artifacts remain frozen. Real-RL
validation uses the full upstream ALFWorld bank and a deterministic frozen
router that selects exactly one Skill before every policy action, including
the initial action. The router sees only the task, current observation,
admissible actions, prior action/observation history, and step index. It never
sees reward or success and is not updated with the policy.

- Protocol: `phase1/config/phase1_step_routing_protocol.json`
- Skill Bank: `memory_data/alfworld/claude_style_skills.json`
- Router: `agent_system/memory/step_skill_router.py`
- Launcher: `examples/grpo_trainer/run_alfworld_phase1_step_router.sh`

Run the five-epoch, one-seed pilot only after reviewing its immutable run ID:

```bash
conda activate skill-RL
export ALFWORLD_DATA=/home/wangyifan/skill-RL/data/alfworld
export PHASE1_RUN_ID=phase1-step-router-pilot-seed-101
bash examples/grpo_trainer/run_alfworld_phase1_step_router.sh vllm
```

The launcher defaults to `/home/wangyifan/model/Qwen2.5-1.5B-Instruct`, two
GPUs, seed 101, five epochs, and checkpoint/validation cadence 1. It runs the
v2 preflight before creating the run manifest. Override the pilot length only
with a new run ID, for example `PHASE1_TOTAL_EPOCHS=30` for the later formal
run.

Every v2 trajectory records the candidate Skill IDs, selected/injected Skill,
router version, routing scores and observable state flags at every step. Its
top level records distinct selected Skills and per-Skill invocation counts.
For matched evaluation add `--step-routing` and use the full-bank path:

```bash
python -m phase1.eval_skill_margin \
  --step-routing \
  --checkpoint /path/to/checkpoint \
  --skill-bank memory_data/alfworld/claude_style_skills.json \
  --skill-id cle_003 \
  --context-id clean \
  --game-ids-file phase1/config/game_ids/valid_seen/clean.txt \
  --eval-seeds 11 22 33 \
  --conditions full_bank minus_skill no_skill \
  --run-id eval-clean-cle003-checkpoint-1 \
  --output artifacts/evaluations/step-router-rollouts.jsonl
```

Only states/trajectories where the target was selected in the full-bank arm
are eligible for target-Skill attribution. `minus_skill` removes that Skill
from the candidate set and deterministically reroutes at the identical state;
`no_skill` injects no Skill.

After the pilot, archive its routing statistics and automatic invariant gate:

```bash
python -m phase1.summarize_step_routing \
  --trajectory-index artifacts/trajectories/index.jsonl \
  --run-id phase1-step-router-pilot-seed-101 \
  --output artifacts/metrics/phase1-step-router-pilot-seed-101/routing-summary.json
```
