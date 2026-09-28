# SkillScope Phase1–2 portable source snapshot

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: reproducibility handoff
- Origin Date: 2026-09-28
- Verification Status: source/package checks and local CPU tests only; no non-ALFWorld end-to-end run has been validated
- Version Label: phase12-crossbench-source-2026-09-28

This package supplies the current Phase1–2 algorithm, training stack, environment
lock, tests, and the ALFWorld reference implementation. It does **not** claim
that a different benchmark can be launched by changing `--data-root` alone.
No model weights, benchmark datasets, checkpoints, trajectories, reports, API
credentials, or local caches are included.

## Install on a new server

The tested source environment was Linux x86-64, Python 3.12, CUDA 13.0,
8×RTX 5090, Qwen3.5-4B, vLLM 0.22.0 and PyTorch 2.11.0. The lock is a
record of that environment, **not** a portability guarantee for different
GPU architectures, CUDA drivers, or operating systems. Install into a fresh
environment, never into an existing experiment environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-vllm-phase12.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B -m pytest -p no:cacheprovider -q \
  tests/phase2/test_stable_direction.py \
  tests/skillnet_cohort/test_reward_variants.py \
  tests/skillnet_cohort/test_reward_recovery.py::test_native_cpu_optimizer_multiple_and_partial_minibatches \
  tests/skill_router/test_embedding_batch_router.py
```

Keep the exact installed package list, CUDA driver, GPU model, model/tokenizer
revision, and source manifest with each new experiment. The shorter
`requirements-vllm-phase12.in` lists direct requirements; the `.lock` is the
reproducibility reference. The separate
`requirements-skillnet-phase12.txt` is the older HF/text-only path and must
not be mixed into the vLLM environment.

This is a portable CPU smoke test, not a complete system acceptance test.
Some other reference-cohort tests compare historical artifact-bound source
hashes; they are expected to fail against a newer source snapshot or without
the original archived artifacts. Do not rewrite those historical expectations
to make a new benchmark appear validated.

The repository contains no base Qwen3.5-4B weights or 0.6B embedding router
weights. Obtain and hash the model/tokenizer/router snapshots separately.
For a new benchmark, provide its own task data and a **frozen initial skill
bank**; do not silently reuse ALFWorld SkillNet-37 as benchmark knowledge.
External LLM credentials are not needed for the local embedding router or
Phase1–2 readout. Never put credentials in a protocol file or Git history.

## Reusable method versus ALFWorld adapter

The most portable numerical core is `phase2/stable_direction.py`:
`token_signals(old_skill, new_skill, old_control, new_control, actions,
advantages)` accepts aligned full-vocabulary log-probability rows at the
**same recorded decision states**. The model, tokenizer, response prefix,
action-token IDs, loss mask, and skill version must match across conditions.
`skillnet_cohort/reward_variants.py` and Phase2 ranking/analysis modules
provide the registered aggregation and diagnostic variants. Do not interpret
a whole-trajectory GRPO advantage as a ground-truth local action utility.

The following reference path is ALFWorld-specific and needs a new adapter:

| Contract | Current ALFWorld implementation | New benchmark requirement |
| --- | --- | --- |
| Task inventory/splits | `skillnet_cohort/assets.py` expects 3,553 train, 140 Seen, 134 Unseen text games and six ALFWorld task types | Freeze benchmark-native task IDs, split membership, success definition, and data hashes |
| Interaction/archive | `agent_system/environments/env_package/alfworld/`, `phase1/archive.py`, `phase2/capture.py` | Record each observation, history, action, selected skill/version, task ID, reward, and actual training advantage/mask |
| Frozen skill/router | `memory_data/alfworld/skillnet37/`, `agent_system/memory/skillrl_embedding_router.py` | Supply a benchmark-appropriate initial bank and deterministic, frozen routing contract shared across compared policies |
| Matched control | `phase1/controls.py`, `skillnet_cohort/assets.py` | Remove only target guidance; preserve other context and check prompt/token controls with the new tokenizer/template |
| Utility gold | `phase1/first_invocation.py`, `phase2/evaluate.py` | Capture the first natural call of each skill per reference trajectory; restore/replay that state and run paired skill/control continuations under both policy endpoints with registered seeds |
| Training/evaluation runner | `skillnet_cohort/prepare.py`, `skillnet_cohort/run.py` | Replace ALFWorld paths, fixed counts, GPU-8 authorization, historical upload receipt, and seed/cohort assumptions; do not run this supervisor unchanged |

If the environment cannot resume from the **same decision anchor** under the
old/new policies and skill/control conditions, the existing marginal-utility
label is not identifiable by this protocol. A task-level alternative would
be a new estimand, not an equivalent replication.

## Required Phase1–2 sequence for each benchmark

1. Pre-register the benchmark split, new seed(s), frozen bank/router,
   checkpoints `U_t→U_{t+5}`, supported-skill rule, anchor sampling,
   continuation seeds, and primary readout before opening utility gold.
2. Train with the benchmark adapter and archive the **first rollout batch**
   sampled by `U_t` in each window, including actual advantages and masks.
   Keep the bank fixed within the window. Verify that endpoint `U_{t+5}` and
   start `U_t` score exactly these same decision contexts with and without
   the target skill. A new window uses the preceding endpoint as its start.
3. Compute magnitude and reward-directed scores without reading paired
   continuation outcomes. Log naturally invoked skills and support counts;
   unsupported skills abstain rather than receiving fabricated zero scores.
4. Independently collect first-natural-call anchors from `U_t` reference
   trajectories. For each anchor, use the same benchmark state and registered
   random seeds for old/new × skill/control continuations. Compute
   `M_t(s)`, `M_{t+5}(s)`, `ΔM(s)` and uncertainty at the game/task cluster
   level. Report direction (`−ΔM`) and magnitude (`|ΔM|`) prediction, all
   supported skills, coverage, actual evaluation cost, and negative/null
   controls; do not select the best readout after inspecting these labels.
5. Preserve source hashes, protocol amendments, failed attempts, and the
   exact metric/report code. Keep Phase3 editing/gate outcomes out of the
   Phase1–2 diagnostic validation.

The ALFWorld cohort files under `skillnet_cohort/` include one-off recovery
scripts and historical source-hash gates. They are evidence of the reference
execution, **not** a generic cross-benchmark launcher. Build and test a new
adapter for each benchmark before spending GPU time; passing CPU unit tests
does not validate a new environment's rollouts, state replay, vLLM parity,
or utility labels.
