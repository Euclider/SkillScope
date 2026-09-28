# Phase1–2 vLLM migration and parameter review — 2026-09-18

Origin Skill: experiment-agent
Origin Mode: plan
Origin Date: 2026-09-18
Verification Status: UNVERIFIED (GPU execution / ALFWorld throughput)
Version Label: phase12-vllm-v1

## Actual stop point

The user requested replacing HF **generation** with vLLM and receiving the actual
RL settings for confirmation before starting. Code migration, isolated dependency
installation, static vLLM engine-configuration validation, and offline tests are
complete. **No vLLM engine has loaded policy weights or generated tokens yet. No
new eight-GPU preflight or formal ALFWorld RL has started.** Model-class import can
initialize a CUDA context; it is not an inference benchmark.

The prior HF preflight failed on the second backward. Its logs/checkpoints remain
intact. The old 150-iteration task remains stopped at 0 updates. No historical
report, rollout, optimizer batch, checkpoint or router ledger was overwritten.
No API request, Git commit, rollback, cleanup or GitHub push occurred in this migration.

New preparation: [manifest](preparation-s404-8gpu/manifest.json).
Actual composed start-segment configuration: [parameter-review snapshot](resolved-training-review.json).
This is a **parameter-review snapshot, not an execution permit**; local router-call
authorization is still zero. The intended new run directory has not been created.

## Parameters awaiting user confirmation

| Item | Current effective setting |
|---|---|
| Policy / seed | Qwen3.5-4B, text-only full-parameter training; 404 |
| RL algorithm | GRPO; group-relative advantages standardized by standard deviation |
| Training horizon | **5 rollout/update iterations**, fixed U0→U5 window; not 50–100 |
| Sampling per iteration | 16 sampled training games × group 8 = 128 trajectories; nominal total 640 |
| Task pool | All six ALFWorld task types; sampled train pool, not exhaustive training traversal |
| Optimizer | Native GPU AdamW; lr 1e-6; betas (0.9, 0.999); weight decay .01 |
| Schedule | Constant lr, no warmup; one PPO epoch per batch |
| PPO minibatch | 128 flattened state/action rows globally, not 128 complete trajectories |
| Training microbatch | 1 row/GPU; 8 GPUs ⇒ 16 accumulation forwards for a full minibatch |
| PPO / regularization | clip .2 (low/high .2, dual clip 3); grad clip 1; token-mean loss |
| KL / entropy | low_var_kl loss coefficient .01; entropy coefficient .001; no reward KL |
| Reward | success 10, failure 0; invalid-action penalty coefficient .1 |
| Training decode | temperature 1.0, top-p 1.0; one action response per state; thinking disabled |
| Evaluation decode | temperature .4, top-p 1.0; request-specific fixed seeds |
| Context / horizon | prompt ≤4096, response ≤512, ≤50 environment steps, history 2; overflow raises |
| Training parallelism | verl native FSDP1, transformer-layer wrapping, sequence parallel 1 |
| Memory policy | BF16 mixed precision; native FP32 training/master state; gradient checkpointing; optimizer state stored on CPU between updates; CPU shard initialization |
| Rollout inference | vLLM 0.22.0; 8 TP=1 replicas; ≤16 concurrent sequences per GPU; prefill token budget 8192; max context 4608 |
| Engine memory / kernels | gpu_memory_utilization .45; native Qwen3.5 kernels; chunked prefill enabled; prefix caching off; eager mode (CUDA graphs/compile not yet enabled) |
| Monitor / checkpoint | Every 5 iterations; monitor 64 episodes; native model/optimizer/RNG checkpoint preserved |
| Full performance | U0 and U5: all 140 seen + 134 unseen games, reported separately |
| Skill bank / router | Frozen full SkillNet-37; frozen Qwen3-Embedding-0.6B; per-state top-1; CPU FP32 batch/dedup with 8 threads; no external router API |
| Phase2 | unseen only; max12 naturally supported skills × max12 anchors; 1 evidence +2 gold seeds; fixed five-iteration readout window |
| Recording | Step trajectories, skill IDs, rewards/advantages, actual training batches/optimizer updates, exact old/new FP32 full-vocabulary readout preserved |

“Iteration” is **not** a single optimizer.step: trajectories are flattened into
state/action rows, so a single iteration may contain several optimizer steps.
The quoted 20–30/50–100-update, group4/8 timing examples did not authorize a new
horizon or group size; neither the 8×8 proposal nor anchor12→8 was applied.

New documentation uses the actual scope above, not the label “预算受限验证”, per the
user's request. This does not establish 150-iteration convergence or multi-seed
robustness. Old frozen documents retain their historical wording.

The old scope profile's generation microbatch=2 was an HF memory bound. The
separately hashed inference profile overrides that implementation-level setting
to 16 for vLLM (reflected in the resolved configuration); GRPO remains 16×8.
FSDP2 and a new RL framework were **not** introduced: only generation is migrated.
Training and exact readout still use native differentiable model forwards, not
vLLM's sampled-token log probabilities.

## Implementation and compatibility

- Independent environment: `/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918`.
  Old environment untouched. Python3.12, torch2.11.0+cu130, transformers5.10.4,
  vLLM0.22.0. Exact resolved packages: `requirements-vllm-phase12.lock`.
- `configs/phase12_vllm_v1.json` is a new hash-bound inference identity, propagated
  to rollout, performance, anchor collection and Phase2 O/P/N. Old manifests keep
  historical behavior; new ones never fall back to HF on error.
- vLLM exposes a native Qwen3.5 text implementation but registers the multimodal
  wrapper by default. `vllm_qwen35.py` registers that native text implementation,
  borrowing its upstream hybrid-cache descriptors and mapping original VL/text
  checkpoint key names. It does not implement an HF-generate backend.
- Live actor weights are streamed from FSDP sharded state via public
  `LLM.apply_model` in the external-launcher, in-process engine. Missing destination
  parameters fail closed. No private `model_executor.driver_worker` traversal.
- New actor updates/restores invalidate the weight version. Repeated environment
  steps keep the engine awake and reuse weights; native actor/reference forward,
  update and checkpoint entry points suspend colocated engines first.
- Training rollout is batch generation. Current offline evaluators run **eight GPU
  shards but sequential episodes within each shard**. The shared policy exposes
  generate_batch, but evaluators do not yet interleave multiple environments within
  each shard. Do not claim such interleaving or maximum GPU utilization.
- Static engine-config validation initially found vLLM's dummy-config probe;
  the override now preserves that probe and handles the actual text config after
  loading. Final validation resolved SkillScopeQwen35Text, is_hybrid=True,
  model_type=qwen3_5_text, max length4608. No engine/weights/rollouts were started.

Official API/code checked: [public LLM.apply_model](https://github.com/vllm-project/vllm/blob/v0.22.0/vllm/entrypoints/llm.py),
[external-launcher executor](https://github.com/vllm-project/vllm/blob/v0.22.0/vllm/v1/executor/uniproc_executor.py),
[native Qwen3.5 implementation](https://github.com/vllm-project/vllm/blob/v0.22.0/vllm/model_executor/models/qwen3_5.py),
[sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/).
These support implementation choices, **not** a measured speedup or 15–30h guarantee.

## Verification

- `uv pip check`: all269 installed packages compatible.
- Initial relevant regression:172 PASS,76.56s (`offline-regression-initial.xml`).
- Full regression:445 PASS,83.29s (`offline-tests-final.xml`), including Phase3,
  router/bank/settings/cohort/Phase1 and selected Phase2 protocol/ranking tests.
- Ten adapter tests passed initially and again after the config-probe fix;
  latest10 PASS,8.92s (`adapter-tests-final.xml`). These are included in the445
  test scope, not455 independent tests.
- Offline tests use CPU/fakes for real weight sync, sleep/wake and generation.
  They do not prove GPU equivalence, native checkpoint/sync correctness on eight
  ranks, continuous-update memory fit, full-vocabulary capture capacity or runtime.
- 07:15 UTC snapshot: eight GPUs idle (2MiB each), ~711GiB free disk. Snapshot only.

## Next gate — NOT EXECUTED

First receive the user's parameter confirmation. Under the experiment-agent audit
workflow, a new GPU run also needs an explicitly reviewed command; a failed HF
experiment is not automatically retried. Proposed **single synthetic vLLM engineering
preflight**, no ALFWorld, no paid API, at most30min, new output directory:

```bash
cd /mnt/workspace/users/wangyifan/skill-RL/SkillRL
timeout --signal=TERM --kill-after=60s 30m env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
  /mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m skillnet_cohort.gpu_preflight \
  --model /mnt/workspace/users/wangyifan/model/Qwen3.5-4B \
  --output /mnt/workspace/users/wangyifan/skill-RL/vllm-gpu-preflight-20260918-v1 \
  --embedding-router-model /mnt/workspace/users/wangyifan/model/Qwen3-Embedding-0.6B \
  --embedding-router-backend skillrl_embedding_state_batch \
  --inference-profile configs/phase12_vllm_v1.json \
  --rollout-micro-batch-size 16 --generation-cap 512 \
  --optimizer-offload --total-training-steps 5 --execute
```

Before launch, check exact target is unused, GPU ownership/free space, archive the
command/source hashes, and capture stdout/stderr into a new log. No automatic retry
after failure. Leave synthetic checkpoint evidence intact.

Required outcomes:8/8 forward/backward, two consecutive native updates, native
weight/optimizer/RNG restore, 16×(4096+≤512) vLLM generation per rank, resync after
update and no repeated per-state transfer. This still does not verify a complete
128-row optimizer minibatch, Ray orchestration, real ALFWorld full-prompt coverage,
exact full-vocabulary archive I/O or complete wallclock budget. Those remain part
of measured admission before any formal RL authorization.
