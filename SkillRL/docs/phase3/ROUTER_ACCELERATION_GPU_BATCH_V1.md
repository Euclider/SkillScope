# Phase3 FP32 GPU-batched router: execution record

Status: **v4 running, not a Phase3 result** (2026-09-25 UTC). This is an engineering
protocol change, not a new selector, skill bank, reward, or RL recipe.

## Stop boundary and new cohort

The original CPU-singleton six-arm queue was stopped at 19:15 UTC with explicit
user authorization. `readout_d` had 0 completed optimizer updates and no
checkpoint; all old episodes, direction batch, router ledger, logs, preparation,
and bank remain at
`/data/disk1/wangyifan/skill-scope-phase3-v3-20260925`.
The stop receipt is `router-acceleration-stop-20260925.md` there.

The first GPU-batch startup at
`/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v1-20260925`
failed before rollout because OmegaConf's `DictConfig` was rejected by a new
plain-`dict` check. Its log and plan are preserved. A targeted compatibility
test was added. The v2 startup reached routing but failed because Ray's
CPU-only TaskRunner had already initialized PyTorch with no visible CUDA
device. A v3 attempt exposed one GPU at TaskRunner actor creation; that
actor's runtime environment then polluted its child Ray GPU workers and
caused their device-ID mapping to fail. Both runs stopped before a policy
update, and both failure logs remain. The final design leaves Ray CUDA masks
untouched and starts a local GPU embedding sidecar. Its new formal root is
`/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v4-20260925`.
Do not resume or splice the three failed/partial roots into v4.

## Frozen semantics and execution change

The six arms retain the same Qwen3.5-4B B0, seed 707, GRPO recipe, 16×8
rollouts/update, 5-update edit windows, U20 first stop, initial SkillNet-37
bank, Qwen3-Embedding-0.6B revision, visible-state query, complete active-bank
catalog, FP32/SDPA encoder, normalized vector dot product, and canonical
top-1 tie break. The encoder is now in a local subprocess on physical GPU 0,
while the Ray CPU-only TaskRunner remains GPU-invisible; live states are encoded
in groups of up to 8 over a bounded binary pipe. Each evaluator GPU shard
launches a sidecar on its own assigned physical GPU. Forward-only cache is emptied after GPU
encoding, retaining model weights. The batch execution has an explicit new
router version/hash and new per-bank cache plus branch-wide budget ledger.

This is not bitwise equivalent to CPU-singleton routing: changing batch shape
and device changes FP32 reduction details. Do not claim universal unchanged
skill choices. The two protocols must not be mixed inside a reported arm;
all six v4 arms use the same batch protocol.

## Preflight evidence

- 128 real states: CPU batch-8, 8 CPU threads, 57.852 s for queries
  (0.452 s/state); GPU FP32 batch-8, 1.338 s (0.0105 s/state), both 128/128
  top-1 matches against old CPU-singleton records. These figures exclude
  local ledger writes and model/index load.
- All 5,100 immutable old first-arm router decisions were rescored by both
  the direct GPU-batch prototype and final GPU-sidecar protocol:
  **5,100/5,100 top-1 matches** in each,
  maximum absolute score difference `4.172325134277344e-06`,
  146.36 s direct / 148.71 s sidecar wall including SQLite decision accounting.
  The direct prototype peaked at 3,897.53 MiB PyTorch allocation; the sidecar
  report does not claim an in-process VRAM peak. The smallest old top-2 margin was
  `8.940696716308594e-07`, so future near-ties remain a numerical caveat.
  Full reports: `/data/disk1/wangyifan/skill-scope-phase3-router-accel-parity-v1/report.json`
  and `/data/disk1/wangyifan/skill-scope-phase3-router-accel-sidecar-parity-v2/report.json`.
- A Ray CPU-only actor successfully routed through the sidecar while retaining
  an empty `CUDA_VISIBLE_DEVICES`; a subsequent Ray GPU actor still received
  its normal GPU assignment. This does **not** replace the concurrent PPO
  memory acceptance check.
- Phase3 and existing batch-router tests: 123 PASS. Two older Phase3 tests
  were updated to reflect the already-approved v3 setting; the new OmegaConf
  path has its own regression test.

## Monitoring and acceptance

The v4 six-arm launcher is `run_six_arms.sh` in the v4 root. It runs
`readout_d`, `skillrl_failure`, `readout_magnitude`, `readout_gated_d`,
`readout_p`, `readout_c` sequentially and stops on a failed arm without
automatic retry. Editor credentials are passed only through the process
environment; no key is stored in the configuration, this document, or logs.

Check `runs/readout_d/logs/train-u0000-u0005.log`, the local router ledger,
GPU0 memory, and eventually `runs/readout_d/metrics/u0001.json` and
`checkpoints/global_step_5`. A first successful route is only routing
acceptance. A full U1 optimizer update is the minimum training acceptance;
the native checkpoint is deliberately saved at U5, not U1. A Phase3 outcome
requires a sealed U20 milestone and independent Seen/Unseen reporting.

Keep the old and failed launch directories. No prior reports, checkpoints,
or source changes were overwritten or rolled back.

### v4 first-rollout acceptance (20:11 UTC, still running)

The first `readout_d` U1 rollout has all 128 episode files. Its local router
ledger has 4,953 successful new decisions, zero failures, median 10.48 ms per
decision, and 87.96 s summed router latency (which is not the rollout wall
time). First selection was at 20:02:38 UTC and all episode files were present
by 20:10:32 UTC, about 7 minutes 53 seconds for this rollout phase. The
observed batch-size accounting includes many batch-8 groups. The prior
CPU-singleton rollout took about 3 hours, but stochastic episode differences
preclude a strict paired end-to-end speedup ratio.
The 128 trajectories include 34 successes, but this is not a Phase3 comparison
result. GPU0 and the seven policy GPUs were simultaneously active without an
observed OOM during this rollout. The U1 old/reference forward and PPO update
had not yet produced a metric or checkpoint; actor-backward co-residency remains
unverified at this acceptance point.
