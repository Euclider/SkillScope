# Phase3 GPU router memory recovery (2026-09-26)

Status: v5 launched; full PPO co-residency past U1 still pending.

## Failure boundary

The v4 first arm at `/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v4-20260925`
completed U1 (metric `runs/readout_d/metrics/u0001.json`) but failed when the
next rollout reset invoked the GPU embedding sidecar. The sidecar log records
an OutOfMemoryError on GPU0 while requesting 46 MiB: the policy worker held
26.97 GiB, the sidecar held 4.34 GiB, and only 11.56 MiB remained. The queue
stopped at the first arm. U1 is not a resumable native checkpoint; checkpoints
are written at U5. The complete v4 root, including U1 metric, log, router
ledger, and failed-attempt evidence, remains unchanged.

## Registered mitigation

The v5 root is `/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v5-20260926`.
It retains the same FP32/SDPA model, query, full-bank state-aware top-1 rule,
outer groups of up to eight states, six-arm order, seed 707, GRPO settings,
and U20 first stop. The SentenceTransformer forward pass now uses deterministic
microbatches of two inside each outer group. This changes numerical execution,
so the router mode and protocol version are new; v4 and v5 caches are not mixed.
No CPU fallback or paid router API is used. All six arms use the same v5 mode.

On 4,961 recorded v4 first-arm states, v5 top-1 matched 4,961/4,961;
maximum absolute skill-score difference was 5.781650543212891e-06. Complete
rescoring took 146.39 s without memory pressure. With a separate process
holding 27,634 MiB on GPU0 (approximating v4's policy-worker occupancy), the
same 4,961 states passed in 149.97 s with the same top-1 choices. An observed
sidecar allocation was 3,194 MiB, leaving roughly 1 GiB free at that instant;
this is not a certified peak or proof against all future PPO states. The
pressure holder was stopped and all GPUs were free before v5 launch.

Evidence: `parity-v4-u1/report.json` and
`parity-under-27gb-pressure/report.json` under the v5 root. Phase3 and
batch-router regression tests: 128 PASS. The full real-training acceptance
gate is the transition from U1 update into U2 routing without an OOM, followed
by the first U5 native checkpoint. No prior partial U1 result is spliced into
v5, and no Phase3 outcome should be reported before a sealed U20 milestone.
