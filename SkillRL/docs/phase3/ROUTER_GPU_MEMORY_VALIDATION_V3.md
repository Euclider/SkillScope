# Phase3 U5 validation OOM and v6 recovery (2026-09-26)

Status: v6 prepared and preflighted; real U5 validation remains the acceptance gate.

The v5 first arm at `/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v5-20260926`
saved U1--U4 optimizer metrics. U5 training rollout was collected, but native
Seen monitoring failed during its first embedding route, before the U5 metric
and native checkpoint. The router sidecar reported an OutOfMemoryError on GPU0:
8.25 MiB free and a 12 MiB request. The policy worker held 26.81 GiB and the
sidecar was already resident. The queue stopped; no later arm started. All v5
files remain untouched. Since U5 has no native checkpoint, v6 starts from B0
in a new directory and must not splice v5's partial updates.

The v6 root is `/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v6-20260926`.
Its registered vLLM profile changes only `gpu_memory_utilization` from 0.45
to 0.38, giving roughly 2.2 GiB more capacity on a 31.36 GiB card for the
co-resident router and validation transients. The training coordinator's
FP32 router sidecar moves from physical GPU0 to GPU7; evaluator shards still
use their individually assigned GPU. The model, FP32 microbatch-2 top-1
retrieval, prompts, 16 x 8 GRPO batch, learning rate, optimizer horizon,
seed 707, five-update validation and checkpoint cadence, six-arm order, and
U20 first stop are unchanged. The new preparation and router ledgers are
independent. No silent CPU fallback, API router, or failed-attempt retry is used.

Checks before launch:

- Phase3 and batch-router regression tests: 129 PASS.
- Eight-card synthetic native/vLLM preflight at `v6/gpu-preflight-038`:
  all eight rank receipts and `complete.json` PASS; 16-sequence 4096+512
  generation, two optimizer updates, strict checkpoint/RNG restore, and
  post-update weight synchronization passed. This is not an ALFWorld RL run.
- GPU7 sidecar rescored 4,961 real v4 U1 states with 4,961/4,961 top-1
  agreement, maximum score difference 5.781650543212891e-06 and 149.51 s
  wall time. Repeating with a separate 27,634 MiB allocation on GPU7 passed
  the same states in 150.55 s; an instantaneous GPU report showed 1,545 MiB
  free. That pressure process was stopped before formal launch.
- The 0.38 synthetic rank-7 maximum-length generation stage took 22.28 s;
  the earlier 0.45 preflight's rank-0 stage took 23.02 s. This is an
  engineering smoke comparison, not a paired proof of equal RL throughput.

The full real U1--U5 training/validation path has not yet been validated by
these component tests. It must produce U5 metrics and a complete native
checkpoint before any Phase3 readout or editor action is considered ready.
No Phase3 performance conclusion is available from v5's partial updates.

Launch acceptance at 2026-09-26 13:32 UTC: the v6 sequential queue and first
`readout_d` training process are alive, all eight Qwen3.5-4B/vLLM GPU workers
are loaded, and the first 16 local GPU7 router attempts all succeeded. The
formal RL update count is still zero at this acceptance point.
