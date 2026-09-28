# Phase3 U5 router OOM recovery (v7, 2026-09-27 UTC)

## Observed failure and evidence boundary

The v6 `readout_d` arm completed U1–U4 training metrics but stopped during the
U5 native Seen validation router forward. GPU7 had about 12 MiB free when a
second Qwen3-Embedding-0.6B sidecar requested 20 MiB. The training parent had
been paused for the editor-model handoff, but its training child exited with
`EmbeddingRouterError(OutOfMemoryError)`. No U5 native checkpoint exists. No
U4 native checkpoint was configured in v6, so U1–U4 metrics are evidence of
the failed attempt, **not** an optimizer-resumable state. The editor ledger has
zero calls and the v6 editor profile remains `o3`; no skill revision occurred.

The v6 run root and its logs, metrics, episodes, router ledger and frozen assets
remain untouched. The stale paused parent was terminated only after its
training child had exited and all GPUs were idle. The new v7 run uses a fresh
root and a fresh preparation; it reruns U0→U5 rather than claiming to resume
from nonexistent U4 weights.

## Scoped v7 change

- Train and native-validation environment managers in one Ray coordinator
  reuse one thread-safe frozen GPU encoder sidecar when model, profile,
  microbatch, GPU and ledger identity all match. The old implementation
  instantiated two copies on GPU7. Encoder weights, FP32 math, top-1 scoring,
  full-bank candidates and cache keys are unchanged.
- The registered vLLM profile reserves 0.36 of GPU memory instead of 0.38.
  The profile is used uniformly for all six arms and evaluation. An independent
  GPU7 load-and-generate preflight passed at 0.36; this does not prove a full
  U5 co-residency pass.
- The native trainer saves an extra checkpoint at the penultimate update of
  each five-update window (U4, U9, U14, ...). It retains up to two checkpoints
  within the block, so a U5 validation failure leaves U4 model, optimizer,
  dataloader and RNG shards for explicit recovery. After the successor event
  is sealed, only that exact penultimate checkpoint is removed under a
  retention receipt. Other trajectories, metrics and reports are retained.
- Intra-window recovery requires an explicit `--resume-update`; there is no
  automatic retry. The original five-update segment and failed log are never
  overwritten. The RL seed, 16×8 trajectories/update, GRPO settings, five-
  update readout/edit window, router model, editor model and budgets remain
  unchanged.

If a later U5 validation fails, first verify `checkpoints/global_step_4` is
complete and reconcile any interrupted router-cache reservation. Only then
invoke the same branch runner with `--resume-update 4 --execute`; it restores
the native model/optimizer/dataloader/RNG and writes a distinct recovery
segment and log. Do not run that command while the queue is still active or
assume a JSON metrics file can replace the native checkpoint.

Validation before launch: 131/131 Phase3 offline tests passed, including
encoder sharing, recovery configuration and checkpoint-retention guards.
The new six-arm v7 queue has started with `readout_d`. It is not yet a Phase3
result; the next meaningful acceptance point is a complete U4 native checkpoint
followed by U5 native validation and a sealed U5 event.
