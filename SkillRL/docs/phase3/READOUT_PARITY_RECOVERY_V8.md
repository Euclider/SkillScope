# Phase3 U5 readout parity recovery (2026-09-27 UTC)

The v7 `readout_d` arm completed U1–U5 GRPO, U5 native validation, its
eight-rank model/optimizer/extra-state checkpoint, and the FP32 HF export.
The first U0→U5 prediction attempt stopped at the chosen-token live/offline
parity check. Its registered tolerance was 0.03 log-prob units. The original
`predict-u0005.log`, BF16-weight `predictions/u0000-u0005/` partial directory,
and BF16-weight calibration remain untouched. No editor call occurred.

## Cause and controlled numerical check

The live verl FSDP actor loads FP32 weights, then uses BF16 autocast for each
forward (`verl/workers/fsdp_workers.py`, `verl/workers/actor/dp_actor.py`). The
original Phase3 offline readout instead loaded both checkpoints as BF16
weights. On four fixed decisions from the hash-verified U1 direction batch,
the largest absolute chosen-token discrepancies with BF16-loaded weights
were 0.0170, 0.1684, 0.0569 and 0.0336. Reloading the unchanged U0 model
as FP32 and forwarding under BF16 autocast yielded exactly 0.0 discrepancy
on all four decisions. These are numerical path checks, not utility results
or proof that the entire batch passes.

## Recovery contract

- Both endpoint models now load with FP32 weights, run under BF16 autocast,
  and use the original FP32 softmax/readout arithmetic. They reside on GPU0
  and GPU1 respectively to avoid fitting two FP32 4B models on one GPU.
- The original 0.03 parity tolerance, U1 batch, U0/U5 checkpoints, 37-skill
  bank, reward scores, and Phase3 budgets are unchanged. The same numerical
  backend is used for all six arms from this point onward.
- An explicit `--repair-prediction` is required to bind the failed U0→U5
  attempt to a new `predictions/u0000-u0005-fp32-autocast-v2/` directory and
  distinct log. `events/u0005/prediction_recovery.json` records the binding;
  later runner invocations resolve it without silently retrying or
  overwriting either attempt. The numerical noise calibration uses a new
  `calibration-fp32-autocast-v2.json` file.
- U1–U5 training is never replayed. If the corrected prediction passes,
  normal U5 editing/gate evaluation can seal the event, and the existing U5
  native checkpoint resumes U6–U10. The six arms remain independent.

The 132 Phase3 offline tests pass after the change. The recovered full-batch
parity, editor/gate behavior and subsequent RL are not yet claimed here.
