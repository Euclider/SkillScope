# SkillRL-only U50 continuation and editor timeout recovery

The user approved extending only the current SkillRL branch to U50, after
finishing the sealed U20 policy's outstanding edit, gate, and full seen/unseen
evaluation. The other five arms retain U20 targets and are not started by
this continuation launcher. Comparisons at U50 are not matched-horizon
comparisons to arms that only reached U20.

## Scope and unchanged scientific settings

- Complete U20 first, then run six adjacent windows U20→25 through U45→50.
- Keep GRPO, learning rate 1e-6, 16×8 trajectories/update, frozen within-window
  bank, five-update editing cadence, and the existing 150-update learning-rate
  schedule. U50 is the stopping point, not a redefinition of the scheduler.
- Preserve editor model/prompt/evidence, SkillRL failure-driven selection,
  gate tasks/seeds/tolerances, router, all diagnostic readouts, and evaluation
  coverage. No outcome-dependent threshold or sample-size adjustment.
- Retain existing checkpoint retirement, disk guards, 30 editor calls per
  branch, and 600,000 local router calls. No caps are silently removed.

## Timeout change

The original U20 editor attempt
`2fb316be3d76df8fbccab00aef0fa1b352ff6ff1af050d91891b1c70a32ceca1`
failed after about 61 seconds. The user authorized one explicit retry and
an effective timeout of 600 seconds for subsequent SkillRL editor calls.

`JSONClient.authorize_transport_timeout` stores a checksummed authorization
in a separate SQLite table. It does not rewrite the 60-second historical
profile or old request keys/results. Completed calls remain cache hits;
failed calls still require a separate one-shot authorization. New attempts
record their effective timeout and transport authorization hash. The retry
shares the original API call cap; SDK automatic retries remain disabled.

## Training continuation implementation

`scripts/continue_skillrl_u50.py` is an independently authorized entrypoint.
It verifies the original acceleration request, original candidate source
fingerprints, dependency-completion receipt, software/hardware identity,
and cached eight-GPU acceptance. It does not patch the old launcher's U20
bound or modify accepted tensor kernels. A separate authority restricts the
new entrypoint to SkillRL windows starting at 20, 25, 30, 35, 40, or 45 in the
exact existing run directory.

The original `phase3.run` orchestrator is reused. Only training subprocesses
in the newly authorized windows are routed to the continuation entrypoint;
model export, readout, editing, gate, and reporting retain their existing
paths. The old U20 RL, export, and readout are not replayed. A failed new
stage stops without automatic resubmission or overwriting evidence.

Records:
`/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v7-20260927/recovery-skillrl-u50-v1`.

## Pre-training performance review

U16–U20 averaged 94.55 minutes/update, versus 88.73 minutes for U11–U15;
average token volume increased about 5.2%. Actor time/token was effectively
unchanged (0.5182 versus 0.5173 ms); actor compute still accounts for about
67% of total time. Reference scoring improved from about 9.07 to 8.14
minutes/update. These are different batches, not controlled speed trials.

The existing four-pair/eight-GPU readout completed 5,803 decisions in 63.59
minutes, versus the preceding serial window's 5,410 decisions in 221.50
minutes. This is a 3.48x observed wall-time difference across different
windows, not an exact same-batch comparison. Keep this accepted profile.

Gate and milestone evaluation already use eight GPU shards. More aggressive
padding trimming, attention-kernel replacement, larger microbatches, or
concurrent per-GPU episode batching require new numerical/RNG/memory
acceptance. None are enabled in this continuation. In particular, do not
change the experiment to obtain a claimed speedup or reduce the validation
sample size. No new tensor-kernel optimization is claimed here.

## Verification

- New timeout and continuation-boundary tests were observed failing before
  implementation, then passing.
- Phase3 plus stable-direction suite: 174 passed, including the real isolated
  training import check. Full trainer/FSDP/environment entrypoints and U20/U45
  configurations passed the local continuation preflight.
- Whole `tests/` collection stopped at
  `tests/gpu_utility/test_torch_functional.py` because `flash_attn` is absent.
  This is not a full-suite pass; dependencies were not installed or changed.
- The registered acceleration inputs and existing acceptance are unchanged.
  Future full-window timings must be measured from their own run records.

## Launch observation

The SkillRL-only queue started on 2026-10-01 at 19:37:12 UTC (PID 3701729).
The explicit U20 editor retry returned successfully in 53.10 seconds with
`effective_timeout_seconds=600`. Its proposal was saved and eight-GPU paired
gate evaluation started. At this observation point U20 milestone evaluation
was not yet complete and U21 had not started. The queue proceeds to the U50
stage only after U20 seen/unseen evaluation seals successfully; it does not
skip those evaluations or repeat the completed U20 training/readout.
