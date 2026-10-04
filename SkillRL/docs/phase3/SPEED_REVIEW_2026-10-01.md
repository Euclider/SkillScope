# Phase3 execution-only speed review (2026-10-01)

Scope: accelerate training and scoring without changing tasks, seeds, rollout
counts, sampling, reward/advantage definitions, optimizer boundaries, skill
selection, editor evidence, gate tolerances, or evaluation coverage. Existing
processes finish with the implementation already loaded in their memory.

## Measured bottlenecks

For readout_d U1–U20, recorded training time totals 91,520 seconds. Actor
updates account for 63,046 seconds (68.9%), OLD scoring 8,946 seconds,
reference scoring 9,404 seconds, and generation 9,022 seconds. Four readout
windows took 13,467 / 8,586 / 11,173 / 10,284 seconds, using two GPUs and
scoring one decision at a time. These are historical timings, not projected
speedups or a fair end-to-end comparison of experiment arms.

## Training: preserve the already registered candidate

The existing `rl-speed-next-window-v1.json` request dispatches training at
the next SkillRL U15 boundary and at U0 for the four remaining arms. Its
isolated candidate is `/mnt/workspace/users/wangyifan/phase3-speed-742f1aa-BlbHzX/candidate`.
It proposes response-only vocabulary projection, GPU-resident reference
weights, and reduced synchronization during gradient accumulation. Native
eight-GPU gradient/logprob/memory acceptance is required before adoption.
The current review does not alter the candidate or its registered hashes.
Its numerical contract is tolerance-based, not bitwise equivalence.

## Readout: independent GPU pairs and exact ordered merge

The additional implementation lives in `phase3/parallel_predict.py`.
Decisions are deduplicated and sorted exactly as in the serial scorer, then
split into contiguous ranges. Each worker has two independent GPUs for the
unchanged old/new model forward passes. Batch size stays one, with identical
padding, response prefixes, precision, temperature, controls, full-vocabulary
support, and OLD parity threshold. One worker produces the original
eight-decision repeated-forward calibration when needed; others reuse it.

Workers save compact token-level scalars, identity/support metadata, and
hashes, never vocabulary tensors. The parent merges contiguous shards in
the original decision/token order and only then applies the existing
token→decision→trajectory→game weighting. Averaging shard-level skill scores
would be wrong and is not used. Duplicate/missing decisions and changed
shard inputs fail closed. Shard failures stop only newly created workers;
there is no implicit retry of failed work.

`phase3/fast_direction.py` calls the same `_fp64_rows` implementation with
the same 16-token chunks. It avoids the preliminary legacy C/P/D/KL/JS
calculation whose relevant values are overwritten by the stable version.
Phase2's comparator and all seven Phase3 output scores remain unchanged.
This optimization is enabled only for the accepted parallel readout path.

Serial scoring remains available. Run-scoped activation requires a frozen
acceptance receipt and matching source hashes. The in-flight U10→U15
SkillRL readout is left untouched. New scoring processes use the accepted
profile at registered subsequent windows, across all remaining arms.

## Evaluation, routing, storage

Gate and milestone evaluation already shard fixed game/seed jobs over eight
GPUs. Training already uses batched frozen GPU routing. These remain intact.
Concurrent rollout batching, attention-kernel replacements, changed decoding
precision, smaller samples, skipped baseline shadow readouts, and changed
validation intervals are not part of this change. Checkpoint retirement and
integrity hashing retain their current behavior.

## Verification and accounting

CPU tests compare every fast stable scalar bitwise with the original,
including zero/negative/positive advantages and peaked distributions. Tests
also compare full serial and merged bundles, unsupported skills, and support
counts, and reject duplicate/missing/foreign records.

GPU acceptance replays small, stratified subsets of saved training batches
on spare GPUs 2–7. BF16 uses retained U10/U15 checkpoints. FP32 uses the
captured U0 batch and retained U15 weights solely as an engineering test
pair because the U5 export was already retired. This is not a new scientific
update window. Both implementations see exactly the same test pair. There
are no environment continuations, editor requests, optimizer steps, or
changes to experiment artifacts during these probes.

Both complete readout bundles and scoring counts/parity must match exactly.
Elapsed time includes worker/model startup and is reported separately for
the engineering subset. Six-GPU measurements must not be presented as
measured eight-GPU full-window throughput. Formal execution records its GPU
allocation, code hashes, compact shards, actual wall time, and forward costs.

## Accepted evidence and activation

The Phase3 suite plus `tests/phase2/test_stable_direction.py` passed:
**167 tests**, including exact scalar parity, four-shard ordered aggregation,
completed-cache source rejection, and boundary/fingerprint activation guards.

Engineering measurements (seconds include model/worker startup):

| Replay | Precision | Decisions | Serial, 2 GPUs | Parallel, 6 GPUs | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 | BF16 | 48 | 140.56 | 61.51 | 2.29x |
| v1 | FP32 | 48 | 175.56 | 79.10 | 2.22x |
| v2, final code | BF16 | 24 | 82.31 | 44.48 | 1.85x |
| v2, final code | FP32 | 24 | 95.63 | 49.07 | 1.95x |

All four comparisons matched the entire readout bundle exactly. Forward
counts, input-token counts, and OLD parity also matched; maximum chosen-token
logprob error was zero on these probe rows. v2 repeated acceptance after
adding completed-cache provenance validation and stricter worker-source
checks; numerical kernels did not change between v1 and v2.

Evidence roots (preserved, no formal experiment reports overwritten):

- `/mnt/workspace/users/wangyifan/phase3-readout-speed-20261001-v1`
- `/mnt/workspace/users/wangyifan/phase3-readout-speed-20261001-v2`

Run-scoped activation has been registered in
`/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v7-20260927/readout-speed-next-window-v1.json`.
It pins the v2 receipt and eight source hashes, enables four GPU pairs at
SkillRL start=15 and the four remaining readout arms at start=0, and excludes
the already-completed readout_d arm. The in-flight SkillRL U10→U15 scorer
continues serially. No queue restart is needed. All nine input fingerprints
of the independently registered training-speed request remain unchanged;
its native eight-GPU acceptance is still pending at the next boundary.

These measurements are engineering replays, not new utility labels or
additional RL runs. The final eight-GPU full-window speedup is unmeasured.
Do not compare historical arm wall times as if execution profiles matched,
or interpret the wall-time reduction as an equal reduction in GPU-hours.
