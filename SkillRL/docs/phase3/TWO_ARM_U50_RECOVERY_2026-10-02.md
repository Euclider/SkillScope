# SkillRL then reward readout: U50 recovery

## Authorized scope

Continue the existing SkillRL branch from its sealed U25 endpoint through
U50, including evolution/gates and the full registered seen/unseen milestone.
Only after that milestone is sealed and GPU workers exit, resume `readout_d`
(`D_sign_balance`) from its retained U20 endpoint through U50 with the same
evaluation. The other four arms are not started. The optimizer/scheduler
horizon remains 150, learning rate 1e-6, GRPO 16 x 8 trajectories/update,
five-update windows, and existing task/evaluation seeds. U50 is an execution
stop, not a change to the learning-rate schedule.

## Failure and evidence boundary

SkillRL completed U25, its readout, editor/gate, and retained native/HF
checkpoints. U26 completed sampling and OLD/reference scoring but emitted
non-finite gradients. The speed audit then rejected the missing Adam step.
Do not weaken that audit or count a skipped optimizer step as successful.
No U26 completed metric or resumable checkpoint exists. Completed speed-audit
records through U25 all report finite gradient norms.

The captured U26 batch contains 5,224 rows and 79,313 actual loss tokens;
none of its rows has an empty loss mask. Advantages and captured OLD
logprobs are finite. Reference replay of 24 low-probability decisions found
a reference-minus-OLD gap up to about 68.96. The actual NEW probabilities at
the failed internal minibatch were not saved, so this does **not** prove that
exponential overflow was the unique cause of that failed full update.

Independent regression tests reproduce a concrete bug: PPO's dual-clipped
objective and clipped K3 can have finite saturated forward losses but NaN
backward gradients when `exp` overflows before clipping. Zero advantages
can also produce `0 * inf`. The repair caps only the mathematically already
saturated positive exponential tail. The PPO ceiling exceeds both existing
clip bounds; K3 clamps its intermediate log-ratio to 20 before computing
the entire expression, whose final result is already capped at 10. Tests
include extreme gaps, all advantage signs, and exact ordinary-range loss
and gradient parity. The objective, coefficient, clipping bounds, and
optimizer-step accounting are unchanged.

## Verification

The original deployment remains intact. The repaired deployment is
`/mnt/workspace/users/wangyifan/phase3-speed-742f1aa-BlbHzX/candidate-finite-exp-v2`.
Only `verl/trainer/ppo/core_algos.py` differs from that original candidate.
The same scalar fix is present in the main repository for portability.

Native eight-GPU regression uses the U25 checkpoint and the 128 U26 rows
with lowest captured action logprobs. The probe captures gradients but never
updates parameters or writes a checkpoint. v2 baseline and accelerated
probes passed, with zero chosen-logprob error, maximum gradient relative L2
about 0.0070 and minimum cosine about 0.9999755. A final v3 accelerated probe
reuses the sealed v2 baseline and tests the final K3 extreme-tail algebra.
These are numerical engineering checks, not a replay or successful
completion of the full failed U26 training update.

Evidence is under `finite-exp-acceptance-v2` and `finite-exp-acceptance-v3`
in the deployment parent directory. The continuation is gated on the final
receipt, source fingerprints, checkpoint structure, and 185 passing
Phase3/stable-direction tests (including actual deployment imports).

## Preservation and execution

`scripts/resume_two_arm_u50.py` preserves the failed U26 batch, its 128
episodes, segment configuration, and speed-profile record under a new
`preserved-incomplete-u26` directory before retrying this unfinished update.
Original error logs remain in place. Completed updates U1–U25 and the
reward arm's U1–U20, all completed editor/gate results, and prior milestone
reports are reused and not overwritten. Because the partial U26 optimizer
state was not checkpointed, recovery restores U25 and retries U26; it does
not claim to resume the failed in-memory minibatch.

Both branches retain the accepted speed options and use the already tested
four-pair parallel readout implementation. Readout equations, aggregation,
selected skill pool, evidence source, editor, and gate are unchanged. Editor
transport timeout remains 600 seconds; API/local-router limits and disk
protection are unchanged. Any subsequent failure stops the queue rather
than silently retrying. The runtime finite-gradient and exact-step audits
remain mandatory; full-update recovery is not established by probe success.

Queue/authority/evidence directory:
`/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v7-20260927/recovery-two-arm-u50-finite-exp-v2`.

Final v3 eight-GPU regression passed. The targeted suite passed 185 tests.
The wider `pytest -q tests --maxfail=1` run stopped during collection of
`tests/gpu_utility/test_torch_functional.py`: the optional `flash_attn`
package is absent. This is not a full-repository pass; the deployed
SDPA/vLLM path was checked separately above.

## Launch observation

The ordered queue launched at 2026-10-02 07:40:33 UTC (PID 3786394).
At 07:45 UTC its training log showed `Training Progress: 25/150`, confirming
restoration to the existing SkillRL U25 state; all eight GPUs held the
training workers. The denominator retains the original scheduler horizon,
not the execution stop, which remains U50. No completed U26 update was
observed at this handoff. Reward remains at U20 and is queued only after
SkillRL U50 and both seen/unseen milestone evaluations complete.
The archived incomplete-U26 files were hash-verified after preservation.
