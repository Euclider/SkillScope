# Phase3 router memory handoff recovery — 2026-10-02

User authorized fixing the stopped queue and restarting: SkillRL to U50
including evaluation, then reward D_sign_balance from U20 to U50.

## Observed stop and preservation

The previous queue stopped at 09:04 UTC. SkillRL U26 completed with eight
finite-gradient audits and 41 Adam steps per rank, but was not checkpointed.
U27 failed during environment reset, before its first policy generation:
the frozen router on physical GPU7 requested 594 MiB with only 156 MiB free.
Native resume is U25, not U26. The retry explicitly redoes uncheckpointed
U26. Its prior metrics, trajectories, direction batch and audits are archived
with hashes, not silently replaced or counted twice. Completed U1–U25 and
reward U1–U20 remain untouched. Old failure logs remain in place.

## Final repair (v2)

The active speed configuration has both actor parameter and optimizer CPU
offload disabled. An initial v1 offload-specific hypothesis was rejected at
configuration review, before any queue launch. Its isolated synthetic test
and registration are retained, but are not evidence that the active path
was fixed. The offload-function change was removed from the main source.

The final change wraps the actor update RPC with a post-return CUDA
synchronize/empty-cache boundary. Function-local tensors have gone out of
scope before this runs. Only unused allocator pages are released; live
actor, Adam and reference tensors remain on GPU. No microbatch, loss,
optimizer, routing rule or random seed changes. This covers the first
router call in the next training rollout and native validation, both of
which occur before vLLM's own forward-time cache cleanup.

Recovery checkpoints are now saved every update; validation/edit windows
remain every five, scheduler horizon remains 150, execution target is 50.
The existing retention limit of two rolling native saves is used, with
the prior window endpoint retained until its successor is sealed. Existing
explicit intra-window resume support remains available; failures do not
trigger automatic retries.

Fifteen known failed local-router queries receive one explicit retry. Prior
ledger and bank cache are backed up first; failed attempt keys are archived
under deterministic recovery keys in one attached-database transaction.
Original failure payloads and counts remain, successful decisions and cache
hits are unchanged, and new attempts still count against the same limit.

## Validation and limits

Regression tests first failed for the missing handoff, then passed.
Synthetic eight-GPU pressure testing kept 22 GiB live (including GPU Adam)
and 8 GiB cached. The repair increased free memory from about 0.75 GiB to
8.75 GiB on every GPU, with exact Adam values and unchanged CUDA RNG.
Real frozen-router index vectors were bitwise identical with and without
the simulated policy memory pressure. No RL update or API call was used
in that preflight. This proves the cache-handoff mechanism and router
parity, not completion of the next full U26→U27 transition.

The targeted Phase3/stable-direction suite contains 189 tests. Full-suite
collection is not passing: `tests/gpu_utility/test_torch_functional.py`
requires the absent optional `flash_attn` package. Bare repository pytest
was interrupted during broad filesystem collection; `pytest -q tests
--maxfail=1` identifies the dependency error. Current SDPA/vLLM imports are
covered separately by the targeted deployment test.

Deployment: `phase3-speed-742f1aa-BlbHzX/candidate-router-handoff-v2`.
Preflight: `phase3-speed-742f1aa-BlbHzX/router-handoff-acceptance-v2`.
Queue: `/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v7-20260927/recovery-two-arm-u50-router-handoff-v2`.
Launcher: `scripts/resume_two_arm_u50_router.py`.

## Launch receipt

At 13:39:00 UTC the new ordered queue started (PID 3825865). The initial
training subprocess declares SkillRL start=25, stop=50. Post-preservation
verification matched all 142 archived file hashes; SQL set comparison
confirmed all 4,543 existing successful decisions in the affected bank
cache are unchanged. Exactly 15 failed queries were reconciled. Active
metric counts remain SkillRL 25 and reward 20, with the previous U26 metrics
retained under `preserved-uncheckpointed-u26`, not destroyed. The final
targeted suite passed 189 tests in 19.18 seconds before launch.

At 13:43 UTC the live trainer showed `Training Progress: 25/150`; all eight
GPU workers were initialized and the queue had no stopped marker. The 150
denominator is the unchanged scheduler horizon; the queue target remains
50. No new completed U26 or full update-to-router transition was observed
at handoff.
