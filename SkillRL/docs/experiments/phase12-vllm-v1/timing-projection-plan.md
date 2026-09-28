# Measured timing plan after user confirmation — 2026-09-18

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-18
- Verification Status: UNVERIFIED
- Version Label: timing-projection-v1

User confirmed running the existing5-iteration setting first and estimating50/100
iterations afterwards. They then waived the30min **engineering-preflight** hard
limit. This does not automatically remove the existing30h formal-run deadline,
storage guards or change the scientific scope. Formal training has not started at
this plan's creation. New preflight is in `R/vllm-gpu-preflight-20260918-v1`;
`deadline-waiver.json` documents the timer-only adjustment without stopping workers.

## Analytic workload, not a measured ETA

All larger-horizon counts below assume the SAME protocol at EVERY fixed5-iteration
window: complete seen/unseen performance, unseen natural-anchor selection, max12
skills ×12anchors ×(1evidence+2gold seeds) ×3arms ×2endpoints. They do not authorize
50/100-iteration runs or create alternative experimental preparations.

| Iterations | Windows | Model endpoints | Training trajectories | Full performance episodes | Anchor-source episodes | O/P/N continuations upper bound |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 2 | 640 | 548 | 134 | 2592 |
| 50 | 10 | 11 | 6400 | 3014 | 1340 | 25920 |
| 100 | 20 | 21 | 12800 | 5754 | 2680 | 51840 |

Train monitor adds64 episodes every5iterations (current val_before_train=False).
The existing admission workload helper conservatively includes an extra initial
64-episode allowance:128/704/1344 respectively, an upper bound rather than the
actual current monitor schedule. Skill support and early termination affect the
number/length of actual continuations. Unsupported skills must remain abstentions.

## Measurements required

1. Native worker and vLLM initialization/JIT separately from steady-state compute.
2. Per training iteration: trajectory count, environment steps, prompt/output
   tokens, router walltime, generation, reference/old/new forwards, optimizer
   update, exact-vocabulary archive I/O, monitor and checkpoint time.
3. Evaluation per shard: walltime, completed episodes, active GPU time, steps/
   tokens. Aggregate using parallel walltime/max shard duration, not summed GPU
   seconds or single-GPU episode duration multiplied by all episodes.
4. Phase2 per window: anchor collection, O/P/N, feature readout, aggregation,
   prediction commitment, report, compact seal and authorized temporary-row reclaim.
5. Storage high-water mark and sampled-token volume, not just checkpoint size.

Use `metrics/uNNNN.json` and `timing_s/{gen,old_log_prob,ref,update_actor,
phase2_post_log_prob,testing,save_checkpoint,step}` once real training exists.
The supervisor currently restarts native workers each5-iteration block, so future
50/100 estimates must include10/20 block initializations; GPU-kernel disk caches
may make later startup faster, but this is not yet measured.

Present two distinct ETAs: RL training only versus the complete Phase1–2 protocol.
Avoid multiplying the first window's entire walltime by10/20: B0 export is once,
adjacent performance endpoints are shared, startup/warmup changes, and trajectory
length/support may evolve. Report a range and assumptions from observed iterations,
not a guaranteed6–15h completion claim.

## Storage warning for extension, not a current-run failure

Historical native synthetic checkpoint bytes:53,033,819,038 (~49.39GiB).
Historical B0 export bytes:8,431,563,005 (~7.85GiB). These are measured historical
file sizes, not verified sizes of the new vLLM run. Retaining a native checkpoint
at every5iterations plus endpoint exports would be approximately580GiB at50 and
1159GiB at100 iterations, before trajectories/probability intermediates/other data.
New layer-wrapped checkpoint size must be measured. The machine had~711GiB free
before this preflight; existing100GiB free-space and80GiB next-checkpoint reserve
guards remain. Extending beyond5 therefore needs a storage plan too; old reports,
raw evidence and checkpoints may not be silently deleted.

Full FP32 old+new vocabulary capture costs1,986,560 bytes per output token before
metadata. Its temporary peak must be included even with authorized per-window
reclamation; no fake admission PASS may be created from the desire to start.
