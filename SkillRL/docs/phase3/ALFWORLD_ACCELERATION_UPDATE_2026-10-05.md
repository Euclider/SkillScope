# ALFWorld Phase3 acceleration and recovery update

This additive publication starts from SkillScope commit
`742f1aa99840b267d644634fe8a4b0fd44b1683f`. It does not replace the repository
with the older local research worktree. Existing LogicBench rollout, dataset,
position handling, exact skill-payload rendering and encoder runtime support
are retained. The active local experiment is not changed by publication.

## Included implementation

- `alfworld_rl_speed/`: run-scoped registration, forward prescreen, native
  eight-GPU gradient/memory acceptance and guarded training dispatch.
- Existing response-only logits, optional no-sync gradient accumulation and
  GPU-reference controls remain available; ALFWorld adds per-rank Adam-step
  and finite-gradient audits. Common-padding trimming remains disabled in
  the accepted ALFWorld profile.
- `phase3/parallel_predict.py` and `fast_direction.py`: independent old/new
  GPU pairs, compact scalar shards, exact ordered merging and provenance
  checks. No shard-level averaging of skill scores or changed readout formula.
- Finite saturated PPO/K3 exponentials and post-actor unused-CUDA-cache
  release. Live actor/Adam/reference tensors stay on their configured device.
- Explicit editor retry diagnostics, failed-router evidence preservation,
  rolling per-update recovery checkpoints and retention-aware authorization.
- Regression tests, engineering receipts/results, and chronological recovery
  notes through October 3. No new scientific result is implied by this release.

## Reading order

1. [Current protocol](SETTING.md), [startup instructions](START.md).
2. [Speed review and measured readout probes](SPEED_REVIEW_2026-10-01.md).
3. [Numerical repair](TWO_ARM_U50_RECOVERY_2026-10-02.md).
4. [Final router handoff repair](ROUTER_HANDOFF_RECOVERY_2026-10-02.md).
5. [Editor U35 recovery](EDITOR_U35_RECOVERY_2026-10-03.md) and
   [retention U40 recovery](RETENTION_U40_RECOVERY_2026-10-03.md).

## Portability and acceptance boundaries

The `scripts/resume_*`, `prepare_*_recovery.py`, and ALFWorld registration
scripts include intentionally hard-coded original run roots, starting
updates and evidence hashes. They are historical, guarded operational
scripts, **not portable one-command launchers**. Do not execute them on a
new server without creating its own reviewed run registration. The original
speed dispatcher is restricted to its registered pre-U20 windows; subsequent
U50 continuation uses the separately registered recovery launchers.

Fresh servers must follow START.md to install the pinned environment, supply
their own model/ALFWorld data paths and inject editor credentials securely.
No dependency upgrade, new GPU experiment or API request was performed for
this upload. The optional `flash_attn` package is absent in the local
SDPA/vLLM environment, so broad test collection is not a full-suite pass.

Copied `evidence/2026-10-05/` JSON files are original engineering receipts,
not fresh-server acceptance. Their absolute paths and source hashes refer
to the original isolated deployments. This publication combines those
changes with the remote's newer cross-benchmark code; consequently old
whole-tree fingerprints must not be reused to authorize it. Re-run CPU
regression, native restore/gradient/memory checks and serial/parallel readout
parity on a fresh server before enabling the corresponding speed options.

Historical small-subset speedups are not a measurement of full-run throughput
or GPU-hour savings. The router pressure test is synthetic with a real frozen
encoder, not a complete RL update. Older notes are dated observations rather
than current live-status claims.

No credentials, checkpoints, full-vocabulary tensors, SQLite ledgers, raw
rollouts, model weights or datasets are included in this update. File hashes
and source provenance are listed in the additive release manifest; unrelated
remote files and the earlier full-research manifest are preserved.

## Publication validation (October 5)

The combined Phase3, stable-direction and router test run finished with
**347 passed, 3 failed**. All three remaining failures were reproduced
against the unmodified remote parent under the same local environment:

- `test_router_readiness_probe_does_not_inherit_editor_secret`: the local
  SDK/environment does not meet that LogicBench runtime's readiness checks.
- `test_capture_single_step_actual_batch_and_versions`: its configured
  `/home/wangyifan/model/Qwen3.5-4B` path is absent on this server.
- `test_banked_window_parquet_resume_preserves_numpy_chat_cells`: existing
  partial-window dataset checksum failure, reproduced on the parent.

An integration conflict in the editor timeout cap was caught before
publication. The release retains the remote parent's editor maximum of
600 seconds (router maximum 60), together with the new explicit retry
auditing. Its timeout-recovery tests now pass. No previously failing tests
were silently skipped or reclassified as passing. Broad test collection
still stops at `tests/gpu_utility/test_torch_functional.py` because the
optional `flash_attn` package is absent. No full-suite/GPU acceptance is
claimed for the combined release.
