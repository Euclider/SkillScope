# U15 accelerated-training import recovery

The SkillRL branch stopped on 2026-10-01 at 04:31 UTC, after the four native
eight-GPU speed probes had passed and U15 evolution had been sealed. The
isolated speed candidate omitted `gigpo/core_gigpo.py`, which the common
trainer imports unconditionally. The configured algorithm is still GRPO;
this repair does not switch to GiGPO.

Recovery adds a byte-identical copy of the main repository's module to the
isolated candidate. Existing accepted sources, optimization options,
dependency versions, training configuration, and acceptance files remain
unchanged. The added dependency is separately hashed in
`/mnt/workspace/users/wangyifan/phase3-speed-742f1aa-BlbHzX/dependency-completion-v1.json`.
The recovery preflight verifies this receipt, all original source/input
fingerprints, the saved native U15 checkpoint, and the complete trainer,
FSDP worker, and environment import chain before starting any worker.

Regression evidence: the new real-subprocess import test first reproduced
the missing `gigpo` error and then passed after deployment completion.
The Phase3 suite plus stable-direction tests passed 169 tests. A separate
test restricts log redirection to the recorded pre-training U15 failure and
rejects replay if U16 metrics, its captured batch, or the U20 checkpoint
already exists. Other subprocess logs retain their original handling.

`scripts/resume_phase3_speed_import.py` resumes the original experiment via
its original `phase3.run` implementation. Only the failed training log path
is redirected into the new recovery directory. The original traceback is
untouched. U1–U15 updates, U15 readout, editor call, and gate are not repeated.
The sequence remains SkillRL, magnitude, legacy gated D, -P, +C; each stops
at U20 as previously approved. Credentials are passed only in process
environment, never in the launch record or arguments. There is no automatic
retry on subsequent failures.

Recovery logs and process records:
`/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v7-20260927/recovery-u15-speed-import-v1`.
The previously registered parallel-readout activation remains unchanged.

The unscoped repository `pytest -q --maxfail=1` invocation was interrupted
after 115.95 seconds during filesystem collection, before any tests ran;
the repository includes large experiment artifact trees. This is not a
full-repository pass. The 169-test result above is the completed targeted
suite, including the isolated deployment import regression.

The recovery queue was launched at 2026-10-01 09:53:36 UTC (PID 3657350).
Actual progress must be read from the recovery logs and per-update metrics;
launch alone does not establish completion of U16.

At 09:57 UTC the native checkpoint restore had passed the trainer's
`global_steps == segment_start` assertion and the progress bar resumed at
15. All eight workers had allocated model memory; actor logs confirmed
`response_logits_only=True`, `trim_common_padding=False`, and
`accumulate_no_sync=True`. A new GPU router sidecar had started. No U16
completion metric existed yet; full-update throughput remains unmeasured.
