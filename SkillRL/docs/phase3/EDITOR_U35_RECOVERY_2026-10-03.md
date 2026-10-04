# U35 editor-response recovery — 2026-10-03

User requested repair and continuation after a read-only status check.
Scope: SkillRL U35 editor/gate, then training to U50 and seen/unseen
evaluation; reward D_sign_balance subsequently resumes U20→U50. No
completed RL, checkpoint export or readout is replayed.

## Failure evidence

The router-handoff-v2 queue stopped at 04:36:25 UTC. SkillRL U35 native
checkpoint (8 model/optimizer/extra-state shards), HF export and U30→U35
readout were complete. The gpt-5.5 editor returned a response in 49.5 s,
but local checking raised ProtocolError. The old ledger recorded neither
the validation phase nor the rejected response, so its exact cause cannot
be reconstructed. This is not evidence of another router OOM or RL failure.

Original request:
`425c059be1df8bc54a365fc6a7d754bfd17db486f43a8f397302bb93a1ce9f7c`.
The original failed row, including its recorded 446,024 total tokens,
is preserved. A SQLite backup precedes retry authorization.

## Minimal API changes

- Failures record transport/completion/message/JSON/validation phase,
  an allowlisted local constraint message when available, finish reason,
  response-text hash and length. No gateway exception body, credentials
  or arbitrary response text is logged.
- One explicitly authorized editor-response retry uses a distinct request
  key linked to the original request/result hashes. Messages, schema,
  model, mutation budget, evidence, validation and provider parameters
  remain unchanged; the failed and retried calls both count against the
  same API limit. Failed retries remain terminal; successful retries are
  reused on resume. Timeout recovery remains independently supported.
- The accepted isolated GPU training deployment is not modified.

The old U35 failure is still causally unresolved; this repair closes the
diagnostic gap and enables a bounded recovery rather than claiming that
an unknown validation error has been eliminated.

## Validation and execution

Four added regression tests failed before the patch. Afterwards 193
Phase3/stable-direction tests passed. Full `pytest -q tests --maxfail=1`
still stops at collection of `tests/gpu_utility/test_torch_functional.py`
because optional `flash_attn` is missing; no dependency was installed.

The first new launcher was stopped after a credential transcription issue
was detected before any new API reservation; its record is retained under
`recovery-two-arm-u50-editor-u35-v1`. No additional call was spent there.
Corrected queue `recovery-two-arm-u50-editor-u35-v2` started at about
05:20 UTC, PID 3906252. Source/evidence hashes are in `authorization.json`;
only `phase3/api.py` differs among the prior registered orchestration
sources. New launcher: `scripts/resume_two_arm_u50_editor.py`.

At 05:22 UTC the authorized retry succeeded with the unchanged validator:
three MODIFY operations, 37 active skills retained. The proposal is saved
and U35 paired candidate validation is starting. Reported retry usage:
442,851 prompt tokens (442,624 cached), 2,706 completion tokens, 445,557
total. These are additional to the failed original call, not replacements.
The original failed response's exact cause remains unknown. Subsequent
gate acceptance and U36 training were not yet complete at this observation.
