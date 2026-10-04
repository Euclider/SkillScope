# Explicit U15 editor-timeout recovery

The readout_d arm completed U1–U15 training, the U15 native/HF checkpoint,
and U10→U15 prediction. Its third editor attempt failed with APITimeoutError
after 60.714 seconds, before any proposal or gate result was persisted.
The user authorized resuming with one explicit retry on 2026-09-29.

Original request SHA-256:
`531272e33bf439c1df524bd550d5634bfe01d632d768f47c9c83747e52a8299a`.

`phase3.run --execute --retry-editor-request SHA256` authorizes one retry
only for that exact recorded editor timeout. The original request/result
remain unchanged in `editor.sqlite3`. A separate retry authorization and
attempt link to their hashes; both attempts count toward the original
per-arm API budget and report totals. Missing usage after a timeout remains
unknown rather than zero. The provider may have processed the timed-out
request, so duplicate provider cost cannot be ruled out.

The explicit retry uses a 600-second SDK request timeout. This is recorded
in the authorization, retry request, and accounting. The frozen base API
profile remains unchanged. Model, prompt, evidence, decoding parameters,
token limits, ranking, and gate rules are unchanged. The failed U15 input
estimate was 318,966 tokens; the previous U5 success took 56.679 seconds,
making the original 60-second transport timeout plausibly too short.

The retry is not an automatic retry policy. Its failure or an unresolved
reservation stops the queue again. Success can be cached on later resumes
without resending, even if the CLI flag is omitted. Non-timeout failures,
router calls, and attempts that are already retries are not eligible.

Local recovery uses a new `recovery-u15-editor-timeout-v1` log directory
under the existing v7 experiment. The queue resumes readout_d at U15 editing,
then finishes U20 and its milestone evaluation before starting the other
five arms in the existing order. U1–U15 training and sealed prediction are
reused. Checkpoint retention follows the previously approved policy after
an event is sealed.

Verification: all 147 tests under `tests/phase3` passed. Added coverage for
immutable failed-attempt history, unchanged request content, cached success
after restart, single-retry enforcement, total-call accounting and budget,
and rejection of unknown, ambiguous, or non-timeout attempts.

Live recovery: the original ledger entry remained failed and attempt 4
returned successfully in 57.747 seconds. The response reported 318,651
prompt tokens and 3,213 completion tokens and proposed three MODIFY
operations. The paired development gate then started successfully. This
receipt establishes restored execution, not performance gains.

Both sides of the U15 gate completed 24 episodes across 12 games. The
candidate succeeded on 5/24 versus 7/24 for the current bank and was rejected
under the frozen tolerance. The U15 event was sealed and the U16–U20
training subprocess launched with `resume_path=global_step_15`. Approved
retention released `global_step_14`, `global_step_10`, and `models/u0010`;
U15 and the trajectories, batches, bank versions, and reports remain.
