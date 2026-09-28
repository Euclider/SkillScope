# Phase3 U5 editor visibility recovery (2026-09-27 UTC)

> 此文记录 **v10 未执行的候选方案**；在任何外部编辑 API 调用前，用户将
> 编辑证据改为窗口终点当次验证的所有相关失败轨迹。实际执行协议见
> [EDITOR_CURRENT_POLICY_EVIDENCE_V11.md](EDITOR_CURRENT_POLICY_EVIDENCE_V11.md)。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: experiment protocol amendment
- Origin Date: 2026-09-27
- Verification Status: coverage rule and offline tests verified; external U5 edit and gate pending
- Version Label: phase3_candidate_only_editor_v10

The `readout_d` U0→U5 prediction completed with 5,351 decisions and exact
chosen-token parity. Evolution then stopped **before any editor API call**:
the old request put all 37 current skill bodies plus eight evidence trajectories
into the gpt-5.5 prompt (local estimate 92,519), crossing the old 64,000
local input guard. The old guard was an implementation limit, not a paper
requirement. The U5 legacy selection had chosen three skills; including the
other 34 skill bodies would weaken the distinction between branch selectors.

## Corrected common six-arm protocol

- At each window, each readout arm ranks naturally supported current-version
  skills by its own pre-registered score and exposes only its **top five** to
  the shared editor. The editor can spend at most **three mutation units**;
  five candidates are not five mandatory edits. The exact fifth-window
  selection is preserved in `selection-v10.json`, leaving the old top-three
  `selection.json` untouched.
- The failure-driven baseline has **no skill ranking**: it exposes the full
  active bank body and up to eight complete failed train/Seen trajectories,
  sampled across task categories. The same editor may ADD, MODIFY, DELETE,
  MERGE or NOOP under the same three-unit mutation budget. This is a
  **SkillRL-style failure-driven, shared-editor adaptation**, not the pinned
  official updater, which supplies all existing *titles* and truncated
  failures while mainly adding skills. The formerly proposed failure-count
  top-three rule was never run and is not part of this experiment.
- Readout editor evidence is at most eight complete **failed** train/Seen
  trajectories containing at least one of the arm's top-five skills. First
  reserve one trajectory for each candidate with observed failed evidence
  (one trajectory may cover several candidates), choosing the highest total
  top-five call count with trajectory-ID tie break. Then fill remaining slots
  by the same global call-count ranking. No post-update
  utility label is supplied. The editor may target only supplied candidate
  IDs/versions for MODIFY, DELETE,
  or MERGE; ADD and NOOP remain possible. The same mutation-unit budget,
  editor model, gate, and evidence count limit apply to all six arms.
- The **runtime router continues to use the complete active skill bank** in
  every arm. Candidate-only readout visibility changes editor inputs, not
  policy conditioning, skill routing, or the active bank itself. Because
  readout editors do not see other bank members, an ADD proposal cannot be
  guaranteed semantically unique to unseen skill text; record this limitation.
- Remove the local editor input-token rejection and all automatic trajectory
  excerpting. Keep input-token estimates, provider usage, API-call budget,
  output-token limit, and sanitized request ledger for cost accounting. The
  historical `editor.max_input_tokens=64000` value remains in the frozen
  runtime/API profile for provenance but is **not enforced**. Router input
  protection is unchanged. The provider may still impose its own context
  limit; such failure is logged and never silently retried.

The original U5 `source.json`, `selection.json`, `evidence.json`, readout,
U4/U5 checkpoints, and failed-run log are preserved. The corrected readout
artifacts use `selection-v10.json`, `evidence-v10.json`, and
`editor_evidence-v10.json`, leaving those originals untouched. Offline U5
inspection found 704 unique training/Seen trajectories, 511 failures, and 343
failures invoking at least one top-five candidate. The approved coverage rule
selects eight full failed trajectories (400 steps), covers all five candidates,
and has a local request estimate of 56,293 input tokens. No API call has yet
been made for this variant. A representative estimate using the existing
`readout_d` episodes but the failure-baseline input rule is 98,372 tokens
for 37 bodies and eight complete failed train/Seen trajectories (400 steps);
it is **not** a measured baseline branch cost. A token-saving claim must use
actual six-arm usage and must attribute savings to this intentionally
different editor input scope, not to readout computation alone.

No RL update or readout needs replay. U4 may be
retired only after the U5 edit/gate event and running
summary are sealed and U5 native checkpoint passes structural checks.

This amendment was made before any Phase3 editor request. It was triggered
by a technical failure and the user's explicit protocol clarification,
not held-out performance; disclose it in the paper appendix.
