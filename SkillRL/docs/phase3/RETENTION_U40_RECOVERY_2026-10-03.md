# U40 retention-aware recovery — 2026-10-03

## Scope and cause

User requested repair and continuation. The prior queue stopped at
13:17:52 UTC while preparing SkillRL U40→U45. U40 RL, export, readout,
editor and gate had completed; three edits were accepted. No U41 update
had begun. Reward D_sign_balance remained at U20.

The U40 retention receipt confirms that `models/u0035` and the U35 native
checkpoint were intentionally retired after U40 sealing. The recovery
launcher nevertheless rehashed a historical evidence entry inside that
directory (`phase2_export.json`), causing FileNotFoundError. This was a
control-plane retention/authorization conflict, not an RL or OOM failure.

## Repair and evidence boundaries

- Missing historical export metadata is excused only when explicitly
  registered with a checksummed retention receipt. The path must be the
  exact old endpoint export of an in-scope branch; the receipt must match
  its successor's completed event and endpoint identity, and the retired
  model directory must be absent. Changed present files still fail.
- The original missing file's hash remains in the authorization. Its
  deleted bytes cannot be rehashed; the sealed retention chain explains
  that absence, rather than claiming recovery of those bytes.
- New authority evidence contains persistent event/metric/receipt files,
  not metadata under U40 model directories scheduled for later retirement.
  Latest native checkpoints and exports are checked before launch; normal
  orchestration continues to check model identities and bank lineage.
- Original launcher bytes and all prior logs/authorizations are retained.
  The failed U40→U45 log is untouched; the resumed subprocess has a new log.
- No training implementation, optimizer settings, speed options, router,
  readout definitions, editor prompt/budgets or gate settings changed.
  No API retry was authorized; completed edits and results are reused.

## Verification and execution

The regression first failed with the same missing-export exception.
After repair, Phase3 plus stable-direction tests: **204 passed, 1 skipped**.
The 12 retention checks cover both recovery launchers, including unknown
missing files, changed receipts, unsealed events and changed present files.

Full `pytest -q tests --maxfail=1` remains blocked during collection at
`tests/e2e/sft/test_sp_loss_match.py`: optional `flash_attn` is absent.
No dependency was installed or running training environment changed.

Both SkillRL U40 and Reward U20 native checkpoints passed the structural
check: 8 model, 8 optimizer and 8 extra-state shards each. Prior accepted
training and readout source hashes passed validation.

New queue:
`/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v7-20260927/recovery-two-arm-u50-retention-u40-v1`

Launcher: `scripts/resume_two_arm_u50_retention.py`; initial PID 3954994.
Order remains SkillRL U40→U50 plus seen/unseen evaluation, followed by
Reward D_sign_balance U20→U50 plus evaluation. All completed results are
reused. Queue launch alone is not evidence that another RL update finished.
