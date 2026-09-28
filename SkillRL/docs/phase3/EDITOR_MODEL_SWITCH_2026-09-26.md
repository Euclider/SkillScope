# Phase3 editor-model amendment (2026-09-26 UTC)

The approved six-arm run keeps its RL seed, frozen initial bank, local router,
update schedule, evidence budget, editor prompt/schema, mutation budget, gateway
URL, and editor credential unchanged. Only the shared editor model changes from
`o3` to `gpt-5.5`; Chat Completions uses `reasoning_effort=medium` and strict
JSON-schema output as before. No policy is used as the router.

The original `assets-launch-v1/setting.json` is a hashed v3 record and remains
unchanged. The executable setting for newly prepared runs is v4. For the active
v6 run, `editor_model_switch.json` in the experiment root is the required
amendment; the actual requested/response model and usage remain in each branch's
`editor.sqlite3` and event accounting. Do not describe the v3 file alone as
the final editor setting.

At the request, the first branch's U0→U5 training subprocess was still running
and its editor ledger contained zero requests. The old orchestration parent was
paused without signalling the training subprocess or its workers. A one-shot
handoff waits for a successful U5 training exit, eight-shard checkpoint, and U5
metrics. Only then does it end the old parent, preserve the old API profile in
`profile_migrations`, switch the zero-attempt ledger profile, write the amendment,
and resume the six-arm queue from the U5 checkpoint. If any check fails, it
stops closed; no old editor call, fallback model, training rerun, or automatic
API retry is permitted. This is a checkpoint-boundary orchestration restart,
not a restart of the in-flight RL updates.

Official OpenAI documentation lists `gpt-5.5` for Chat Completions with medium
reasoning and structured outputs. A minimal real request through the configured
third-party gateway returned model `gpt-5.5`, a complete response, and valid JSON.
This verifies the tested request shape, not all future editor outputs or gateway
model identity. The gateway's actual response model and request ID are recorded
per call.

Validation before handoff: `pytest -q tests/phase3` passed (128 tests), including
both historical o3 replay and gpt-5.5 request/cache behavior, an unused-ledger
migration, and rejection of migration after any API reservation. No Phase3
performance result is implied by this engineering validation.

Update, 2026-09-27 UTC: the v6 U5 training child failed from GPU router OOM
before producing a checkpoint, so this handoff never executed. Its editor
ledger remained unused and pinned to o3. The fresh v7 preparation pins
gpt-5.5 directly and does not claim to have resumed v6 U1–U4 weights. See
[ROUTER_GPU_MEMORY_RECOVERY_V7.md](ROUTER_GPU_MEMORY_RECOVERY_V7.md).
