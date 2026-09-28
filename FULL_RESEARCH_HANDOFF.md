# SkillScope complete research handoff — 2026-09-28

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: reproducibility handoff
- Verification Status: source inventory, secret scan, archive/hash audit, and local CPU tests; no non-ALFWorld end-to-end validation
- Version Label: research-handoff-2026-09-28

The [release](https://github.com/Euclider/SkillScope/releases/tag/research-handoff-2026-09-28)
contains a public, research-oriented snapshot of the current working tree.
`RELEASE_MANIFEST.json` enumerates every published file with its SHA256 and
size. The GitHub `main` branch holds the same published files at their project
paths, without deleting older repository history.

This handoff expands the earlier Phase1–2 source-only archive to include the
root-level dated research notes, complete source documentation and figures,
frozen skill-bank content, experiment preparation/protocol records, selected
report tables, aggregate metrics, and result summaries. The Phase1–2
cross-benchmark setup and adapter contract are in
[`SkillRL/docs/phase12/CROSS_BENCHMARK_PORTABILITY.md`](SkillRL/docs/phase12/CROSS_BENCHMARK_PORTABILITY.md).

The package deliberately excludes model snapshots and weights, external
benchmark datasets, raw trajectories, optimizer/probability tensors, runtime
locks/caches, and credentials. One third-party installation note is published
with a credential-like string redacted **only in the public copy**; its path is
listed under `redacted_paths` in the manifest. The original local file is not
changed. The many historical `SkillRL/artifacts/` files omitted here are
runtime/raw evidence, not a claim that they never existed. A separate governed
data archive would be needed for full raw-data reproduction.

No new benchmark adapter, model weights, datasets, or RL run is bundled.
ALFWorld remains the reference implementation; other benchmarks require their
own frozen skill bank, interaction/replay adapter, and paired-utility protocol.
