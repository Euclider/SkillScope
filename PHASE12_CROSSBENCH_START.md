# Phase1–2 cross-benchmark source handoff (2026-09-28)

The current portable source/environment snapshot is published in the
[`phase12-crossbench-2026-09-28` release](https://github.com/Euclider/SkillScope/releases/tag/phase12-crossbench-2026-09-28).
For setup, method invariants, and the benchmark-adapter checklist, read
[`SkillRL/docs/phase12/CROSS_BENCHMARK_PORTABILITY.md`](SkillRL/docs/phase12/CROSS_BENCHMARK_PORTABILITY.md).

The archive contains source, tests, and the pinned vLLM/PyTorch environment;
it excludes model weights, benchmark data, checkpoints, trajectories, reports,
and credentials. The ALFWorld pipeline is a reference implementation, not a
validated launcher for another benchmark. Keep the supplied release manifest
and its hashes with each new experiment.
