# Phase-I environment

Create and activate the environment with:

```bash
conda env create --override-channels -f environment.yml
conda activate skill-RL
```

The smoke and fallback models must be stored at
`/home/wangyifan/model/Qwen2.5-{0.5B,1.5B}-Instruct` and
`ALFWORLD_DATA` must point to the directory created by `alfworld-download`.
Run `python -m phase1.preflight` before any probe, rollout, or RL update.

`phase1-constraints.txt` is the human-reviewed direct dependency lock;
`requirements-lock.txt` is the exact transitive `pip freeze` captured from the
prepared environment on 2026-08-25.
