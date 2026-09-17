# 0.5B Phase-I smoke validation report

Run ID: `smoke-0p5b-20260825`

## Result

The engineering and archive pipeline passed, but the 0.5B capability gate
failed. All 36 expected paired-condition episodes and 36 full trajectory files
were archived without duplicate keys or missing step fields.

- Episodes: 36 across 12 independent games and 6 contexts
- Environment steps: 1,080
- Successes: 0/36 (0%)
- Mean trajectory length: 30/30 steps
- Max-step failures: 36/36 (100%)
- Invalid-action rate: 32.87% (gate requires no more than 30%)
- Action-format compliance: 67.13%
- Archive integrity: pass
- Fixed probe states extracted: 48, eight per context

The zero success floor makes every preliminary success margin zero. These
margins are not evidence that Skill injection has no effect; the model did not
solve any task under any condition, so success cannot resolve condition-level
differences.

## Condition breakdown

| Condition | Episodes | Success | Invalid actions / steps | Invalid rate | Format compliance | Target injected | Mean distinct Skills |
|---|---:|---:|---:|---:|---:|---:|---:|
| FULL_BANK | 12 | 0% | 92 / 360 | 25.56% | 74.44% | 100% | 2.0 |
| MINUS_SKILL | 12 | 0% | 82 / 360 | 22.78% | 77.22% | 0% | 1.0 |
| NO_SKILL | 12 | 0% | 181 / 360 | 50.28% | 49.72% | 0% | 0.0 |

Skill text is associated with substantially better action-format compliance in
this smoke sample, especially relative to NO_SKILL. This is diagnostic only:
two games and one evaluation seed per context are insufficient for causal or
inferential claims.

## Invalid-action rate by context

| Context | Invalid actions / steps | Invalid rate | Successes |
|---|---:|---:|---:|
| clean | 45 / 180 | 25.00% | 0/6 |
| cool | 71 / 180 | 39.44% | 0/6 |
| heat | 56 / 180 | 31.11% | 0/6 |
| look_at_obj_in_light | 50 / 180 | 27.78% | 0/6 |
| pick_and_place | 73 / 180 | 40.56% | 0/6 |
| pick_two | 60 / 180 | 33.33% | 0/6 |

## Routing incident and recovery

Natural-language annotations used `cooked` for heat and `turning the lamp on`
for look-at tasks. The original template router rejected those games before
writing an invalid record. Synonym routing was corrected and regression-tested;
the evaluator then resumed only the missing unique keys. No completed trajectory
was overwritten. Exact initial/recovery worktree fingerprints and incident
details are stored in `completion_manifest.json`.

## Interpretation

According to the frozen Phase-I protocol, the 0.5B result is suitable for
validating the engineering chain only. It should not be used to reject the
research hypothesis. The predefined next capability step is to run the same
smoke design with Qwen2.5-1.5B-Instruct before any RL update or action-flip
claim.
