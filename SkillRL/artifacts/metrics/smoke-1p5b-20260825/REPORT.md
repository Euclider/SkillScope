# 1.5B Phase-I smoke validation report

Run ID: `smoke-1p5b-20260825`

## Result

The run and archive pipeline passed, but the 1.5B capability gate failed.

- Episodes: 36 across the same 12 games and 6 contexts as the 0.5B run
- Environment steps: 1,080
- Successes: 0/36 (0%)
- Mean trajectory length: 30/30 steps
- Max-step failures: 36/36 (100%)
- Invalid-action rate: 30.74% (gate requires no more than 30%)
- Action-format compliance: 69.26%
- Archive integrity: pass
- Fixed probe states extracted: 48, eight per context

The zero success floor makes every preliminary success margin zero. No
Skill/context pair is eligible for a success-based pre/post action-flip study
under this smoke configuration.

## Condition breakdown

| Condition | Episodes | Success | Invalid actions / steps | Invalid rate | Format compliance | Target injected | Mean distinct Skills |
|---|---:|---:|---:|---:|---:|---:|---:|
| FULL_BANK | 12 | 0% | 74 / 360 | 20.56% | 79.44% | 100% | 2.0 |
| MINUS_SKILL | 12 | 0% | 78 / 360 | 21.67% | 78.33% | 0% | 1.0 |
| NO_SKILL | 12 | 0% | 180 / 360 | 50.00% | 50.00% | 0% | 0.0 |

Skill injection again correlates with substantially better output formatting,
but it does not produce task success in this sample. This is diagnostic only
and is not a causal estimate of Skill utility.

## Invalid-action rate by context

| Context | Invalid actions / steps | Invalid rate | Successes |
|---|---:|---:|---:|
| clean | 40 / 180 | 22.22% | 0/6 |
| cool | 67 / 180 | 37.22% | 0/6 |
| heat | 76 / 180 | 42.22% | 0/6 |
| look_at_obj_in_light | 28 / 180 | 15.56% | 0/6 |
| pick_and_place | 63 / 180 | 35.00% | 0/6 |
| pick_two | 58 / 180 | 32.22% | 0/6 |

## Execution note

An unrelated process began using GPU 1 during the pick-two batch. More than
58 GiB remained free, no OOM or evaluator error occurred, and all record keys
and trajectories passed integrity checks. Wall-clock timing is therefore not
used in the 0.5B/1.5B comparison.

## Interpretation

The 1.5B fallback improves action formatting slightly but does not clear the
predefined format gate and does not solve any task. Under the frozen Phase-I
protocol, this smoke result does not support proceeding directly to RL updates
or interpreting success-based Skill margins.
