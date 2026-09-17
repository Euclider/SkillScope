# Paired 0.5B versus 1.5B smoke comparison

All 36 records were paired on identical game, context, Skill, condition,
environment seed, and evaluation seed.

## Overall comparison

| Metric | 0.5B | 1.5B | 1.5B minus 0.5B |
|---|---:|---:|---:|
| Success rate | 0.00% | 0.00% | 0.00 pp |
| Invalid-action rate | 32.87% | 30.74% | -2.13 pp |
| Format compliance | 67.13% | 69.26% | +2.13 pp |
| Mean completion tokens per step | 120.23 | 69.05 | -51.18 |

The 1.5B model is more concise and modestly more format-compliant, but neither
model demonstrates ALFWorld task success.

## Invalid-action change by condition

| Condition | 0.5B | 1.5B | Change |
|---|---:|---:|---:|
| FULL_BANK | 25.56% | 20.56% | -5.00 pp |
| MINUS_SKILL | 22.78% | 21.67% | -1.11 pp |
| NO_SKILL | 50.28% | 50.00% | -0.28 pp |

## Invalid-action change by context

| Context | 0.5B | 1.5B | Change |
|---|---:|---:|---:|
| clean | 25.00% | 22.22% | -2.78 pp |
| cool | 39.44% | 37.22% | -2.22 pp |
| heat | 31.11% | 42.22% | +11.11 pp |
| look_at_obj_in_light | 27.78% | 15.56% | -12.22 pp |
| pick_and_place | 40.56% | 35.00% | -5.56 pp |
| pick_two | 33.33% | 32.22% | -1.11 pp |

## Conclusion

Increasing model size from 0.5B to 1.5B does not remove the task-success floor.
The result supports engineering validity and suggests Skill text affects output
format, but success-based Skill utility and RL-induced action flips are not
identifiable under the current policy/prompt/action setup.
