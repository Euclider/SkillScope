"""Engineering admission based on measured whole RL cost, without gold labels."""
import math
import re


def update_router_cost(performance, training_log):
    timings = re.findall(r'timing_s/step:([0-9.]+)', training_log)
    if not timings or performance['step_calls'] < 2:
        raise ValueError('Complete actual RL timing and multiple routing calls are required')
    seconds = float(timings[-1]); router = float(performance['router_seconds'])
    if not math.isfinite(seconds) or seconds <= 0 or not 0 <= router <= seconds:
        raise ValueError('Invalid routing/update timing')
    fraction = router/seconds
    if fraction > .10:
        raise ValueError('Router exceeds10% of the actual complete RL update')
    return {'metric': 'router_share_of_complete_RL_update', 'threshold': .10, 'fraction': fraction,
            'router_seconds': router, 'update_seconds': seconds, 'passed': True,
            'rollout_fraction_also_reported': performance.get('router_fraction_of_observed_rollout')}
