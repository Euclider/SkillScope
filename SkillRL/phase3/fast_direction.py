"""Identical stable equations without unused legacy KL/JS computation."""
import math
import torch

from phase2.stable_direction import _fp64_rows


def token_signals(old_o, new_o, old_control, new_control, actions, advantages,
                  *, tau_delta=1e-8, tau_c=0., epsilon=1e-12):
    if not all(math.isfinite(x) for x in (tau_delta, tau_c, epsilon)) or epsilon <= 0 or tau_delta < 0:
        raise ValueError('Finite gates and positive epsilon required')
    vectors = (old_o, new_o, old_control, new_control)
    if len({tuple(x.shape) for x in vectors}) != 1 or old_o.ndim != 2:
        raise ValueError('Four aligned token-by-vocabulary distributions are required')
    if actions.shape != advantages.shape or actions.shape != old_o.shape[:1]:
        raise ValueError('Actions and advantages must align to every decision token')
    if not all(torch.isfinite(x).all() for x in (*vectors, advantages)):
        raise ValueError('Non-finite input to reward-directed signal')
    chunks = []
    # Preserve both the original chunk size and its reduction order.
    for first in range(0, len(actions), 16):
        sl = slice(first, first + 16)
        chunks.append(_fp64_rows(*(x[sl] for x in vectors), actions[sl], advantages[sl],
                                 tau_delta=tau_delta, tau_c=tau_c, epsilon=epsilon))
    if not chunks:
        raise ValueError('At least one actual loss token is required')
    return {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}
