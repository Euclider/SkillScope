"""Opt-in dense text forward optimizations; caller keeps the original batch."""
from contextlib import nullcontext


def gradient_sync_context(module, *, enabled, last):
    if enabled and not last:
        if not hasattr(module, 'no_sync'):
            raise ValueError('Deferred gradient synchronization requires a distributed no_sync() module')
        return module.no_sync()
    return nullcontext()


def trim_common_left_padding(input_ids, attention_mask, position_ids, response_length):
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError('Expected dense 2D text input and attention mask')
    if position_ids.shape[-1] != input_ids.shape[-1]:
        raise ValueError('Position length differs from input')
    prompt_length = input_ids.shape[-1] - response_length
    if prompt_length < 1:
        raise ValueError('A response needs at least one prompt position')
    occupied = attention_mask[:, :prompt_length].bool().any(dim=0)
    if not occupied.any():
        raise ValueError('Cannot score an empty prompt')
    first = int(occupied.nonzero()[0].item())
    return input_ids[:, first:], attention_mask[:, first:], position_ids[..., first:]
