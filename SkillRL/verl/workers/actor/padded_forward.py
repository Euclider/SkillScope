"""Opt-in dense text forward optimizations; caller keeps the original batch."""
from contextlib import nullcontext


def gradient_sync_context(module, *, enabled, last):
    if enabled and not last:
        if not hasattr(module, 'no_sync'):
            raise ValueError('Deferred gradient synchronization requires a distributed no_sync() module')
        return module.no_sync()
    return nullcontext()


def prompt_bucket_width(length,multiple=1):
    if length<1 or multiple<1:raise ValueError('Prompt length and bucket multiple must be positive')
    return ((int(length)+int(multiple)-1)//int(multiple))*int(multiple)


def trim_common_left_padding(input_ids, attention_mask, position_ids, response_length,*,prompt_multiple=1,prompt_width=None):
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
    width=prompt_bucket_width(prompt_length-first,prompt_multiple) if prompt_width is None else int(prompt_width)
    if width<prompt_length-first or width>prompt_length:
        raise ValueError('Canonical prompt width would truncate facts or exceed stored input')
    first=prompt_length-width
    return input_ids[:, first:], attention_mask[:, first:], position_ids[..., first:]
