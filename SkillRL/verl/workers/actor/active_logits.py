"""Single-row output projection at real response positions; context stays dense."""
import torch


def response_position_indices(input_length, response_mask):
    if response_mask.ndim != 2 or response_mask.shape[0] != 1:
        raise ValueError('Active-position projection requires microbatch one')
    response_length = response_mask.shape[1]
    if response_length < 1 or input_length <= response_length:
        raise ValueError('Response must have a conditioning prompt')
    active = response_mask[0].bool().nonzero().flatten()
    predecessors = active + input_length - response_length - 1
    if not len(active):
        predecessors = torch.tensor([input_length-response_length-1], device=response_mask.device)
    return predecessors, active


def scatter_response_values(values, active, response_length):
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError('Expected one row of active response values')
    if not len(active):
        return values.new_zeros((1, response_length)) + values.sum()*0
    if values.shape[1] != len(active):
        raise ValueError('Active positions and projected values differ')
    return values.new_zeros((1, response_length)).scatter(1, active[None], values)
