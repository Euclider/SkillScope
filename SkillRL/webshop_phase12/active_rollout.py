"""Generate only live WebShop rows; preserve the complete trajectory layout."""
import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor,unpad_dataproto


def generate_active(group,prompts,active_masks,*,pad_token_id):
    indices=np.flatnonzero(active_masks)
    if not len(indices):raise ValueError('No live environments to generate')
    selected=prompts.select_idxs(indices)
    padded,pad_size=pad_dataproto_to_divisor(selected,group.world_size)
    output=unpad_dataproto(group.generate_sequences(padded),pad_size)
    if len(indices)==len(prompts):return output
    tensors={}
    for key,tensor in output.batch.items():
        fill=pad_token_id if key in ('prompts','responses','input_ids') else 0
        tensors[key]=torch.full((len(prompts),*tensor.shape[1:]),fill,dtype=tensor.dtype,device=tensor.device)
    length=prompts.batch['input_ids'].shape[-1]
    tensors['prompts'][:]=prompts.batch['input_ids'].to(tensors['prompts'].device)
    for key in ('input_ids','attention_mask','position_ids'):
        tensors[key][...,:length]=prompts.batch[key].to(tensors[key].device)
    for key,tensor in output.batch.items():tensors[key][indices]=tensor
    if output.non_tensor_batch:raise ValueError('Active-only native HF output must be tensor-only')
    return DataProto(batch=TensorDict(tensors,batch_size=len(prompts)),meta_info=output.meta_info)
