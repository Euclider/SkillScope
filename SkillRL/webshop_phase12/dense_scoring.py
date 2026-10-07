"""Score the exact recorded dense actor inputs, including all response slots."""
import torch
from verl.utils.model import compute_position_id_with_mask


def recorded_inputs(tensors,index):
    return {key:tensors[key][index:index+1].clone() for key in ('input_ids','attention_mask','position_ids')}


def control_inputs(tensors,index,prompt_ids,*,pad_token_id):
    length=tensors['prompts'].shape[-1]
    if not prompt_ids or len(prompt_ids)>length:raise ValueError('Control prompt outside recorded dense budget')
    inputs=recorded_inputs(tensors,index)
    if inputs['position_ids'].ndim!=2:raise ValueError('WebShop dense control requires text-only positions')
    inputs['input_ids'][:,:length]=pad_token_id
    inputs['input_ids'][:,length-len(prompt_ids):length]=torch.as_tensor(prompt_ids,dtype=inputs['input_ids'].dtype)
    inputs['attention_mask'][:,:length]=0
    inputs['attention_mask'][:,length-len(prompt_ids):length]=1
    prompt_positions=compute_position_id_with_mask(inputs['attention_mask'][:,:length])
    inputs['position_ids'][:,:length]=prompt_positions
    response_length=inputs['input_ids'].shape[-1]-length
    inputs['position_ids'][:,length:]=prompt_positions[:,-1:]+torch.arange(1,response_length+1)
    return inputs


@torch.inference_mode()
def score_dense(model,inputs,response_length,loss_mask,*,trim_padding=False):
    device=next(model.parameters()).device
    forwarded={key:value.to(device) for key,value in inputs.items()}
    if trim_padding:
        from verl.workers.actor.padded_forward import trim_common_left_padding
        values=trim_common_left_padding(forwarded['input_ids'],forwarded['attention_mask'],
                                        forwarded['position_ids'],response_length)
        forwarded=dict(zip(('input_ids','attention_mask','position_ids'),values))
    if forwarded['position_ids'].ndim==3:forwarded['position_ids']=forwarded['position_ids'].transpose(0,1)
    output=model(**forwarded,use_cache=False,logits_to_keep=response_length+1)
    logits=output.logits[0,-response_length-1:-1]
    if logits.shape[0]!=response_length or loss_mask.numel()!=response_length:
        raise ValueError('Dense readout response positions disagree')
    return torch.log_softmax(logits[loss_mask.to(device)].float(),dim=-1)
