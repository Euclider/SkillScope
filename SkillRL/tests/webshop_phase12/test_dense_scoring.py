import torch
import pytest
from types import SimpleNamespace


def recorded():
    return {'input_ids':torch.tensor([[99,99,11,12,13,14,31,32,99,99]]),
        'prompts':torch.tensor([[99,99,11,12,13,14]]),
        'responses':torch.tensor([[31,32,99,99]]),
        'attention_mask':torch.tensor([[0,0,1,1,1,1,1,1,0,0]]),
        'position_ids':torch.tensor([[0,0,0,1,2,3,4,5,6,7]])}


def test_control_retains_full_response_and_recorded_padding_width():
    from webshop_phase12.dense_scoring import control_inputs
    tensors=recorded();before=tensors['input_ids'].clone()
    inputs=control_inputs(tensors,0,[22,23,24],pad_token_id=99)
    assert inputs['input_ids'].tolist()==[[99,99,99,22,23,24,31,32,99,99]]
    assert inputs['attention_mask'].tolist()==[[0,0,0,1,1,1,1,1,0,0]]
    assert inputs['position_ids'].tolist()==[[0,0,0,0,1,2,3,4,5,6]]
    assert torch.equal(tensors['input_ids'],before)
    with pytest.raises(ValueError):control_inputs(tensors,0,list(range(7)),pad_token_id=99)


def test_dense_readout_keeps_masked_response_tokens_in_the_conditioning_prefix():
    from webshop_phase12.dense_scoring import recorded_inputs,score_dense
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.))
        def forward(self,input_ids,attention_mask,position_ids,use_cache,logits_to_keep):
            logits=torch.zeros(1,input_ids.shape[-1],3)
            logits[:,:,1]=input_ids.float()
            return SimpleNamespace(logits=logits[:,-logits_to_keep:])
    tensors=recorded();tensors['input_ids'][0,8]=33;tensors['attention_mask'][0,8]=1
    result=score_dense(Model(),recorded_inputs(tensors,0),4,torch.tensor([True,False,True,False]))
    expected=torch.log_softmax(torch.tensor([[0.,14.,0.],[0.,32.,0.]]),-1)
    assert torch.allclose(result,expected)


def test_paired_policy_uses_registered_dense_prompt_budget_without_truncation():
    from transformers import AutoTokenizer
    from webshop_phase12.assets import BASE_MODEL
    from webshop_phase12.prompts import policy_inputs
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,local_files_only=True)
    prompt='Find a blue shirt.'
    chat=tokenizer.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    expected=tokenizer.encode(chat,add_special_tokens=False)
    inputs=policy_inputs(tokenizer,prompt,device='cpu',budget=128)
    assert inputs['input_ids'].shape==(1,128)
    assert inputs['input_ids'][0,inputs['attention_mask'][0].bool()].tolist()==expected
    with pytest.raises(ValueError):policy_inputs(tokenizer,prompt,device='cpu',budget=8)
