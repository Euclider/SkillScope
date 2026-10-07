import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from types import SimpleNamespace


def test_generation_groups_share_exact_prompt_length_and_restore_every_row():
    from webshop_phase12.accelerated import generation_groups
    mask=torch.tensor([[0,0,1,1,1,1],[0,0,0,0,1,1],[0,0,1,1,1,1],
                       [0,0,0,0,1,1],[0,0,0,1,1,1]])
    assert generation_groups(mask,2)==[[1,3],[4],[0,2]]
    assert generation_groups(mask,1)==[[1],[3],[4],[0],[2]]
    assert sorted(i for g in generation_groups(mask,3) for i in g)==list(range(5))


@pytest.mark.parametrize('mask,limit',[(torch.zeros(1,4),2),(torch.tensor([[0,1,0,1]]),2),
                                     (torch.ones(2,4),0)])
def test_generation_groups_reject_empty_or_non_left_padding(mask,limit):
    from webshop_phase12.accelerated import generation_groups
    with pytest.raises(ValueError):generation_groups(mask,limit)


def test_new_forward_contract_rejects_partially_enabled_trim():
    from webshop_phase12.accelerated import forward_contract
    config={'webshop_phase12':{'forward_contract':'hf-exact-length-v2'},
            'actor_rollout_ref':{'actor':{'trim_common_padding':True,'ppo_micro_batch_size_per_gpu':1},
                                'ref':{'trim_common_padding':True,'log_prob_micro_batch_size_per_gpu':1},
                                'rollout':{'trim_common_padding':True}}}
    assert forward_contract(config)=='hf-exact-length-v2'
    config['actor_rollout_ref']['ref']['trim_common_padding']=False
    with pytest.raises(ValueError):forward_contract(config)
    assert forward_contract({})=='hf-dense-v1'


def test_trimmed_scoring_preserves_all_response_slots_and_original_tensors():
    from webshop_phase12.dense_scoring import score_dense
    inputs={'input_ids':torch.tensor([[99,99,11,12,13,14,31,32,33,99]]),
            'attention_mask':torch.tensor([[0,0,1,1,1,1,1,1,1,0]]),
            'position_ids':torch.tensor([[0,0,0,1,2,3,4,5,6,7]])}
    before={k:v.clone() for k,v in inputs.items()}
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.));self.seen=None
        def forward(self,input_ids,attention_mask,position_ids,use_cache,logits_to_keep):
            self.seen=input_ids.clone()
            logits=torch.zeros(1,input_ids.shape[-1],3);logits[:,:,1]=input_ids.float()
            return SimpleNamespace(logits=logits[:,-logits_to_keep:])
    model=Model()
    result=score_dense(model,inputs,4,torch.tensor([True,False,True,False]),trim_padding=True)
    assert model.seen.tolist()==[[11,12,13,14,31,32,33,99]]
    assert result.shape==(2,3)
    assert all(torch.equal(inputs[k],before[k]) for k in inputs)


def test_trimmed_hf_generation_preserves_dense_storage_and_original_row_order():
    from verl import DataProto
    from verl.workers.rollout.hf_rollout import HFRollout
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.))
            self.config=SimpleNamespace(model_type='test');self.seen=[]
        def generate(self,input_ids,attention_mask,generation_config,**kwargs):
            self.seen.append((input_ids.shape[-1],len(input_ids)))
            assert attention_mask.all()
            response=torch.stack([input_ids[:,-1]+1,torch.full_like(input_ids[:,-1],2)],dim=1)
            return SimpleNamespace(sequences=torch.cat([input_ids,response],dim=1))
    ids=torch.tensor([[0,0,0,0,3,4],[0,0,0,5,6,7],[0,0,0,0,8,9]])
    mask=(ids!=0).long();positions=(mask.cumsum(-1)-1).clamp(min=0)
    batch=DataProto.from_single_dict({'input_ids':ids,'attention_mask':mask,'position_ids':positions,
                                     'row_id':np.array(['a','b','c'],dtype=object)})
    batch.meta_info.update(eos_token_id=2,pad_token_id=0,do_sample=False)
    config=OmegaConf.create({'micro_batch_size':8,'trim_common_padding':True,'do_sample':False,
                            'temperature':1.,'response_length':4,'n':1})
    model=Model();result=HFRollout(model,config).generate_sequences(batch)
    assert model.seen==[(2,2),(3,1)]
    assert torch.equal(result.batch['prompts'],ids)
    assert result.batch['responses'].tolist()==[[5,2,0,0],[8,2,0,0],[10,2,0,0]]
    assert result.batch['input_ids'].shape==(3,10)
    assert torch.equal(batch.batch['input_ids'],ids)


def test_compact_policy_inputs_keep_complete_prompt_and_overflow_guard():
    from transformers import AutoTokenizer
    from webshop_phase12.assets import BASE_MODEL
    from webshop_phase12.prompts import policy_inputs
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,local_files_only=True)
    dense=policy_inputs(tokenizer,'Find a blue shirt.',device='cpu',budget=128)
    compact=policy_inputs(tokenizer,'Find a blue shirt.',device='cpu',budget=128,compact=True)
    assert compact['attention_mask'].all()
    assert compact['input_ids'][0].tolist()==dense['input_ids'][0,dense['attention_mask'][0].bool()].tolist()
    with pytest.raises(ValueError):policy_inputs(tokenizer,'Find a blue shirt.',device='cpu',budget=8,compact=True)


def test_accelerated_hydra_config_keeps_registered_rl_scale(monkeypatch):
    from hydra import compose,initialize_config_dir
    from pathlib import Path
    from webshop_phase12.accelerated import forward_contract
    from webshop_phase12.assets import ROOT
    monkeypatch.setenv('WEBSHOP_BASE_MODEL','/model/qwen')
    with initialize_config_dir(config_dir=str(ROOT/'verl/trainer/config'),version_base=None):
        cfg=compose(config_name='webshop54_phase12_accel_v2',overrides=[
            'webshop_run.seed=404','webshop_run.prepared=/prepared','webshop_run.run_root=/run'])
    assert forward_contract(cfg)=='hf-exact-length-v2'
    assert cfg.data.max_prompt_length==16384 and cfg.data.max_response_length==512
    assert cfg.env.rollout.n==8 and cfg.webshop_run.tasks_per_update==128
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size==128
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu==1
    assert cfg.actor_rollout_ref.rollout.micro_batch_size==16
