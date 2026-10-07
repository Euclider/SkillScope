import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto


def test_finished_episodes_skip_generation_and_keep_original_row_alignment():
    from webshop_phase12.active_rollout import generate_active
    inputs=DataProto(batch=TensorDict({'input_ids':torch.tensor([[1,2],[3,4],[5,6],[7,8]]),
        'attention_mask':torch.ones(4,2,dtype=torch.long),'position_ids':torch.tensor([[0,1]]*4)},batch_size=4))
    class CPUGenerationGroup:
        world_size=2
        def generate_sequences(self,prompts):
            ids=prompts.batch['input_ids'];responses=ids[:,:1]+10
            # Real tensor values depend on input rows, so a wrong scatter/order
            # fails independently of how the wrapper implements its mapping.
            return DataProto(batch=TensorDict({'prompts':ids,'responses':responses,
                'input_ids':torch.cat((ids,responses),-1),
                'attention_mask':torch.ones(len(ids),3,dtype=torch.long),
                'position_ids':torch.tensor([[0,1,2]]*len(ids))},batch_size=len(ids)))
    result=generate_active(CPUGenerationGroup(),inputs,np.array([True,False,True,False]),pad_token_id=99)
    assert result.batch['responses'].tolist()==[[11],[99],[15],[99]]
    assert result.batch['prompts'].tolist()==[[1,2],[3,4],[5,6],[7,8]]
    assert result.batch['attention_mask'][:,-1].tolist()==[1,0,1,0]
    assert result.batch['input_ids'][2].tolist()==[5,6,15]


def test_hf_microbatch_limit_holds_for_partial_active_batches():
    from verl.workers.rollout.hf_rollout import HFRollout
    from omegaconf import OmegaConf
    rollout=HFRollout(torch.nn.Linear(1,1),OmegaConf.create({'micro_batch_size':4}))
    sizes=[]
    def cpu_forward(prompts):
        sizes.append(len(prompts));return prompts
    rollout._generate_minibatch=cpu_forward
    prompts=DataProto(batch=TensorDict({'row':torch.arange(10)[:,None]},batch_size=10))
    output=rollout.generate_sequences(prompts)
    assert output.batch['row'].flatten().tolist()==list(range(10))
    assert sizes==[4,4,2]
