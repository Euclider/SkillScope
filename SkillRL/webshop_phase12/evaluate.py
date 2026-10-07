"""First-natural-call anchors and paired old/new, target skill/control replay."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from logicbench_phase12.fixed_state import _load_model
from webshop_phase12.assets import BASE_MODEL, WebshopBank,digest
from webshop_phase12.envs import ShopWorld
from webshop_phase12.prompts import action_list, build_state_prompt, project_action,policy_inputs
from webshop_phase12.visible_state import VisibleMemory,replay_visible
from webshop_phase12.router import RouterClient
from webshop_phase12.accelerated import COMPACT,DENSE,load_forward_contract


def run(seed_dir,new_model,prepared,*,smoke=False):
    if not (seed_dir/'readout/prediction-locked.json').is_file():
        raise ValueError('Readout must be locked before collecting independent outcomes')
    contract=load_forward_contract(seed_dir)
    prediction=json.loads((seed_dir/'readout/prediction-locked.json').read_text())
    if prediction.get('forward_contract',DENSE)!=contract:
        raise ValueError('Readout and paired evaluation forward contracts differ')
    out=seed_dir/'paired_eval';out.mkdir(exist_ok=False)
    spec=json.loads((prepared/'manifest.json').read_text())
    bank=WebshopBank();world=ShopWorld(1);router=RouterClient(f'eval-{seed_dir.name}')
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,local_files_only=True)
    old=_load_model(BASE_MODEL,'cuda');new=_load_model(new_model,'cuda')
    models={'old':old,'new':new};max_steps=6 if smoke else 50
    continuation_seeds=(0,1) if smoke else tuple(range(16))

    def transition(memory,action,obs,info):
        old=memory.admissible
        valid=(action.startswith('search[') and 'search[<your query>]' in old) or action.casefold() in [a.casefold() for a in old]
        memory.transition(action,valid,obs,action_list(info['available_actions']),info['visible_page'])

    @torch.inference_mode()
    def generate(model,prompt,seed):
        inputs=policy_inputs(tokenizer,prompt,device='cuda',compact=contract==COMPACT)
        length=inputs['input_ids'].shape[-1]
        if length>16384:
            raise ValueError('Paired continuation prompt exceeds registered budget')
        torch.manual_seed(seed)
        ids=model.generate(**inputs,do_sample=True,temperature=.7,top_p=1.,max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id)[0,length:]
        return tokenizer.decode(ids,skip_special_tokens=True),int(ids.numel())

    # Reference trajectories use only OLD policy and the unchanged natural router.
    anchors=[]
    with (out/'reference_trajectories.jsonl').open('w') as records:
        for task in spec['eval_ids']:
            obs,info,memory=replay_visible(world,0,task,[],max_steps);history=[];seen=set();prefix=[]
            for step in range(max_steps):
                visible=memory.state()
                bundle=router.route_many([visible])[0];sid=bundle['selected_skill_id'];skill=bank.get(sid)
                prompt=build_state_prompt(visible,skill.payload)
                if sid not in seen:
                    anchor={'task_id':task,'skill_id':sid,'prefix_actions':list(prefix),'observation':obs,
                        'admissible_actions':visible['admissible_actions'],'history':list(history),'anchor_step':step,
                        'payload_sha256':skill.payload_sha256,'environment_state_sha256':world.state_digest(0),
                        'visible_state_sha256':digest(visible)}
                    anchors.append(anchor);seen.add(sid)
                text,ntokens=generate(old,prompt,1000000+task*100+step)
                action,valid=project_action(text);previous=obs
                obs,reward,done,info=world.step_one(0,action)
                transition(memory,action,obs,info)
                records.write(json.dumps({'task_id':task,'step':step,'skill_id':sid,'action':action,'raw_output':text,
                    'reward':reward,'won':info['won'],'response_tokens':ntokens,'valid':valid},ensure_ascii=False)+'\n');records.flush()
                history.append({'observation':previous,'action':action});prefix.append(action)
                if done:break
            print(json.dumps({'reference_task':task,'anchors':len(anchors)}),flush=True)
    (out/'anchors.json').write_text(json.dumps(anchors,ensure_ascii=False)+'\n')

    def continuation(anchor,checkpoint,control,seed):
        task=anchor['task_id'];target=anchor['skill_id']
        obs,info,memory=replay_visible(world,0,task,anchor['prefix_actions'],max_steps)
        if (obs!=anchor['observation'] or action_list(info['available_actions'])!=anchor['admissible_actions']
                or world.state_digest(0)!=anchor['environment_state_sha256']
                or digest(memory.state())!=anchor['visible_state_sha256']):
            raise ValueError('Environment replay differs from the recorded decision anchor')
        history=list(anchor['history']);used=0;score=0.;success=False
        for step in range(anchor['anchor_step'],max_steps):
            visible=memory.state()
            sid=target if step==anchor['anchor_step'] else router.route_many([visible])[0]['selected_skill_id']
            payload='' if control and sid==target else bank.get(sid).payload
            prompt=build_state_prompt(visible,payload)
            text,ntokens=generate(models[checkpoint],prompt,seed*1000+step-anchor['anchor_step'])
            action,_=project_action(text);previous=obs
            obs,_,done,info=world.step_one(0,action);used+=ntokens
            transition(memory,action,obs,info)
            history.append({'observation':previous,'action':action})
            score=info['task_score'];success=info['won']
            if done:break
        return {'success':float(success),'task_score':float(score),'generated_tokens':used}

    results=[]
    with (out/'paired_anchors.jsonl').open('w') as file:
        for i,anchor in enumerate(anchors):
            paired=[]
            for seed in continuation_seeds:
                conditions={f'{checkpoint}_{arm}':continuation(anchor,checkpoint,arm=='control',seed)
                    for checkpoint in ('old','new') for arm in ('skill','control')}
                mo=conditions['old_skill']['success']-conditions['old_control']['success']
                mn=conditions['new_skill']['success']-conditions['new_control']['success']
                paired.append({'seed':seed,'conditions':conditions,'m_old':mo,'m_new':mn,'delta_m':mn-mo})
            row={'anchor_id':i,'task_id':anchor['task_id'],'skill_id':anchor['skill_id'],'paired':paired,
                'm_old':float(np.mean([r['m_old'] for r in paired])),'m_new':float(np.mean([r['m_new'] for r in paired])),
                'delta_m':float(np.mean([r['delta_m'] for r in paired]))}
            file.write(json.dumps(row,ensure_ascii=False)+'\n');file.flush();results.append(row)
            print(json.dumps({'anchors_completed':i+1,'total':len(anchors)}),flush=True)
    groups=defaultdict(list)
    for row in results:groups[row['skill_id']].append(row)
    utility=[{'skill_id':sid,'n_eval_questions':len(rows),'n_eval_tasks':len({r['task_id'] for r in rows}),
        **{k:float(np.mean([r[k] for r in rows])) for k in ('m_old','m_new','delta_m')}} for sid,rows in sorted(groups.items())]
    (out/'skill_utility.json').write_text(json.dumps(utility,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({'schema_version':'skillscope.webshop_paired_utility.v1','reference_tasks':len(spec['eval_ids']),
        'anchors':len(anchors),'continuation_seeds':continuation_seeds,'bank_frozen':True,'same_prefix_replay_verified':True,
        'same_visible_memory_replay_verified':True,'router_frozen':True,'state_version':'webshop-visible-evidence-v1',
        'dense_prompt_width':None if contract==COMPACT else 16384,'same_prompt_padding_budget_as_training':True,
        'prompt_budget':16384,'prompt_storage_width':16384,'forward_contract':contract,
        'generation_padding':'exact-valid-length' if contract==COMPACT else 'dense-16384',
        'estimand':'success_skill-success_control; delta_M=M_new-M_old','control':'target payload only, selected ID and candidate pool retained',
        'gold_used_by_readout':False,'smoke':smoke},indent=2)+'\n')
    router.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed-dir',type=Path,required=True);p.add_argument('--new-model',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run(a.seed_dir,a.new_model,a.prepared,smoke=a.smoke)
