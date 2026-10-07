"""Four-condition fixed-prefix readout; no Eval task outcomes are read."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch
from transformers import AutoTokenizer

from logicbench_phase12.fixed_state import _load_model
from webshop_phase12.dense_scoring import recorded_inputs,control_inputs,score_dense
from phase2.stable_direction import token_signals
from webshop_phase12.assets import BASE_MODEL, WebshopBank
from webshop_phase12.prompts import remove_guidance
from webshop_phase12.accelerated import COMPACT,load_forward_contract
from webshop_phase12.training_storage import load_training_archive


def run(seed_dir, new_model, *, smoke=False,output=None,devices=('cuda','cuda')):
    output=output or seed_dir/'readout';output.mkdir(exist_ok=False)
    bank=WebshopBank()
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,local_files_only=True)
    archive=load_training_archive(seed_dir/'phase2/batches/u0001')
    if archive['schema_version']!='phase2.exact_training_batch.v1':
        raise ValueError('Wrong actual-training-batch schema')
    tensors,metadata=archive['tensors'],archive['metadata']
    contract=load_forward_contract(seed_dir);trim_padding=contract==COMPACT
    old=_load_model(BASE_MODEL,devices[0]);new=_load_model(new_model,devices[1])
    primary_device=next(old.parameters()).device
    executor=ThreadPoolExecutor(max_workers=2)
    rows=[];original_prompt_matches=0;chosen_max_error=0.
    for i,meta in enumerate(metadata):
        mask=tensors['phase2_actual_loss_mask'][i].bool()
        if not mask.any():
            continue
        info=meta['info'];sid=info['selected_skill_id'];skill=bank.get(sid)
        if info['phase2_payload_text']!=skill.payload or info['skill_version_sha256']!=skill.payload_sha256:
            raise ValueError('Training guidance differs from frozen source skill')
        original_text=info['prompt_text']
        control_text=remove_guidance(original_text,skill.payload)
        def token_ids(text):
            chat=tokenizer.apply_chat_template([{'role':'user','content':text}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
            return tokenizer(chat,add_special_tokens=False)['input_ids']
        plen=tensors['prompts'].shape[-1]
        original=tensors['prompts'][i,tensors['attention_mask'][i,:plen].bool()].tolist()
        if token_ids(original_text)!=original:
            raise ValueError(f'Recorded prompt and recreated template differ in row {i}')
        original_prompt_matches+=1
        control=token_ids(control_text)
        response=tensors['responses'][i,mask].tolist()
        dense=recorded_inputs(tensors,i)
        padding=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        dense_control=control_inputs(tensors,i,control,pad_token_id=padding)
        response_length=tensors['responses'].shape[-1]
        old_future=executor.submit(score_dense,old,dense,response_length,mask,trim_padding=trim_padding)
        new_future=executor.submit(score_dense,new,dense,response_length,mask,trim_padding=trim_padding)
        oo=old_future.result();no=new_future.result().to(primary_device)
        actions=torch.tensor(response,dtype=torch.long,device=primary_device)
        if 'old_log_probs' not in tensors:raise ValueError('Native old-logprob parity witness missing')
        error=(oo.gather(1,actions[:,None]).flatten().cpu()-tensors['old_log_probs'][i,mask].float()).abs().max().item()
        chosen_max_error=max(chosen_max_error,error)
        if error>1e-3:raise ValueError(f'Native old-forward parity failed in row {i}: {error}')
        old_future=executor.submit(score_dense,old,dense_control,response_length,mask,trim_padding=trim_padding)
        new_future=executor.submit(score_dense,new,dense_control,response_length,mask,trim_padding=trim_padding)
        oc=old_future.result();nc=new_future.result().to(primary_device)
        advantage=tensors['advantages'][i,mask].to(primary_device)
        signals=token_signals(oo,no,oc,nc,actions,advantage)
        u=no.double()-oo.double();delta=u-(nc.double()-oc.double())
        uc=u-u.mean(-1,keepdim=True);dc=delta-delta.mean(-1,keepdim=True)
        chosen_u=u.gather(1,actions[:,None]).flatten()
        real=-advantage.double()*chosen_u*(uc*dc).sum(-1)/(uc.square().sum(-1)+1e-12)
        for j in range(len(response)):
            p=signals['P_int'][j].item();valid=bool(signals['direction_valid'][j])
            rows.append({'skill_id':sid,'task_id':int(info['task_id']),'trajectory_id':meta['trajectory_id'],'decision_id':meta['decision_id'],
                'response_token_offset':j,'action_token_id':response[j],
                **{k:v[j].item() for k,v in signals.items() if v.ndim==1},
                'D_real':real[j].item(),'D_sign_balance':-float((p>0)-(p<0)) if valid else 0.})
        del oo,no,oc,nc,signals,u,delta,uc,dc,real
        if original_prompt_matches%32==0:
            print(json.dumps({'decisions_scored':original_prompt_matches,'tokens':len(rows)}),flush=True)
    if not rows:
        raise ValueError('No real loss tokens for readout')
    executor.shutdown()
    frame=pd.DataFrame(rows);frame.to_parquet(output/'token_signals.parquet',index=False)
    records=[]
    for sid,group in frame.groupby('skill_id',sort=True):
        directed=group.advantage*group.chosen_delta
        orientation=directed.sum()/(directed.abs().sum()+1e-12)
        records.append({'skill_id':sid,'n_questions':group.task_id.nunique(),'n_trajectories':group.trajectory_id.nunique(),
            'n_loss_tokens':len(group),'D_sign_balance':group.D_sign_balance.mean(),
            'D_signed_gate':-(group.P_int*group.gate).mean(),'D_original':group.D_contribution.mean(),
            'D_signed':-group.P_int.mean(),'D_real':group.D_real.mean(),
            'D_orientation':-orientation,'D_factor':-group.delta_centered_norm.mean()*orientation,
            'M_delta_raw':group.delta_norm.mean(),'M_delta_centered':group.delta_centered_norm.mean()})
    pd.DataFrame(records).to_csv(output/'skill_scores.csv',index=False)
    (output/'prediction-locked.json').write_text(json.dumps({'status':'locked_before_independent_eval',
        'training_decisions':original_prompt_matches,'training_loss_tokens':len(frame),'skills':len(records),
        'all_recorded_prompts_verified':True,'full_vocabulary':True,'actual_GRPO_advantages':True,
        'four_condition_backend':'same HF bf16 SDPA; versioned padding, recorded masks/positions and all response slots',
        'forward_contract':contract,'trim_common_padding':trim_padding,
        'native_old_forward_parity_passed':True,'native_old_forward_parity_tolerance':1e-3,
        'native_old_chosen_max_abs_difference':chosen_max_error,
        'gold_outcomes_used':False,'primary_score':'D_sign_balance','primary_aggregation':'token mean',
        'unsupported_skill_policy':'no score for uninvoked skills'},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed-dir',type=Path,required=True);p.add_argument('--new-model',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--devices',default='cuda,cuda')
    a=p.parse_args();devices=tuple(a.devices.split(','))
    if len(devices)!=2:p.error('Exactly two scoring devices are required')
    run(a.seed_dir,a.new_model,output=a.output,devices=devices)
