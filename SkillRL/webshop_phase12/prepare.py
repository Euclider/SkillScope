import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from webshop_phase12.assets import ROOT,RUN_ROOT,WebshopBank,make_schedule
from webshop_phase12.envs import ShopWorld


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    world=ShopWorld(1)
    plan=make_schedule(len(world.server.goals))
    bank=WebshopBank()
    destination=RUN_ROOT/('prepared-smoke' if args.smoke else 'prepared')
    destination.mkdir(exist_ok=False)
    (destination/'bank-manifest.json').write_text(json.dumps(bank.manifest,ensure_ascii=False,indent=2)+'\n')
    if args.smoke:
        plan['updates']=1;plan['tasks_per_update']=8
        plan['seeds']={'404':plan['seeds']['404'][:8]}
        plan['eval_ids']=[1500,1501]
        plan['smoke_training_tasks_used_for_pipeline_check_only']=True
    for seed,ids in plan['seeds'].items():
        directory=destination/f'seed{seed}';directory.mkdir()
        rows=[{'data_source':'webshop','prompt':[{'role':'user','content':'WebShop task'}],
            'env_kwargs':{'task_id':i},'extra_info':{'index':i,'task_id':i,'split':'train'},
            'reward_model':{'style':'rule','ground_truth':''},'ability':'shopping'} for i in ids]
        pd.DataFrame(rows).to_parquet(directory/'train.parquet',index=False)
        pd.DataFrame(rows[:2]).to_parquet(directory/'dev.parquet',index=False)
    plan.update(schema_version='skillscope.webshop_phase12_schedule.v1',bank_manifest_sha256=bank.manifest_sha256,
        skill_bank_frozen=True,environment_catalog_seed=0)
    (destination/'manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
    (destination/'task-inventory.json').write_text(json.dumps([{'task_id':i,'instruction_text':g['instruction_text']} for i,g in enumerate(world.server.goals)],ensure_ascii=False)+'\n')
    print(json.dumps({'prepared':str(destination),'number_of_goals':len(world.server.goals),'train_task_count':len(plan['train_ids']),
        'tasks_per_update':plan['tasks_per_update'],'seeds':list(plan['seeds']),'eval_tasks':len(plan['eval_ids'])}),flush=True)


if __name__=='__main__':
    main()
