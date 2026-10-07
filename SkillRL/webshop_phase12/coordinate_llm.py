"""Fresh WebShop-only recovery queue. Inspect all eight cards, never stop others."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from webshop_phase12.assets import ROOT,RUN_ROOT
from webshop_phase12.run import pipeline,run_stage,write,now,verify_idle


def gpu_rows():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(parts[0]):{'memory_mib':int(parts[1]),'utilization':int(parts[2])}
        for line in raw.splitlines() if (parts:=line.split(','))}


def choose_layout(rows):
    idle=sorted(i for i,row in rows.items() if row['memory_mib']<=128 and row['utilization']<=5)
    if len(idle)<2:return None
    # Keep the global 128-task minibatch unchanged: train ranks must divide it.
    n=4 if len(idle)>=5 else 2
    training=idle[:n];routers=idle[n:] if len(idle)>n else [training[-1]]
    return {'train_gpus':training,'router_gpu':routers[0],'router_gpus':routers,'evaluation_gpu':training[0],
        'shared_router':bool(set(routers)&set(training)),'actor_world_size':n}


def verify_source():
    manifest=json.loads((RUN_ROOT/'source-manifest-llm-v1.json').read_text())
    for row in manifest['files']:
        if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Registered source changed: '+row['path'])


def main(queue):
    queue.mkdir(exist_ok=False)
    status=queue/'status.json'
    try:
        verify_source()
        while True:
            rows=gpu_rows();layout=choose_layout(rows)
            if layout is not None:break
            write(status,{'status':'waiting_for_free_GPUs','formal_started':False,'inspected_gpu_ids':list(rows),
                'gpu_occupancy':rows,'updated_at_utc':now(),'minimum_idle_cards':2})
            time.sleep(20)
        verify_source();verify_idle(sorted(set(layout['train_gpus'])|set(layout['router_gpus'])))
        write(queue/'layout.json',{**layout,'registered_at_utc':now(),'policy':'all-eight-idle-scan; fixed layout for this cohort'})
        write(status,{'status':'running_frozen_router_gpu_gate','layout':layout,'formal_started':False,'updated_at_utc':now()})
        run_stage('frozen-router-gpu-gate',[sys.executable,'-B','-u','-m','webshop_phase12.gpu_probe','--out',str(queue/'gpu-probe')],
            queue,layout['router_gpus'],layout['router_gpus'])
        verify_source()
        smoke=queue/'smoke';formal=queue/'formal-s404-s505'
        write(status,{'status':'running_end_to_end_preflight','root':str(smoke),'layout':layout,'formal_started':False,'updated_at_utc':now()})
        pipeline(smoke,RUN_ROOT/'prepared-smoke',smoke=True,gpus=tuple(layout['train_gpus']),
            evaluation_gpu=layout['evaluation_gpu'],router_gpu=layout['router_gpus'])
        seed=smoke/'seed404'
        required=[seed/'readout/prediction-locked.json',seed/'paired_eval/manifest.json',seed/'metrics/metrics.json']
        if not all(path.is_file() for path in required):raise ValueError('Missing preflight readout/paired utility/metrics')
        paired=json.loads(required[1].read_text())
        if not all(paired[key] for key in ('same_prefix_replay_verified','same_visible_memory_replay_verified','bank_frozen','router_frozen')):
            raise ValueError('Preflight replay/frozen-router contract failed')
        if not list((seed/'phase2/optimizer_steps').glob('*.jsonl')):raise ValueError('Preflight lacks optimizer evidence')
        verify_source()
        write(status,{'status':'formal_started','formal_started':True,'root':str(formal),'layout':layout,'updated_at_utc':now()})
        pipeline(formal,RUN_ROOT/'prepared',gpus=tuple(layout['train_gpus']),
            evaluation_gpu=layout['evaluation_gpu'],router_gpu=layout['router_gpus'])
        write(status,{'status':'complete','formal_started':True,'root':str(formal),'updated_at_utc':now()})
    except BaseException as error:
        write(status,{'status':'failed','error_type':type(error).__name__,'error':str(error),'automatic_retries':0,'updated_at_utc':now()})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--queue',type=Path,required=True);args=parser.parse_args();main(args.queue)
