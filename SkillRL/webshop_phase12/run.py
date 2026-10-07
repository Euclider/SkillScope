"""Dedicated fail-fast supervisor; operates only on this new experiment root."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

from webshop_phase12.assets import ROOT,RUN_ROOT


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path,value):
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def run_stage(name,command,root,gpu_ids,router_gpu=None):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpu_ids)),PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        JAVA_TOOL_OPTIONS=os.environ.get('JAVA_TOOL_OPTIONS','-Xmx16g'))
    java_home=os.environ.get('WEBSHOP_JAVA_HOME',os.environ.get('JAVA_HOME'))
    if java_home:
        env.update(JAVA_HOME=java_home,JVM_PATH=os.environ.get('JVM_PATH',str(Path(java_home)/'lib/server/libjvm.so')))
    if os.environ.get('WEBSHOP_RUNTIME_LIBRARY_PATH'):
        env['LD_LIBRARY_PATH']=os.environ['WEBSHOP_RUNTIME_LIBRARY_PATH']+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    if os.environ.get('WEBSHOP_LD_PRELOAD'):env['LD_PRELOAD']=os.environ['WEBSHOP_LD_PRELOAD']
    if router_gpu is not None:
        routers=list(router_gpu) if isinstance(router_gpu,(list,tuple)) else [router_gpu]
        env.update(WEBSHOP_ROUTER_DEVICE='cuda:0',WEBSHOP_ROUTER_GPU=str(routers[0]),
            WEBSHOP_ROUTER_GPUS=','.join(map(str,routers)),WEBSHOP_ROUTER_CACHE=str(root/'router.sqlite3'),
            WEBSHOP_ROUTER_SHARED='1' if set(routers)&set(gpu_ids) else '0')
    with (root/f'{name}.log').open('w') as log:
        log.write(json.dumps({'command':command,'gpu_ids':gpu_ids,'started':now()})+'\n');log.flush()
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+96*3600
        while child.poll() is None:
            write(root/'status.json',{'stage':name,'status':'running','pid':child.pid,'gpu_ids':gpu_ids,'updated_at_utc':now(),
                'free_disk_bytes':shutil.disk_usage(RUN_ROOT).free})
            if time.monotonic()>deadline:
                # This process group was created above exclusively for this stage.
                import signal
                os.killpg(child.pid,signal.SIGTERM)
                raise TimeoutError(f'{name} exceeded the registered 96-hour stage timeout')
            time.sleep(20)
    event={'stage':name,'returncode':child.returncode,'finished_at_utc':now(),'gpu_ids':gpu_ids}
    with (root/'stages.jsonl').open('a') as file:file.write(json.dumps(event)+'\n')
    if child.returncode:
        write(root/'status.json',{**event,'status':'failed','automatic_retries':0})
        raise RuntimeError(f'{name} failed; inspect {root/f"{name}.log"}')
    write(root/'status.json',{**event,'status':'stage_complete'})


def verify_idle(ids):
    result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows={int(parts[0]):(int(parts[1]),int(parts[2])) for line in result.splitlines() if (parts:=line.split(','))}
    if any(rows[i][0]>128 or rows[i][1]>5 for i in ids):
        raise RuntimeError(f'Assigned GPUs are no longer idle: {rows}; do not stop other jobs')


def wait_idle(ids,root,stage):
    while True:
        try:verify_idle(ids);return
        except RuntimeError:
            write(root/'status.json',{'stage':stage,'status':'waiting_for_free_GPUs','gpu_ids':list(ids),'updated_at_utc':now()})
            time.sleep(20)


def pipeline(root,prepared,*,smoke=False,gpus=(3,4),evaluation_gpu=7,router_gpu=7,config='webshop54_phase12_v1',train_only=False):
    if config not in ('webshop54_phase12_v1','webshop54_phase12_accel_v2','webshop54_phase12_accel_v3','webshop54_phase12_small_v4'):
        raise ValueError('Unknown registered WebShop training configuration')
    accelerated=config!='webshop54_phase12_v1'
    routers=list(router_gpu) if isinstance(router_gpu,(list,tuple)) else [router_gpu]
    root.mkdir(exist_ok=False)
    spec=json.loads((prepared/'manifest.json').read_text())
    write(root/'protocol.json',{'prepared_manifest':str(prepared/'manifest.json'),'prepared_sha256':hashlib.sha256((prepared/'manifest.json').read_bytes()).hexdigest(),
        'train_gpu_ids':gpus,'evaluation_gpu_id':evaluation_gpu,'router_gpu_id':router_gpu,
        'router':'frozen-Qwen3.5-4B-local-vllm-all54','state_version':'webshop-visible-evidence-v1',
        'router_model_receipt_sha256':hashlib.sha256((RUN_ROOT/'frozen-router-model.json').read_bytes()).hexdigest(),
        'smoke':smoke,'config':config,
        'forward_contract':('hf-bucket256-active-v4' if config.endswith('small_v4') else
                           ('hf-bucket256-v3' if config.endswith('accel_v3') else 'hf-exact-length-v2')) if accelerated else 'hf-dense-v1',
        'native_full_vocabulary_files_retained':False,'four_condition_HF_recompute':True,'automatic_retries':0,
        'registered_stage_timeout_hours':96})
    python=sys.executable
    for seed in spec['seeds']:
        seed_dir=root/f'seed{seed}';seed_dir.mkdir()
        # Complete native model/optimizer/RNG state is kept in dedicated tmpfs;
        # merged endpoints, evidence and results remain on persistent disk.
        minimum_disk=12 if config.endswith('small_v4') else 25
        if shutil.disk_usage(RUN_ROOT).free < minimum_disk*1024**3 or shutil.disk_usage('/dev/shm').free < 55*1024**3:
            raise RuntimeError('Insufficient persistent disk or RAM-backed checkpoint storage; no data deleted')
        checkpoint_namespace=str(root.relative_to(RUN_ROOT)).replace('/','-')
        if config.endswith('small_v4'):
            checkpoint_namespace=hashlib.sha256(str(root).encode()).hexdigest()[:10]+'-'+checkpoint_namespace
        temporary=Path('/dev/shm')/f'wangyifan-webshop-phase12-20261006-{checkpoint_namespace}'/f'seed{seed}'/'checkpoints'
        temporary.mkdir(parents=True,exist_ok=False)
        (seed_dir/'checkpoints').symlink_to(temporary,target_is_directory=True)
        write(seed_dir/'checkpoint-storage.json',{'native_checkpoint_root':str(temporary),'persistent_model_root':str(seed_dir/'merged-endpoint'),
            'model_optimizer_rng_preserved':not (smoke and accelerated),
            'smoke_model_only_checkpoint':bool(smoke and accelerated),
            'native_checkpoint_medium':'RAM-backed tmpfs; lost on reboot','automatic_deletion':False})
        wait_idle(sorted(set(gpus)|set(routers)),root,f'seed{seed}-training')
        command=[python,'-B','-m','verl.trainer.main_ppo','--config-name',config,
            f'webshop_run.seed={seed}',f'webshop_run.prepared={prepared}',f'webshop_run.run_root={root}',
            f'webshop_run.tasks_per_update={spec["tasks_per_update"]}',f'webshop_run.updates={spec["updates"]}',
            f'alignment_run.ray_temp_dir=/tmp/ws12l-{hashlib.sha256(str(root).encode()).hexdigest()[:8]}-{seed}',
            f'trainer.n_gpus_per_node={len(gpus)}']
        if smoke:
            command+=['env.max_steps=8']
            if accelerated:
                # Optimizer steps are still witnessed; only this disposable
                # preflight endpoint omits restart state to reserve tmpfs.
                command+=['actor_rollout_ref.actor.checkpoint.contents=[model]',
                          '+actor_rollout_ref.actor.checkpoint.export_model_only=true']
        else:
            if config=='webshop54_phase12_v1':command+=['+ray_init.object_store_memory=8589934592']
        run_stage(f'seed{seed}-training',command,root,gpus,router_gpu)
        performance_path=root/f'router-performance-train-s{seed}.json'
        performance=json.loads(performance_path.read_text())
        if smoke and (performance['step_calls']<2 or performance['router_fraction_of_observed_rollout']>.20):
            raise RuntimeError('Router exceeded the registered 20% rollout-time gate; formal training not admitted')
        checkpoint=seed_dir/f'checkpoints/global_step_{spec["updates"]}/actor'
        if not (seed_dir/'phase2/batches/u0001/manifest.json').is_file() or not checkpoint.is_dir():
            raise RuntimeError('U1 training evidence or endpoint checkpoint missing')
        merged=seed_dir/'merged-endpoint'
        run_stage(f'seed{seed}-merge',[python,'-B','scripts/model_merger.py','merge','--backend','fsdp','--local_dir',str(checkpoint),'--target_dir',str(merged)],root,[])
        if train_only:continue
        wait_idle([evaluation_gpu],root,f'seed{seed}-readout')
        run_stage(f'seed{seed}-readout',[python,'-B','-m','webshop_phase12.readout','--seed-dir',str(seed_dir),'--new-model',str(merged)],root,[evaluation_gpu])
        command=[python,'-B','-m','webshop_phase12.evaluate','--seed-dir',str(seed_dir),'--new-model',str(merged),'--prepared',str(prepared)]
        if smoke:command+=['--smoke']
        # Paired continuations currently submit one state at a time. One
        # identical frozen replica suffices; extra routing cards are released.
        wait_idle(sorted({evaluation_gpu,routers[0]}),root,f'seed{seed}-paired-eval')
        run_stage(f'seed{seed}-paired-eval',command,root,[evaluation_gpu],routers[0])
        run_stage(f'seed{seed}-metrics',[python,'-B','-m','webshop_phase12.metrics','--seed-dir',str(seed_dir)],root,[])
    completed='training-complete.json' if train_only else 'complete.json'
    write(root/completed,{'status':'training_complete' if train_only else 'complete','smoke':smoke,'finished_at_utc':now(),'seeds':list(spec['seeds'])})
    write(root/'status.json',{'status':'training_complete' if train_only else 'complete','smoke':smoke,'updated_at_utc':now()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--gpus',default='3,4');p.add_argument('--evaluation-gpu',type=int,default=7)
    p.add_argument('--router-gpu',type=int,default=7)
    p.add_argument('--config',default='webshop54_phase12_v1',choices=('webshop54_phase12_v1','webshop54_phase12_accel_v2','webshop54_phase12_accel_v3'))
    a=p.parse_args();pipeline(a.root,a.prepared,smoke=a.smoke,gpus=tuple(map(int,a.gpus.split(','))),evaluation_gpu=a.evaluation_gpu,router_gpu=a.router_gpu,config=a.config)
