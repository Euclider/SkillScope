"""Approved 50-update LogicBench D_sign_balance loop. Preflight is the default.

Run through scripts/run_logicbench_phase3.sh to select the training Conda ABI.
All subprocesses inherit explicit GPU assignment; failed/ambiguous editor calls
stop the run. No automatic replay of a partially trained window is permitted.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .bank import Bank
from .common import digest, require, strict_json, write_new
from .logicbench_loop_data import ROOT, DATA, make_schedule, prepare_window, training_configuration


def status(root, stage, **extra):
    from phase1.archive import atomic_write_json, utc_now
    atomic_write_json(Path(root)/'status.json', {'stage':stage,'updated_at':utc_now(),**extra})
    print(stage, extra, flush=True)


def load_editor_credential(path=None):
    """Inject a private, out-of-repository credential only when env is unset."""
    import stat
    if os.environ.get('SKILLRL_PHASE3_EDITOR_API_KEY', '').strip():
        return False
    path = Path(path or '/home/wangyifan/.local/state/skillrl/credentials/phase3-editor.key')
    if not path.exists():
        return False
    require(not path.is_symlink(), 'Credential must be a private regular file')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and info.st_mode & 0o077 == 0 and info.st_size < 1024, 'Unsafe credential file permissions')
        value = os.read(descriptor, 1024).decode().strip()
        require(value.startswith('sk-') and not any(c.isspace() for c in value), 'Invalid credential format')
        os.environ['SKILLRL_PHASE3_EDITOR_API_KEY'] = value
    finally:
        os.close(descriptor)
    return True


def readiness(setting, root, gpu_ids):
    import torch
    blockers = []
    if not torch.cuda.is_available() or torch.cuda.device_count() < setting['gpu_count']:
        blockers.append('gpu_access')
    if len(gpu_ids) != setting['gpu_count'] or len(set(gpu_ids)) != len(gpu_ids):
        blockers.append('gpu_assignment')
    gpu_state = []
    if 'gpu_access' not in blockers:
        probe = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        if probe.returncode:
            blockers.append('gpu_usage_unavailable')
        else:
            gpu_state = [dict(zip(('index','memory_used_mib','utilization_percent'),map(int,line.split(','))))
                         for line in probe.stdout.splitlines() if line.strip()]
            chosen = [r for r in gpu_state if r['index'] in gpu_ids]
            if len(chosen) != len(gpu_ids) or any(r['memory_used_mib'] > 2048 or r['utilization_percent'] > 10 for r in chosen):
                blockers.append('selected_gpus_not_free')
    if not os.environ.get('SKILLRL_PHASE3_EDITOR_API_KEY','').strip():
        blockers.append('editor_key_missing')
    try:
        sdk = importlib.metadata.version('openai')
    except importlib.metadata.PackageNotFoundError:
        sdk = None
    required_sdk = setting['editor'].get('sdk_version', '3.15.0')
    if sdk != required_sdk:
        blockers.append('editor_sdk_requires_' + required_sdk)
    router_python = setting['router']['python_executable']
    router = None
    if not Path(router_python).is_file():
        blockers.append('router_interpreter_missing')
    else:
        from .gpu_encoder_service import encoder_environment
        probe = subprocess.run([router_python,'-c',
            'import json,torch,importlib.metadata as m; print(json.dumps({"cuda":torch.cuda.is_available(),'
            '"torch":m.version("torch"),"sentence_transformers":m.version("sentence-transformers"),'
            '"transformers":m.version("transformers"),"tokenizers":m.version("tokenizers")}))'],
            capture_output=True,text=True,timeout=60,
            env=encoder_environment(setting['router']['device'], gpu_ids[0]))
        if probe.returncode:
            blockers.append('router_import_failed')
        else:
            router = strict_json(probe.stdout.strip())
            from phase3.embedding_routing import validate_settings
            profile, _ = validate_settings(setting['router'])
            # The encoder checks the complete frozen profile again on startup.
            expected = {'torch':profile.torch_version,'sentence_transformers':profile.sentence_transformers_version,
                        'transformers':profile.transformers_version,'tokenizers':profile.tokenizers_version}
            if (setting['router']['device'] != 'cpu' and not router['cuda']) or any(router[k] != value for k,value in expected.items()):
                blockers.append('router_frozen_runtime_mismatch')
    from skillnet_cohort.runtime import disk_gate
    storage = setting['storage']
    from logicbench_phase3_recovery.runtime import checkpoint_reserve
    try:
        disk = disk_gate(root,checkpoint_reserve(root,setting),
            minimum_free_bytes=storage['minimum_free_bytes'],maximum_run_bytes=storage['maximum_run_bytes'])
    except OSError:
        disk = {'free_bytes':shutil.disk_usage(Path(root).parent).free}
        blockers.append('disk_budget')
    return {'status':'blocked' if blockers else 'ready','blockers':blockers,'training_started':False,
        'visible_cuda_devices':torch.cuda.device_count(),'gpu_ids':gpu_ids,'openai_sdk':sdk,
        'router_runtime':router,'gpu_state':gpu_state,'disk':disk,
        'editor_key_present':bool(os.environ.get('SKILLRL_PHASE3_EDITOR_API_KEY'))}


def prepare(setting, root):
    from .logicbench import initial_bank
    require(setting['execution_approved'] is True and setting['schema_version'] == 'skillrl.phase3.logicbench.loop.v1',
            'Only the approved LogicBench loop can execute')
    require(setting['method'] == 'D_sign_balance' and setting['branch_id'] == 'readout_d'
            and setting['candidate_k'] == 5 and setting['mutation_units'] == 3
            and setting['eval_seeds'] == list(range(16)) and not setting['editor_input_cap_enforced'], 'Changed approved protocol')
    write_new(root/'setting.json',setting)
    schedule = make_schedule(setting)
    write_new(root/'schedule.json',schedule)
    bank = initial_bank(setting['branch_id'])
    bank.save(root/'banks')
    write_new(root/'initial.json',{'bank_sha256':bank.manifest_sha256,'setting_sha256':digest(setting),
        'schedule_sha256':digest(schedule),'updates':50,'optimizer_horizon':150})
    return bank,schedule


def command(root, tag, args):
    log = root/'logs'/f'{tag}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    status(root,tag,training_started=True,log=str(log))
    with log.open('ab') as stream:
        try:
            subprocess.run([sys.executable,'-B','-m',*map(str,args)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        except subprocess.CalledProcessError as error:
            status(root,'failed',failed_stage=tag,exit_code=error.returncode,log=str(log))
            raise


def cleanup(root, end):
    """Only run-owned obsolete model/checkpoint directories after a sealed event."""
    root = Path(root).resolve()
    seal = root/'windows'/f'u{end-5:04d}-u{end:04d}'/'complete.json'
    require(seal.is_file(), 'Cannot clean an unsealed window')
    record = strict_json(seal.read_text())
    require(record['end'] == end and record['start'] == end-5, 'Wrong cleanup boundary')
    from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
    current = root/'checkpoints'/f'global_step_{end}'
    require(validate_full_checkpoint(current)['world_size'] == 4, 'Incomplete current native checkpoint')
    require((root/'models'/f'u{end:04d}'/'phase2_export.json').is_file(), 'Missing current HF export')
    targets = [root/'checkpoints'/f'global_step_{end-1}']
    if end > 5:
        targets += [root/'checkpoints'/f'global_step_{end-5}',root/'models'/f'u{end-5:04d}']
    for path in targets:
        require(path.parent in (root/'checkpoints',root/'models') and path.resolve().is_relative_to(root)
                and not any(p.is_symlink() for p in (path,*path.parents)), 'Unsafe cleanup target')
        if path.exists():
            shutil.rmtree(path)
    write_new(root/'retention'/f'u{end:04d}.json', {'sealed_window':str(seal),'removed':list(map(str,targets))})


def train_stage(setting,root,bank,start,gpu_ids,resume_update):
    from omegaconf import OmegaConf
    path = bank.save(root/'banks')
    cfg = training_configuration(setting,root,bank,path,start=start,gpu_ids=gpu_ids,resume_update=resume_update)
    resume = start if resume_update is None else resume_update
    from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
    if resume:
        require(validate_full_checkpoint(root/'checkpoints'/f'global_step_{resume}')['world_size'] == len(gpu_ids),
                'Native world size changed')
    write_new(root/'segments'/f'u{start:04d}-u{start+5:04d}-resume{resume:04d}.json',OmegaConf.to_container(cfg,resolve=True))
    from verl.trainer.main_ppo import run_ppo
    import ray
    try:
        run_ppo(cfg)
    finally:
        ray.shutdown()


def question_objects(rows):
    from phase1.logicbench_single_step import LogicBenchQuestion
    return [LogicBenchQuestion(q['question_id'],q['question'],q['task_type'],q['answer'],'') for q in rows]


def accuracy(rows):
    groups = {'overall':rows,**{kind:[r for r in rows if r['task_type']==kind] for kind in ('BQA','MCQA')}}
    return {key:{'responses':len(group),'questions':len({r['question_id'] for r in group}),
        'accuracy':sum(r['success'] for r in group)/len(group),
        'format_valid_rate':sum(r['format_valid'] for r in group)/len(group),
        'length_cap_rate':sum(r['hit_length_cap'] for r in group)/len(group)} for key,group in groups.items() if group}


def validate_evaluation_rows(rows, question_ids, seeds, bank_sha256):
    expected = {(q,s) for q in question_ids for s in seeds}
    require(len(rows)==len(expected) and {(r['game_id'],r['eval_seed']) for r in rows}==expected
            and all(r['bank_sha256']==bank_sha256 for r in rows), 'Incomplete/mismatched evaluation coverage')


def bind_evaluation_identity(output, checkpoint, setting, generator):
    import transformers
    from skillnet_cohort.common import file_hash
    names = ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja',
             'tokenizer.model','vocab.json','merges.txt','added_tokens.json')
    original = Path(setting['model_path'])
    generation = Path(checkpoint)/'generation_config.json'
    identity = {'inherited_generation_config':generator.model.generation_config.to_dict(),
        'generation_file_sha256':file_hash(generation) if generation.exists() else None,
        'tokenizer_files':{name:file_hash(original/name) if (original/name).exists() else None for name in names},
        'explicit_overrides':{k:setting[k] for k in ('temperature','top_p','max_new_tokens','max_prompt_tokens','rng_mode','eval_seeds')},
        'pad_token_id':generator.tokenizer.eos_token_id,'do_sample':True,'enable_thinking':False,
        'torch_version':str(generator.torch.__version__),'transformers_version':transformers.__version__}
    write_new(Path(output)/'decoding_identity.json',identity)
    return digest(identity)


def evaluate_stage(setting,root,bank,start,gpu_ids,*,final=False):
    from .api import JSONClient
    from .logicbench import evaluate_bank, revise_once
    from .logicbench_routing import LogicBenchRouterPool
    from .logicbench_run import HFGenerator, prompt_validator, evaluation_prompt_contexts
    from .logicbench_loop_readout import v13_payload
    from .run import model_identity
    from phase1.logicbench_single_step import load_logicbench_eval
    schedule = strict_json((root/'schedule.json').read_text())
    end = start+5
    checkpoint = root/'models'/f'u{end:04d}'
    policy_hash = model_identity(checkpoint)
    router_setting = strict_json(__import__('json').dumps(setting['router']))
    if router_setting['device'] != 'cpu':
        router_setting['execution']['shared_gpu_physical_id'] = gpu_ids[0]
    pool = LogicBenchRouterPool(router_setting,root/'router-local.sqlite3',bank.branch_id)
    generator = HFGenerator(checkpoint,setting)
    audit_path = root/'final' if final else root/'windows'/f'u{start:04d}-u{end:04d}'
    decoding_hash = bind_evaluation_identity(audit_path,checkpoint,setting,generator)
    def evaluate(current,questions):
        rows = evaluate_bank(current,questions,pool,generator,seeds=setting['eval_seeds'],rng_mode=setting['rng_mode'])
        validate_evaluation_rows(rows,{q.instance_id for q in questions},setting['eval_seeds'],current.manifest_sha256)
        return rows
    try:
        if final:
            target = root/'final'
            identity = {'bank_sha256':bank.manifest_sha256,'policy_sha256':policy_hash,
                'setting_sha256':digest(setting),'decoding_sha256':decoding_hash,'update':end}
            write_new(target/'source.json',identity)
            if not (target/'complete.json').exists():
                rows_path = target/'eval.json'
                questions = load_logicbench_eval()
                rows = strict_json(rows_path.read_text()) if rows_path.exists() else evaluate(bank,questions)
                validate_evaluation_rows(rows,{q.instance_id for q in questions},setting['eval_seeds'],bank.manifest_sha256)
                write_new(rows_path,rows)
                write_new(target/'complete.json',{'source_sha256':digest(identity),'rows_sha256':digest(rows),'metrics':accuracy(rows)})
            return
        window = root/'windows'/f'u{start:04d}-u{end:04d}'
        episodes = strict_json((root/'episodes'/f'u{start+1:04d}'/'logicbench.json').read_text())
        import torch
        from skillnet_cohort.common import file_hash
        batch_path = root/'direction_batches'/f'u{start+1:04d}.pt'
        require(file_hash(batch_path)==strict_json(batch_path.with_suffix('.json').read_text())['sha256'],'Changed capture')
        batch = torch.load(batch_path,map_location='cpu',weights_only=False)
        require(episodes['bank_sha256']==bank.manifest_sha256 and episodes['global_update']==start+1
                and episodes['evidence']==batch['evidence'], 'Editor/readout evidence differs')
        scores = strict_json((window/'readout'/'readout.json').read_text())
        payload,priority = v13_payload(bank,batch['evidence'],scores,k=setting['candidate_k'])
        write_new(window/'selection.json',{'bank_sha256':bank.manifest_sha256,'priority_ids':priority,
            'readout_sha256':digest(scores),'evidence_sha256':digest(batch['evidence']),
            'failed_evidence_ids':[r['evidence_id'] for r in payload['evidence']] if payload else []})
        selected = bank
        event = {'abstain':True,'reason':'no_supported_failed_skill'}
        if payload:
            gate = question_objects(schedule['gate'])
            # Warm actual GPU routing before spending the editor call.
            pool.for_bank(bank).route_question(gate[0].question)
            from logicbench_phase3_recovery.runtime import editor_config
            api = JSONClient(editor_config(setting,root),root/'editor.sqlite3',allow_live=True)
            all_inputs = question_objects(strict_json((DATA/'train.json').read_text())+strict_json((DATA/'dev.json').read_text()))
            validator = prompt_validator(generator.tokenizer,all_inputs+evaluation_prompt_contexts(),
                {**setting,'max_edited_skill_tokens':setting['max_prompt_tokens']})
            event = revise_once(bank,api=api,payload=payload,evaluate=lambda b:evaluate(b,gate),output=window/'editor',
                event_id=f'u{end:04d}',tolerance_pp=setting['gate_tolerance_pp'],gate_ids=[q.instance_id for q in gate],
                identity={'policy_sha256':policy_hash,'setting_sha256':digest(setting),'sampling_policy_update':start,
                          'decoding_sha256':decoding_hash},
                payload_validator=validator,enforce_input_cap=False)
            selected = event.pop('bank')
        selected.save(root/'banks')
        monitor_path = window/'monitor.json'
        monitor = strict_json(monitor_path.read_text()) if monitor_path.exists() else evaluate(selected,question_objects(schedule['monitor']))
        validate_evaluation_rows(monitor,{q['question_id'] for q in schedule['monitor']},
                                 setting['eval_seeds'],selected.manifest_sha256)
        write_new(monitor_path,monitor)
        write_new(window/'complete.json',{'start':start,'end':end,'previous_bank_sha256':bank.manifest_sha256,
            'selected_bank_sha256':selected.manifest_sha256,'policy_sha256':policy_hash,'event':event,
            'monitor_sha256':digest(monitor),'monitor':accuracy(monitor)})
    finally:
        pool.close()


def execute(setting,root,gpu_ids,resume_update=None):
    from .logicbench_routing import LogicBenchRouterPool
    from .run import model_identity
    from skillnet_cohort.runtime import disk_gate
    from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
    bank,schedule = prepare(setting,root)
    check = readiness(setting,root,gpu_ids)
    status(root,check['status'],**{k:v for k,v in check.items() if k!='status'})
    require(not check['blockers'], 'Preflight blocked: '+', '.join(check['blockers']))
    write_new(root/'launch.json',{'setting_sha256':digest(setting),'gpu_ids':gpu_ids,
        'initial_model_sha256':model_identity(setting['model_path'])})
    from skillnet_cohort.common import file_hash
    sources = sorted((ROOT/'phase3').glob('*.py')) + [ROOT/'verl/trainer/ppo/ray_trainer.py',
        ROOT/'verl/trainer/main_ppo.py',ROOT/'logicbench_phase12/rollout.py',ROOT/'phase2/stable_direction.py']
    from logicbench_phase3_recovery.runtime import verify_implementation, checkpoint_reserve
    verify_implementation(root,{str(p.relative_to(ROOT)):file_hash(p) for p in sources})
    for window in schedule['windows']:
        start,end = window['start'],window['end']
        target = root/'windows'/f'u{start:04d}-u{end:04d}'
        seal = target/'complete.json'
        if seal.exists():
            sealed = strict_json(seal.read_text())
            require(sealed['previous_bank_sha256']==bank.manifest_sha256,'Changed bank chain')
            bank = Bank.load(root/'banks'/f"{sealed['selected_bank_sha256']}.json",sealed['selected_bank_sha256'])
            if ((root/'checkpoints'/f'global_step_{end}').exists()
                    and (root/'models'/f'u{end:04d}'/'phase2_export.json').exists()):
                cleanup(root,end)
            continue
        storage = setting['storage']
        disk_gate(root,checkpoint_reserve(root,setting),minimum_free_bytes=storage['minimum_free_bytes'],
                  maximum_run_bytes=storage['maximum_run_bytes'])
        router_setting = strict_json(__import__('json').dumps(setting['router']))
        if router_setting['device'] != 'cpu':
            router_setting['execution']['shared_gpu_physical_id'] = gpu_ids[0]
        provider = LogicBenchRouterPool(router_setting,root/'router-local.sqlite3',bank.branch_id)
        try:
            prepare_window(setting,root,bank,window,schedule['monitor'],provider)
        finally:
            provider.close()
        bank_path = bank.save(root/'banks')
        base = ['phase3.logicbench_loop','--setting',root/'setting.json','--root',root,
                '--gpus',','.join(map(str,gpu_ids)),'--start',str(start),'--bank',bank_path,'--bank-sha',bank.manifest_sha256]
        native = root/'checkpoints'/f'global_step_{end}'
        if not native.exists():
            args = base+['--stage','train']
            if resume_update is not None:
                require(start<resume_update<end,'Recovery outside first incomplete window')
                args += ['--resume-update',str(resume_update)]
            else:
                require(not (root/'direction_batches'/f'u{start+1:04d}.pt').exists(),
                    'Partially trained window: explicit native --resume-update required; cannot replay initial batch')
            command(root,f'train-u{start:04d}-u{end:04d}',args)
        resume_update = None
        require(validate_full_checkpoint(native)['world_size']==len(gpu_ids),'Incomplete native endpoint')
        require((root/'metrics'/f'u{end:04d}.json').exists(),'Endpoint missing completed metrics')
        new = root/'models'/f'u{end:04d}'
        command(root,f'export-u{end:04d}',['phase2.export_model','--checkpoint',native,'--target',new])
        command(root,f'readout-u{end:04d}',base+['--stage','readout'])
        command(root,f'edit-gate-u{end:04d}',base+['--stage','evaluate'])
        sealed = strict_json(seal.read_text())
        bank = Bank.load(root/'banks'/f"{sealed['selected_bank_sha256']}.json",sealed['selected_bank_sha256'])
        cleanup(root,end)
    bank_path = bank.save(root/'banks')
    command(root,'final-u0050',['phase3.logicbench_loop','--setting',root/'setting.json','--root',root,
        '--gpus',','.join(map(str,gpu_ids)),'--start','45','--bank',bank_path,'--bank-sha',bank.manifest_sha256,'--stage','final'])
    status(root,'complete',training_started=True,completed_updates=50,final_report=str(root/'final'/'complete.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting',type=Path,default=ROOT/'configs/phase3_logicbench_50updates_s707_cpu_v2.json')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gpus',default='0,1,2,3')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume-update',type=int)
    parser.add_argument('--stage',choices=('train','readout','evaluate','final'))
    parser.add_argument('--start',type=int)
    parser.add_argument('--bank',type=Path)
    parser.add_argument('--bank-sha')
    args = parser.parse_args()
    load_editor_credential()
    setting = strict_json(args.setting.read_text())
    root = args.root.resolve()
    gpu_ids = list(map(int,args.gpus.split(',')))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,gpu_ids))
    if args.stage:
        require(setting['execution_approved'] is True,'Unapproved setting')
        bank = Bank.load(args.bank,args.bank_sha)
        if args.stage == 'train':
            train_stage(setting,root,bank,args.start,gpu_ids,args.resume_update)
        elif args.stage == 'readout':
            from .logicbench_loop_readout import predict
            old = Path(setting['model_path']) if args.start==0 else root/'models'/f'u{args.start:04d}'
            predict(bank,old,root/'models'/f'u{args.start+5:04d}',root/'direction_batches'/f'u{args.start+1:04d}.pt',
                root/'windows'/f'u{args.start:04d}-u{args.start+5:04d}'/'readout',start=args.start,end=args.start+5,
                tau_delta=setting['readout_tau_delta'])
        else:
            evaluate_stage(setting,root,bank,args.start,gpu_ids,final=args.stage=='final')
        return
    from skillnet_cohort.common import exclusive_writer
    with exclusive_writer(root):
        if args.execute:
            execute(setting,root,gpu_ids,args.resume_update)
        else:
            prepare(setting,root)
            check = readiness(setting,root,gpu_ids)
            status(root,check['status'],**{k:v for k,v in check.items() if k!='status'})


if __name__ == '__main__':
    main()
