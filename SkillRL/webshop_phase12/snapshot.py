"""Create local model/source receipts before the frozen WebShop run."""
import argparse
import hashlib
import json
from pathlib import Path
from webshop_phase12.assets import ROOT,RUN_ROOT,BASE_MODEL


def file_hash(path):
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):sha.update(block)
    return sha.hexdigest()


def model_receipt(model):
    model=Path(model).resolve();files=[]
    for path in sorted(model.iterdir()):
        if path.is_file() and path.suffix in ('.json','.safetensors','.jinja','.txt'):
            stat=path.stat()
            files.append({'path':path.name,'sha256':file_hash(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})
    if not any(row['path'].endswith('.safetensors') for row in files):
        raise ValueError('Local frozen-model directory has no safetensors weights')
    return {'model':'Qwen/Qwen3.5-4B','path':str(model),'source':'original local U0, no router fine-tuning','files':files}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,default=BASE_MODEL)
    args=parser.parse_args();RUN_ROOT.mkdir(parents=True,exist_ok=True)
    model_path=RUN_ROOT/'frozen-router-model.json';source_path=RUN_ROOT/'source-manifest-llm-v1.json'
    if model_path.exists() or source_path.exists():
        raise ValueError('Snapshot receipts already exist; select a fresh WEBSHOP_RUN_ROOT')
    model_path.write_text(json.dumps(model_receipt(args.model),indent=2)+'\n')
    files=set()
    for directory in ('webshop_phase12','tests/webshop_phase12'):
        files.update(path.relative_to(ROOT) for path in (ROOT/directory).rglob('*') if path.is_file() and path.suffix in ('.py','.json'))
    files.update(map(Path,('agent_system/memory/router_cache.py','agent_system/memory/frozen_skill_bank.py',
        'agent_system/multi_turn_rollout/rollout_loop.py','verl/trainer/main_ppo.py','verl/trainer/ppo/ray_trainer.py',
        'verl/workers/rollout/hf_rollout.py','verl/workers/actor/dp_actor.py',
        'verl/workers/actor/padded_forward.py','verl/workers/fsdp_workers.py',
        'verl/utils/checkpoint/fsdp_checkpoint_manager.py','phase2/capture.py',
        'verl/trainer/config/webshop54_phase12_v1.yaml','verl/trainer/config/webshop54_phase12_accel_v2.yaml',
        'memory_data/webshop/claude_style_skills.json')))
    source_path.write_text(json.dumps({'source_copy':str(ROOT),'protocol':'webshop-visible-evidence-frozen-qwen35-v1',
        'files':[{'path':str(path),'sha256':file_hash(ROOT/path)} for path in sorted(files)]},indent=2)+'\n')
    print(json.dumps({'model_receipt':str(model_path),'source_receipt':str(source_path),'source_files':len(files)}))


if __name__=='__main__':main()
