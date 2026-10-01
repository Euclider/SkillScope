"""Prepare an explicitly authorized U5 editor retry without model/API execution."""
import argparse
from pathlib import Path
import sqlite3

from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
from phase3.common import digest, require, strict_json, write_new
from phase3.logicbench_loop_data import ROOT
from skillnet_cohort.common import exclusive_writer, file_hash
from .runtime import authorize, verify_implementation


def prepare(root):
    root=Path(root).resolve()
    with exclusive_writer(root):
        window=root/'windows/u0000-u0005'
        require(not (window/'complete.json').exists() and not (window/'editor/proposal.json').exists(),
                'Recovery is only for an unfinished U5 proposal')
        require({p.name for p in (root/'metrics').glob('u*.json')}=={f'u{i:04d}.json' for i in range(1,6)},
                'Expected exactly five completed updates')
        checkpoints={str(i):validate_full_checkpoint(root/'checkpoints'/f'global_step_{i}') for i in (4,5)}
        require(all(v['world_size']==4 for v in checkpoints.values()),'Wrong checkpoint world size')
        exported=strict_json((root/'models/u0005/phase2_export.json').read_text())
        require(Path(exported['parent']).resolve()==root/'checkpoints/global_step_5', 'Foreign U5 export')
        readout=strict_json((window/'readout/readout.json').read_text())
        complete=strict_json((window/'readout/complete.json').read_text())
        require(digest(readout)==complete['readout_sha256'], 'Changed completed readout')
        source=strict_json((window/'readout/source.json').read_text())
        require(digest(source)==complete['source_sha256'] and source['start']==0 and source['end']==5,
                'Changed readout window')
        batch=root/'direction_batches/u0001.pt'
        require(file_hash(batch)==source['batch_sha256'],'Changed actual update batch')
        editor=strict_json((window/'editor/source.json').read_text())
        with sqlite3.connect(f'file:{root}/editor.sqlite3?mode=ro',uri=True) as db:
            failed=db.execute('select key,request from attempts order by id limit 1').fetchone()
        require(failed is not None,'Original attempt missing')
        request=strict_json(failed[1])
        require(request['payload']==editor['payload'] and request['identity']['bank_sha256']==editor['bank_sha256'],
                'Failed request differs from frozen editor evidence')
        files=sorted((ROOT/'phase3').glob('*.py'))+[ROOT/'verl/trainer/ppo/ray_trainer.py',ROOT/'verl/trainer/main_ppo.py',
            ROOT/'logicbench_phase12/rollout.py',ROOT/'phase2/stable_direction.py']
        current={str(p.relative_to(ROOT)):file_hash(p) for p in files}
        original=strict_json((root/'implementation.json').read_text())
        require(set(original)==set(current) and {k for k in current if current[k]!=original[k]} <=
                {'phase3/api.py','phase3/logicbench_loop.py'},'Unexpected implementation change')
        receipt=authorize(root,failed[0],current,authorization='User explicitly requested fixing the U5 timeout and resuming training on 2026-09-30',
                          timeout_seconds=600)
        verify_implementation(root,current)
        report={'state':'prepared_not_launched','completed_updates':5,'resume_stage':'U5 editor',
            'train_u0_u5_reused':True,'readout_reused':True,'editor_evidence_unchanged':True,
            'checkpoint_validation':checkpoints,'recovery_receipt_sha256':digest(receipt),
            'editor_timeout_seconds':600,'sdk_automatic_retries':0,'new_api_calls':0}
        write_new(root/'recovery/preparation.json',report)
        print(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.root)
