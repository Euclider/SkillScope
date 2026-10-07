"""Versioned small-cohort admission, train-first scheduling and parallel gold."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import os
import shutil
import sys

from webshop_phase12.assets import ROOT, RUN_ROOT
from webshop_phase12.coordinate_llm import verify_source
from webshop_phase12.run import now, pipeline, run_stage, wait_idle, write


CONFIG = 'webshop54_phase12_small_v4'


def parallel(stages):
    with ThreadPoolExecutor(max_workers=len(stages)) as executor:
        futures = [executor.submit(run_stage, *arguments) for arguments in stages]
        for future in futures:
            future.result()


def evaluations(root, prepared, gpus=(5, 7), *, smoke=False, reuse_locked=False):
    spec = json.loads((prepared/'manifest.json').read_text())
    seeds = list(spec['seeds'])
    # Both readouts are locked before the first independent reward is collected.
    jobs = []
    for i, seed in enumerate(seeds):
        locked_path=root/f'seed{seed}'/'readout/prediction-locked.json'
        if reuse_locked and locked_path.exists():
            locked=json.loads(locked_path.read_text())
            if locked['status']!='locked_before_independent_eval' or not locked['native_old_forward_parity_passed'] or locked['native_old_chosen_max_abs_difference']>1e-3:
                raise ValueError('Reusable lockedreadout is invalid')
            continue
        stage_root = root/'parallel'/f'readout-s{seed}'
        stage_root.mkdir(parents=True, exist_ok=False)
        path = root/f'seed{seed}'
        command = [sys.executable, '-B', '-m', 'webshop_phase12.readout', '--seed-dir', str(path),
                   '--new-model', str(path/'merged-endpoint')]
        jobs.append((f'seed{seed}-readout', command, stage_root, [gpus[i % len(gpus)]]))
    wait_idle(gpus, root, 'parallel-readout')
    write(root/'status.json', {'status': 'running', 'stage': 'parallel-readout', 'updated_at_utc': now()})
    if jobs:parallel(jobs)
    first = None
    for seed in seeds:
        path = root/f'seed{seed}'
        command = [sys.executable, '-B', '-m', 'webshop_phase12.evaluate', '--seed-dir', str(path),
                   '--new-model', str(path/'merged-endpoint'), '--prepared', str(prepared)]
        if smoke:
            command += ['--smoke']
        if first is None:
            wait_idle([gpus[0]], root, f'seed{seed}-U0-reference')
            run_stage(f'seed{seed}-U0-reference', command+['--stage', 'reference'], root, [gpus[0]], gpus[0])
        else:
            (path/'paired_eval').mkdir()
            for name in ('anchors.json', 'reference_trajectories.jsonl', 'reference-complete.json'):
                shutil.copy2(first/'paired_eval'/name, path/'paired_eval'/name)
            write(path/'paired_eval/reference-reuse.json', {'source': str(first/'paired_eval'),
                  'original_U0_reference_reused': True, 'source_file_hashes': {
                    n: hashlib.sha256((first/'paired_eval'/n).read_bytes()).hexdigest()
                    for n in ('anchors.json', 'reference-complete.json')}})
        jobs = []
        for shard, gpu in enumerate(gpus):
            stage_root = root/'parallel'/f'eval-s{seed}-shard{shard}'
            stage_root.mkdir(parents=True, exist_ok=False)
            args = command+['--stage', 'continuations', '--shard', str(shard), '--shards', str(len(gpus))]
            if first is not None:
                args += ['--reuse-old-from', str(first/'paired_eval')]
            jobs.append((f'seed{seed}-paired-shard{shard}', args, stage_root, [gpu], gpu))
        wait_idle(gpus, root, f'seed{seed}-parallel-paired')
        write(root/'status.json', {'status': 'running', 'stage': f'seed{seed}-parallel-paired', 'updated_at_utc': now()})
        parallel(jobs)
        run_stage(f'seed{seed}-paired-merge', command+['--stage', 'merge', '--shards', str(len(gpus))], root, [])
        run_stage(f'seed{seed}-metrics', [sys.executable, '-B', '-m', 'webshop_phase12.metrics', '--seed-dir', str(path)], root, [])
        first = path
    write(root/'complete.json', {'status': 'complete', 'smoke': smoke, 'seeds': seeds, 'finished_at_utc': now()})
    write(root/'status.json', {'status': 'complete', 'smoke': smoke, 'updated_at_utc': now()})


def main(queue, gpus, reuse_training=None):
    queue.mkdir(exist_ok=False)
    write(RUN_ROOT/'ACTIVE_WEBSHOP_RUN.json', {'pid': os.getpid(), 'queue': str(queue),
          'status_file': str(queue/'status.json'), 'source': str(ROOT), 'tmux_session': 'ws12-small-v4-20261007'})
    legacy=os.environ.get('WEBSHOP_LEGACY_RUN_ROOT')
    if legacy and Path(legacy).is_dir():
        write(Path(legacy)/'ACTIVE_WEBSHOP_RUN.json',json.loads((RUN_ROOT/'ACTIVE_WEBSHOP_RUN.json').read_text()))
    try:
        verify_source()
        probe = json.loads((RUN_ROOT/'fsdp-optimization-probe.json').read_text())
        if not all(row['passed'] for row in probe['results']['admission']):
            raise ValueError('Real two-rank optimization probe has not passed')
        smoke = queue/'smoke'
        write(queue/'status.json', {'status': 'running_preflight', 'formal_started': False, 'updated_at_utc': now()})
        if reuse_training is None:
            pipeline(smoke, RUN_ROOT/'prepared-smoke', smoke=True, gpus=gpus, evaluation_gpu=gpus[0],
                     router_gpu=(gpus[-1],), config=CONFIG, train_only=True)
        else:
            import math
            from webshop_phase12.router_cost import update_router_cost
            routing=update_router_cost(json.loads((reuse_training/'router-performance-train-s404.json').read_text()),
                                       (reuse_training/'seed404-training.log').read_text())
            registered=json.loads((reuse_training.parent/'registered-source-manifest-llm-v1.json').read_text())
            for row in registered['files']:
                if row['path'] in ('webshop_phase12/small_run.py','webshop_phase12/run.py'):continue
                if row['path']=='webshop_phase12/envs.py':
                    from webshop_phase12.replay_identity import environment_behavior_fingerprint
                    amendment=json.loads((RUN_ROOT/'environment-replay-amendment.json').read_text())
                    if amendment['training_source_sha256']!=row['sha256'] or environment_behavior_fingerprint((ROOT/row['path']).read_text())!=amendment['unchanged_training_behavior_sha256']:
                        raise ValueError('Environment behavior changed beyond transportreplay identity')
                    continue
                if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()!=row['sha256']:
                    raise ValueError('Reusable preflight runtime math/environment source changed')
            if hashlib.sha256((reuse_training.parent/'registered-frozen-router-model.json').read_bytes()).hexdigest()!=hashlib.sha256((RUN_ROOT/'frozen-router-model.json').read_bytes()).hexdigest():
                raise ValueError('Reusable preflight frozenmodel receipt changed')
            archive=json.loads((reuse_training/'seed404/phase2/batches/u0001/manifest.json').read_text())
            if archive['config']['webshop_phase12']['forward_contract']!='hf-bucket256-active-v4':
                raise ValueError('Reusable preflight numericalcontract differs')
            expected=math.ceil(archive['row_count']/128)
            for rank in range(len(gpus)):
                rows=[json.loads(x) for x in (reuse_training/f'seed404/phase2/optimizer_steps/u0001-rank{rank}.jsonl').read_text().splitlines()]
                if len(rows)!=expected or any(r['adam_step_after']!=i+1 or not math.isfinite(r['grad_norm']) for i,r in enumerate(rows)):
                    raise ValueError('Reusable preflight optimizerwitness incomplete')
            smoke.mkdir()
            (smoke/'seed404').symlink_to((reuse_training/'seed404').resolve(),target_is_directory=True)
            shutil.copy2(reuse_training/'protocol.json',smoke/'protocol.json')
            write(smoke/'actual-training-reused.json',{'source':str(reuse_training),'all_real_optimizer_steps_verified':expected,
                  'native_math_source_unchanged':True,'routing_execution_unchanged':True,'complete_RL_router_cost_admission':routing})
            checkpoint=smoke/'seed404/checkpoints/global_step_1/actor'
            if not (smoke/'seed404/merged-endpoint').is_dir():
                run_stage('seed404-merge',[sys.executable,'-B','scripts/model_merger.py','merge','--backend','fsdp',
                          '--local_dir',str(checkpoint),'--target_dir',str(smoke/'seed404/merged-endpoint')],smoke,[])
            else:
                write(smoke/'merged-endpoint-reused.json',{'source':str(reuse_training/'seed404/merged-endpoint'),
                      'same_endpoint_as_locked_readout':True})
        evaluations(smoke, RUN_ROOT/'prepared-smoke', gpus, smoke=True, reuse_locked=reuse_training is not None)
        path = smoke/'seed404'
        locked = json.loads((path/'readout/prediction-locked.json').read_text())
        paired = json.loads((path/'paired_eval/manifest.json').read_text())
        if not locked['native_old_forward_parity_passed'] or locked['native_old_chosen_max_abs_difference'] > 1e-3:
            raise ValueError('Native readout parity failed')
        if not all(paired[k] for k in ['same_prefix_replay_verified', 'same_visible_memory_replay_verified', 'bank_frozen', 'router_frozen']):
            raise ValueError('Preflight paired replay or frozen assets failed')
        if not list((path/'phase2/optimizer_steps').glob('*.jsonl')):
            raise ValueError('Preflight actual optimizer witness missing')
        write(queue/'preflight-admission.json', {'status': 'passed', 'native_old_chosen_max_abs_difference':
              locked['native_old_chosen_max_abs_difference'], 'forward_contract': locked['forward_contract'],
              'actual_optimizer_and_parallel_pairing_verified': True, 'registered_at_utc': now()})
        # Only new explicitly-disposable preflight weights are reclaimed. Preserve all evidence.
        native = (path/'checkpoints').resolve()
        if not str(native).startswith('/dev/shm/wangyifan-webshop-phase12-20261006-'):
            raise ValueError('Unexpected temporary preflight namespace')
        shutil.rmtree(native)
        shutil.rmtree(path/'merged-endpoint')
        write(queue/'disposable-preflight-weights-reclaimed.json', {'native': str(native),
              'endpoint': str(path/'merged-endpoint'), 'new_preflight_only': True, 'evidence_and_logs_retained': True})
        verify_source()
        formal = queue/'formal-s404-s505'
        write(queue/'status.json', {'status': 'formal_started', 'formal_started': True, 'root': str(formal), 'updated_at_utc': now()})
        pipeline(formal, RUN_ROOT/'prepared', gpus=gpus, evaluation_gpu=gpus[0], router_gpu=(gpus[-1],), config=CONFIG, train_only=True)
        evaluations(formal, RUN_ROOT/'prepared', gpus)
        write(queue/'status.json', {'status': 'complete', 'formal_started': True, 'updated_at_utc': now()})
    except BaseException as error:
        write(queue/'status.json', {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
              'automatic_retries': 0, 'updated_at_utc': now()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--gpus', default='5,7')
    parser.add_argument('--reuse-preflight-training', type=Path)
    args = parser.parse_args()
    main(args.queue, tuple(map(int, args.gpus.split(','))), args.reuse_preflight_training)
