"""One explicit eight-GPU precision run, followed by immutable CPU reporting."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import REPO, exclusive_writer, file_hash, read_json, write_new_json
from .utility_precision import (DEFAULT_OUTPUT, NEW_GOLD, binding, capacity,
                                jobs_with_ids, prepare, scope)


def worker(output, update, shard):
    if update not in (0, 5) or shard not in range(8) or os.environ.get('CUDA_VISIBLE_DEVICES') != str(shard):
        raise PermissionError('Only registered U0/U5 single-GPU workers')
    plan = binding(output); config = read_json(output/'protocol.json')
    intent = output/'attempts'/f'u{update:04d}-shard{shard}.json'
    if intent.exists():
        raise FileExistsError('Never implicitly retry an attempted worker')
    write_new_json(intent, {'pid': os.getpid(), 'started_unix': time.time(),
        'plan_sha256': file_hash(output/'plan.json'), 'update': update, 'shard': shard})
    from .first_calls_recovery import audit_endpoint
    audit = audit_endpoint(output, update)
    if audit['complete_shards'] or audit['completed'] != plan['reused_continuations']//2:
        raise ValueError('Worker requires exactly the prepared retained boundary')
    # Explicit barrier: even a fast worker must not append while a peer is still
    # auditing the common prepared boundary. A failed peer stops the parent group.
    admitted = output/'admitted'/f'u{update:04d}-shard{shard}.json'
    write_new_json(admitted, {'pid': os.getpid(), 'update': update, 'shard': shard,
                             'prepared_rows': audit['completed']})
    while not all((output/'admitted'/f'u{update:04d}-shard{rank}.json').exists() for rank in range(8)):
        time.sleep(1)
    os.environ['ALFWORLD_DATA'] = config['runtime']['data_root']
    from .first_calls_port_recovery import build_file_engine, RECORD_ENV
    rendezvous = output/'rendezvous'/f'u{update:04d}-shard{shard}.json'
    rendezvous.parent.mkdir(parents=True, exist_ok=True)
    os.environ[RECORD_ENV] = str(rendezvous)
    import torch
    torch.set_num_threads(1)
    from .runtime import BoundedPolicy, make_runtime
    from .vllm_backend import VLLMPolicy
    from phase1.eval_first_invocation_utility import run_branch
    from phase1.first_invocation import PayloadArm
    from phase1.archive import append_jsonl_idempotent, utc_now
    from phase2.measure import phase
    class FilePolicy(VLLMPolicy):
        def __init__(self, checkpoint, profile):
            self.engine = build_file_engine(checkpoint, profile)
            self.tokenizer = self.engine.get_tokenizer()
    ev = config['evaluation']
    memory, router = make_runtime(config['runtime'])
    policy = BoundedPolicy(FilePolicy(output/'models'/f'u{update:04d}',
                           config['runtime']['inference_profile']), ev['max_prompt_tokens'])
    write_new_json(output/'ready'/f'u{update:04d}-shard{shard}.json',
        {'pid': os.getpid(), 'ready_unix': time.time(), 'update': update, 'shard': shard,
         'rendezvous_sha256': file_hash(rendezvous), 'engine': 'vllm_FileStore_TP1'})
    directory = output/'evaluations'/f'u{update:04d}'; index = directory/f'shard-{shard}.jsonl'
    old_rows = [json.loads(line) for line in index.read_text().splitlines()]
    retained = {r['trajectory_id'] for r in old_rows}
    jobs = [r for r in jobs_with_ids(config, update) if r[0] == shard]
    missing = [r for r in jobs if r[1] not in retained]
    if any(r[2]['purpose'] != 'gold' or r[2]['continuation_seed'] not in NEW_GOLD for r in missing):
        raise ValueError('Do not reexecute evidence or old gold')
    for count, (_, tid, identity, job) in enumerate(missing, 1):
        skill, anchor, purpose, seed, arm, placebo = job
        path = directory/'trajectories'/skill/(tid+'.json')
        if path.exists():
            raise FileExistsError('Orphan or already-attempted continuation; no overwrite/retry')
        result = run_branch(policy=policy, anchor={**anchor, 'source_eval_seed': seed},
            memory=memory, router=router, arm=PayloadArm(arm), target_skill_id=skill,
            placebo_text=placebo['text'], temperature=ev['temperature'], top_p=ev['top_p'],
            max_new_tokens=ev['max_new_tokens'], history_length=ev['history_length'], router_general_top_k=37)
        row = {**identity, 'trajectory_id': tid, 'game_id': anchor['game_id'], 'state_id': anchor['state_id'],
            'trigger_step': anchor['trigger_step'], 'phase': phase(anchor['trigger_step']),
            'trajectory_path': str(path), 'created_at': utc_now(), 'context_id': anchor['context_id'],
            'rl_path_id': config['rl_path_id'], **{k: v for k, v in result.items() if k != 'steps'}}
        write_new_json(path, {**row, 'original_anchor': anchor, 'actual_continuation_seed': seed, 'steps': result['steps']})
        append_jsonl_idempotent(index, [row], unique_fields=('trajectory_id',))
        print(f'UTILITY u{update} shard{shard}: new={count}/{len(missing)} retained={len(retained)} '
              f'{skill} seed={seed} arm={arm} success={result["success"]}', flush=True)
    write_new_json(directory/f'shard-{shard}-complete.json', {'created_at': utc_now(),
        'jobs': len(jobs), 'shard': shard, 'shards': 8, 'max_jobs': None,
        'protocol_sha256': file_hash(output/'protocol.json'), 'new_continuations': len(missing),
        'reused_continuations': len(retained)})


def environment(gpu=''):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTHONPATH=str(REPO)+os.pathsep+env.get('PYTHONPATH', ''))
    env.pop('SKILLNET_PHASE12_RECOVERY_AUTHORIZATION', None)
    env.pop('SKILLSCOPE_EVAL_RENDEZVOUS_RECORD', None)
    return env


def commands(output, jobs, stage):
    from .runtime_watch import StageWatch
    plan = binding(output); capacity(output, plan)
    watch = StageWatch(output, output, stage, None); children = []; next_disk = 0
    try:
        for args, label, gpu in jobs:
            path = output/'logs'/(label+'.log'); path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('x')
            try:
                p = subprocess.Popen([sys.executable, '-u', '-B', '-m',
                    'skillnet_cohort.utility_precision_run', *args, '--output', str(output)],
                    cwd=REPO, env=environment(gpu), stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                stream.close(); raise
            children.append((p, stream, label))
        write_new_json(output/'stages'/(stage+'.json'), {'started_unix': time.time(),
            'children': [{'pid': p.pid, 'label': label} for p, _, label in children]})
        watch.tick(children, event='started', force=True)
        while any(p.poll() is None for p, _, _ in children):
            if any(p.poll() not in (None, 0) for p, _, _ in children):
                raise RuntimeError('A child failed; stop this new group and preserve evidence, no retry')
            if time.time() >= next_disk:
                capacity(output, plan); next_disk = time.time()+60
            watch.tick(children); time.sleep(1)
        if any(p.returncode != 0 for p, _, _ in children):
            raise RuntimeError('Stage failed; see preserved child logs')
    finally:
        for p, _, _ in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)  # Only this invocation's own process groups.
        for p, stream, _ in children:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            stream.close()
        watch.tick(children, event='finished' if all(p.returncode == 0 for p, _, _ in children) else 'stopped', force=True)


def run(output):
    from .first_calls_defer import gpu_users
    from .first_calls_recovery import audit_endpoint
    plan = binding(output, retained=True, models=True)
    if gpu_users() or (output/'run-intent.json').exists():
        raise PermissionError('GPUs must be idle and this must be a fresh explicit attempt')
    stage = 'admission'
    with exclusive_writer(output):
        capacity(output, plan, reserve_new=True)
        write_new_json(output/'run-intent.json', {'pid': os.getpid(), 'started_unix': time.time(),
            'plan_sha256': file_hash(output/'plan.json'), 'new_training': False, 'hard_limit_seconds': None})
        try:
            # No gold-label comparisons until both complete endpoints exist.
            for update in (0, 5):
                stage = f'utility-u{update:04d}'
                audit = audit_endpoint(output, update)
                if audit['completed'] != plan['reused_continuations']//2 or audit['complete_shards']:
                    raise ValueError('Unexpected prelaunch endpoint boundary')
                commands(output, [(['worker', '--update', str(update), '--shard', str(rank)],
                    f'u{update:04d}-shard{rank}', rank) for rank in range(8)], stage)
                audit = audit_endpoint(output, update)
                if audit['missing'] or audit['complete_shards'] != list(range(8)):
                    raise ValueError('Incomplete endpoint after workers returned')
                write_new_json(output/'audits'/f'u{update:04d}.json', audit)
            stage = 'analysis'
            commands(output, [(['analyze'], 'analysis', '')], stage)
            binding(output, retained=True, models=True)
            paths = [p for folder in ('reports', 'precision', 'window_metrics', 'audits', 'reuse')
                     for p in sorted((output/folder).rglob('*')) if p.is_file()]
            paths += [output/n for n in ('plan.json', 'protocol.json', 'predictor-lock.json',
                'skill_scores.csv', 'registry.csv', 'independent-verification.json', 'analysis-complete.json',
                'bootstrap-receipts.json', 'bootstrap-label-draws-placebo.parquet', 'bootstrap-label-draws-null.parquet')]
            write_new_json(output/'provenance.json', {'plan_sha256': file_hash(output/'plan.json'),
                'scientific_status': 'ANALYZED_fixed_predictors_posthoc_precision_amendment',
                'files': [{'path': str(p.relative_to(output)), 'sha256': file_hash(p)} for p in paths]})
            write_new_json(output/'complete.json', {'status': 'complete', 'finished_unix': time.time(),
                'seed': 404, 'gold_repeats': 8, 'new_continuations': plan['new_continuations'],
                'total_continuations': plan['total_continuations'], 'old_files_preserved': True,
                'provenance_sha256': file_hash(output/'provenance.json'), 'other_seeds_started': False})
            print('COMPLETE: '+str(output/'reports/phase2-results.md'), flush=True)
        except BaseException as error:
            write_new_json(output/'stopped.json', {'stage': stage, 'error': repr(error),
                'stopped_unix': time.time(), 'automatic_retry': False})
            raise


def launch(output):
    from .first_calls_defer import gpu_users, process_identity
    plan = binding(output)
    if gpu_users() or (output/'launch.json').exists() or (output/'launch-intent.json').exists():
        raise PermissionError('Only a fresh explicit launch with idle GPUs')
    capacity(output, plan, reserve_new=True)
    write_new_json(output/'launch-intent.json', {'requested_unix': time.time(), 'automatic_retry': False,
        'command': [sys.executable, '-u', '-B', '-m', 'skillnet_cohort.utility_precision_run',
                    'run', '--output', str(output)]})
    with (output/'workflow.log').open('x') as stream:
        p = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'skillnet_cohort.utility_precision_run',
            'run', '--output', str(output)], cwd=REPO, env=environment(), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write_new_json(output/'launch.json', {'pid': p.pid, 'identity': process_identity(p.pid),
        'started_unix': time.time(), 'plan_sha256': file_hash(output/'plan.json')})
    print(json.dumps({'status': 'LAUNCHED_NOT_COMPLETE', 'pid': p.pid, 'root': str(output)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'launch', 'run', 'worker', 'analyze'))
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--tests', type=Path)
    p.add_argument('--update', type=int, choices=(0, 5))
    p.add_argument('--shard', type=int, choices=range(8))
    args = p.parse_args(); output = scope(args.output)
    if args.mode == 'prepare':
        if not args.tests:
            p.error('--tests required for preparation')
        prepare(output, args.tests)
    elif args.mode == 'launch':
        launch(output)
    elif args.mode == 'run':
        def stop(signum, frame):
            raise InterruptedError('Termination requested; retain all evidence')
        signal.signal(signal.SIGTERM, stop)
        run(output)
    elif args.mode == 'worker':
        worker(output, args.update, args.shard)
    else:
        from .utility_precision_analysis import report
        result = report(output)
        write_new_json(output/'analysis-complete.json', {'status': 'complete', **result})


if __name__ == '__main__':
    main()
