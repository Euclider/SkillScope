"""Small outcome-blind ALFWorld timing probe; never a formal evaluation result.

Two complete episodes per GPU: eight train-temperature and eight unseen-eval
episodes, selected by a fixed task-balanced hash rule before any outcomes. Uses
the actual frozen policy/router and the same evaluator shard/writer machinery.
No optimizer, protocol edits, external API or automatic failed-job resume.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time

from .common import digest, file_hash, load_preparation, read_json, write_new_json
from .evaluate import execute_jobs, finalize_shards, partition_plan


SELECTION = 'engineering-timing-s404-task-balanced-v1'


def select_games(games, count=8):
    groups = defaultdict(list)
    for game in games:
        groups[game['task_type']].append(game)
    for rows in groups.values():
        rows.sort(key=lambda row: digest([SELECTION, row['game_id']]))
    selected = []
    offset = 0
    while len(selected) < count:
        added = False
        for task in sorted(groups):
            if offset < len(groups[task]) and len(selected) < count:
                selected.append(groups[task][offset])
                added = True
        if not added:
            raise ValueError('Insufficient games for the fixed engineering probe')
        offset += 1
    return selected


def plan(preparation):
    preparation = Path(preparation).resolve()
    load_preparation(preparation)
    spec = read_json(preparation.parent / 'spec.json')
    inventory = read_json(preparation.parent / 'games.json')
    if (spec['seed'] != 404 or spec['training']['iterations'] != 5
            or spec['training']['gpus'] != 8
            or spec['router_backend'] != 'skillrl_embedding_state_batch'
            or not spec.get('inference_profile')):
        raise ValueError('Probe only applies to the confirmed eight-GPU vLLM setting')
    identity = {'purpose': 'engineering_timing_only', 'split': 'train_and_unseen_probe',
                'preparation_sha256': file_hash(preparation), 'selection_rule': SELECTION}
    jobs = []
    for split, temperature in [('train', 1.0), ('valid_unseen', spec['evaluation']['temperature'])]:
        for game in select_games(inventory['splits'][split]['games']):
            row = {**identity, **game, 'source_split': split, 'temperature': temperature,
                   'environment_seed': 1404, 'eval_seed': 714040 + 100 * len(jobs)}
            jobs.append({**row, 'job_id': digest(row)[:32]})
    return {'identity': identity, 'jobs': jobs, 'game_count': len(jobs),
            'expected_episodes': len(jobs), 'engineering_only': True,
            'formal_evaluation': False, 'optimizer_updates': 0,
            'preparation': str(preparation), 'model_path': spec['model_path']}


def require_native_pass(gpu_preflight):
    root = Path(gpu_preflight).resolve()
    done, exited = read_json(root / 'complete.json'), read_json(root / 'process-exit.json')
    if (done.get('status') != 'PASS' or done.get('world_size') != 8
            or exited.get('returncode') != 0 or done.get('training_horizon_configured') != 5
            or file_hash(root / 'run.log') != exited['run_log_sha256']):
        raise PermissionError('A completed, unchanged eight-GPU native/vLLM preflight is required')
    launch = read_json(root / 'launch.json')
    from .common import REPO
    for source, expected in launch['source_sha256'].items():
        if file_hash(REPO / source) != expected:
            raise PermissionError('Runtime/test sources changed after native preflight')
    for rank in range(8):
        row = read_json(root / f'rank-{rank}.json')
        if (row.get('status') != 'PASS' or row.get('inference_backend') != 'vllm_v1'
                or row.get('synthetic_optimizer_steps') != 2):
            raise PermissionError('Incomplete native rank result')
    return {'root': str(root), 'complete_sha256': file_hash(root / 'complete.json'),
            'launch_sha256': file_hash(root / 'launch.json')}


class TimedPolicy:
    def __init__(self, policy, directory):
        self.policy, self.directory = policy, Path(directory)
        self.tokenizer = policy.tokenizer
        self.calls = []

    def generate(self, *args, **kwargs):
        started = time.monotonic()
        result = self.policy.generate(*args, **kwargs)
        row = {'request_index': len(self.calls), 'seconds': time.monotonic() - started,
               'prompt_tokens': result[1], 'completion_tokens': result[2]}
        write_new_json(self.directory / f'{len(self.calls):04d}.json', row)
        self.calls.append(row)
        return result


def execute(preparation, root, gpu_preflight, shard):
    import os
    import torch
    from .inference import make_policy
    from .runtime import make_runtime, runtime_settings
    from phase1.conditions import SkillCondition
    from phase1.eval_skill_margin import run_episode

    if shard not in range(8) or torch.cuda.device_count() != 1:
        raise ValueError('Each engineering shard must own exactly one assigned GPU')
    torch.set_num_threads(1)
    full = plan(preparation)
    proof = require_native_pass(gpu_preflight)
    spec = read_json(Path(preparation).resolve().parent / 'spec.json')
    if read_json(Path(gpu_preflight) / 'launch.json')['preparation_sha256'] != full['identity']['preparation_sha256']:
        raise PermissionError('Native preflight belongs to a different preparation')
    root = Path(root).resolve()
    output = root / 'shards' / f'{shard:02d}'
    if output.exists():
        raise FileExistsError('Do not resume or overwrite a failed engineering shard')
    part = partition_plan(full, shard, 8)
    write_new_json(output / 'launch.json', {'native_preflight': proof, 'engineering_only': True,
                   'formal_rl_started': False, 'shard': shard, 'plan_sha256': digest(part)})
    os.environ['ALFWORLD_DATA'] = spec['data_root']
    settings = spec['evaluation']
    started = time.monotonic()
    memory, router = make_runtime(runtime_settings(spec, root / 'router.sqlite3', max_local_calls=800))
    policy = make_policy(spec['model_path'], spec, settings['max_prompt_tokens'])
    init_seconds = time.monotonic() - started

    def runner(job):
        game = Path(spec['data_root']) / job['game_id']
        if file_hash(game) != job['game_sha256']:
            raise ValueError('Game content changed after preparation')
        timed = TimedPolicy(policy, output / 'generation_progress' / job['job_id'])
        begin = time.monotonic()
        result = run_episode(policy=timed, game_file=game, memory=memory,
            condition=SkillCondition.FULL_BANK, skill_id='',
            environment_seed=job['environment_seed'], eval_seed=job['eval_seed'],
            temperature=job['temperature'], top_p=settings['top_p'],
            max_steps=settings['max_steps'], max_new_tokens=settings['max_new_tokens'],
            history_length=settings['history_length'], step_skill_router=router, router_general_top_k=37)
        result['engineering_timing'] = {'elapsed_seconds': time.monotonic() - begin,
            'generation_seconds': sum(row['seconds'] for row in timed.calls),
            'generation_requests': timed.calls, 'formal_evaluation': False}
        print(f'PROBE shard={shard} split={job["source_split"]} steps={len(result["steps"])} '
              f'output_tokens={result["completion_tokens"]}', flush=True)
        return result

    execute_jobs(part, output, runner)
    write_new_json(output / 'timing.json', {'initialization_seconds': init_seconds,
        'total_seconds': time.monotonic() - started, 'engineering_only': True,
        'formal_rl_iterations': 0, 'external_api_calls': 0})
    router.close()


def main():
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpu-preflight', type=Path)
    p.add_argument('--shard', type=int)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--collect', action='store_true')
    a = p.parse_args()
    if a.collect:
        if a.execute or a.shard is not None:
            p.error('Collection is separate from shard execution')
        print(json.dumps(finalize_shards(plan(a.preparation), a.root, 8), indent=2))
    elif a.execute:
        if a.gpu_preflight is None or a.shard is None:
            p.error('Explicit completed GPU preflight and shard required')
        execute(a.preparation, a.root, a.gpu_preflight, a.shard)
    else:
        print(json.dumps(plan(a.preparation), indent=2))


if __name__ == '__main__':
    main()
