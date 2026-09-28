"""Whole-game evaluation against an explicit policy and evolving bank version."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .common import digest, require, strict_json, write_new


def _run_job(job, *, bank, checkpoint, data_root, policy, router):
    from phase1.eval_skill_margin import run_episode
    from phase1.conditions import SkillCondition
    from skillnet_cohort.common import file_hash
    game = (data_root / job['game']['game_id']).resolve()
    require(game.is_relative_to(data_root) and file_hash(game) == job['game']['game_sha256'],
            'Changed/foreign evaluation game')
    result = run_episode(policy=policy, game_file=game, memory=router.memory,
        condition=SkillCondition.FULL_BANK, skill_id='', environment_seed=job['environment_seed'],
        eval_seed=job['eval_seed'], temperature=.4, top_p=1., max_steps=50,
        max_new_tokens=512, history_length=2, step_skill_router=router,
        router_general_top_k=len(bank))
    require(type(result.get('success')) is bool and bool(result.get('steps')), 'Incomplete evaluator episode')
    for step in result['steps']:
        step.pop('prompt_text', None)
        sid = step.get('selected_skill_id')
        step['skill_version_sha256'] = bank.active_versions.get(sid)
    result.update(game_id=job['game']['game_id'], task_type=job['game']['task_type'],
                  eval_seed=job['eval_seed'], policy_sha256=job['policy_sha256'],
                  bank_sha256=bank.manifest_sha256)
    return result


def evaluate_bank(*, bank, checkpoint, policy_sha256, games, seeds, data_root, output, router_api,
                  inference_profile, gpu_ids=None):
    """No resources loaded when all immutable game/seed results already exist."""
    from skillnet_cohort.common import exclusive_writer
    from skillnet_cohort.inference import registration
    output, data_root = Path(output), Path(data_root).resolve()
    require(games and seeds and len(set(seeds)) == len(seeds), "Empty/duplicate evaluation plan")
    inference = registration(inference_profile)
    jobs = [{"game": game, "eval_seed": seed, "policy_sha256": policy_sha256, "bank_sha256": bank.manifest_sha256}
            for game in games for seed in seeds]
    for job in jobs:
        job['inference_profile_sha256'] = inference['sha256']
        job['environment_seed'] = inference['settings']['seed'] + 1000
    from .embedding_routing import EmbeddingRouterPool
    embedding_router = router_api.for_bank(bank) if isinstance(router_api, EmbeddingRouterPool) else None
    if embedding_router is not None:
        for job in jobs:
            job['router_protocol_sha256'] = embedding_router.protocol_hash
    require(len({(job["game"]["game_id"], job["eval_seed"]) for job in jobs}) == len(jobs), "Duplicate games")
    if gpu_ids is not None:
        require(embedding_router is not None and isinstance(gpu_ids, list)
                and len(gpu_ids) in (4, 8) and len(set(gpu_ids)) == len(gpu_ids)
                and all(type(gpu) is int and gpu >= 0 for gpu in gpu_ids)
                and len(jobs) >= len(gpu_ids), 'Parallel Phase3 evaluation requires explicit 4/8 GPU IDs and local router')
    resources, rows = {}, []
    with exclusive_writer(output):
        write_new(output / "plan.json", {"jobs": jobs})
        expected_files = {f'{digest(job)}.json' for job in jobs}
        require({path.name for path in (output / 'episodes').glob('*.json')} <= expected_files, 'Foreign evaluation result file')
        if gpu_ids is not None and expected_files - {path.name for path in (output / 'episodes').glob('*.json')}:
            bank_path = bank.save(output / 'bank')
            config = {'bank_path': str(bank_path), 'bank_sha256': bank.manifest_sha256,
                'checkpoint': str(Path(checkpoint).resolve()), 'data_root': str(data_root),
                'output': str(output.resolve()), 'jobs': jobs, 'inference_profile': str(Path(inference_profile).resolve()),
                'router_settings': router_api.settings, 'router_ledger': str(router_api.ledger.path.resolve()),
                'gpu_ids': gpu_ids}
            write_new(output / 'worker_config.json', config)
            logs = [output / 'logs' / f'shard-{shard:02d}.log' for shard in range(len(gpu_ids))]
            require(all(not log.exists() for log in logs),
                    'Prior evaluator shard log exists; reconcile before resubmission')
            processes = []
            for shard, gpu in enumerate(gpu_ids):
                log = logs[shard]
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open('x')
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONDONTWRITEBYTECODE': '1'}
                process = subprocess.Popen([sys.executable, '-B', '-m', 'phase3.evaluate', '--worker-config',
                    str(output / 'worker_config.json'), '--shard', str(shard)],
                    cwd=Path(__file__).resolve().parents[1], env=env, stdout=stream, stderr=subprocess.STDOUT)
                stream.close()
                processes.append(process)
            statuses = [process.wait() for process in processes]
            require(all(code == 0 for code in statuses), f'Phase3 evaluator shard failed: {statuses}; inspect shard logs')
        for job in jobs:
            target = output / "episodes" / f"{digest(job)}.json"
            if target.exists():
                record = strict_json(target.read_text())
                require(record["job"] == job and record["result_sha256"] == digest(record["result"]), "Changed evaluation result")
                rows.append(record["result"])
                continue
            require(gpu_ids is None, 'Parallel evaluator omitted an episode')
            if not resources:
                from skillnet_cohort.inference import make_policy
                from .routing import BranchRouter
                router = embedding_router or BranchRouter(bank, router_api)
                resources = {"policy": make_policy(checkpoint, {'inference_profile': inference}, 4096), "router": router,
                             }
            result = _run_job(job, bank=bank, checkpoint=checkpoint, data_root=data_root,
                              policy=resources['policy'], router=resources['router'])
            write_new(target, {"job": job, "result": result, "result_sha256": digest(result)})
            rows.append(result)
        require(len(rows) == len(jobs), "Incomplete evaluation")
        write_new(output / "complete.json", {"episodes": len(rows), "unique_games": len(games),
                  "success_rate": sum(row["success"] for row in rows) / len(rows), "plan_sha256": digest(jobs)})
    return rows


def _worker(config_path, shard):
    from .bank import Bank
    from .embedding_routing import EmbeddingRouterPool
    from skillnet_cohort.inference import make_policy, registration
    from skillnet_cohort.common import file_hash
    config = strict_json(Path(config_path).read_text())
    gpu_ids = config['gpu_ids']
    require(type(shard) is int and 0 <= shard < len(gpu_ids)
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu_ids[shard]), 'Wrong evaluator GPU shard')
    bank = Bank.load(config['bank_path'], config['bank_sha256'])
    output, data_root = Path(config['output']), Path(config['data_root']).resolve()
    require(strict_json((output / 'plan.json').read_text()) == {'jobs': config['jobs']}, 'Evaluator plan changed')
    router_pool = EmbeddingRouterPool(config['router_settings'], config['router_ledger'], bank.branch_id)
    try:
        router = router_pool.for_bank(bank)
        policy = None
        for job in config['jobs'][shard::len(gpu_ids)]:
            target = output / 'episodes' / f'{digest(job)}.json'
            if target.exists():
                record = strict_json(target.read_text())
                require(record['job'] == job and record['result_sha256'] == digest(record['result']),
                        'Changed evaluator shard result')
                continue
            game = (data_root / job['game']['game_id']).resolve()
            require(game.is_relative_to(data_root) and file_hash(game) == job['game']['game_sha256'],
                    'Changed evaluator shard game')
            if policy is None:
                policy = make_policy(config['checkpoint'],
                    {'inference_profile': registration(config['inference_profile'])}, 4096)
            result = _run_job(job, bank=bank, checkpoint=config['checkpoint'], data_root=data_root,
                              policy=policy, router=router)
            write_new(target, {'job': job, 'result': result, 'result_sha256': digest(result)})
    finally:
        router_pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-config', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    _worker(args.worker_config, args.shard)


if __name__ == '__main__':
    main()
