"""Explicit recovery of the pre-training U15 speed-deployment import failure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from phase3.common import require, strict_json, write_new
from skillnet_cohort.common import exclusive_writer, file_hash

ROOT = Path('/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v7-20260927')
OUTPUT = ROOT / 'recovery-u15-speed-import-v1'
ARMS = ('skillrl_failure', 'readout_magnitude', 'readout_gated_d', 'readout_p', 'readout_c')


def now():
    return datetime.now(timezone.utc).isoformat()


def recovery_log(args, log, *, run_root, output, expected_failure_sha256):
    """Redirect only this proven import failure, never an arbitrary failed job."""
    run_root, log = Path(run_root), Path(log)
    if log != run_root / 'logs/train-u0015-u0020.log':
        return log
    args = list(map(str, args))
    require(args[0] == 'phase3.training' and args[args.index('--start') + 1] == '15'
            and args[args.index('--branch') + 1] == 'skillrl_failure'
            and Path(args[args.index('--root') + 1]) == run_root,
            'Recovery is restricted to SkillRL U15-to-U20 training')
    require(file_hash(log) == expected_failure_sha256
            and "ModuleNotFoundError: No module named 'gigpo'" in log.read_text(),
            'Original pre-training import failure changed')
    require(not (run_root / 'metrics/u0016.json').exists()
            and not (run_root / 'direction_batches/u0016.pt').exists()
            and not (run_root / 'checkpoints/global_step_20').exists(),
            'Training advanced: do not replay this block')
    return Path(output) / 'train-u0015-u0020-recovery.log'


def preflight():
    from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
    request_path = ROOT / 'rl-speed-next-window-v1.json'
    request = strict_json(request_path.read_text())
    candidate = Path(request['candidate'])
    proof_path = candidate.parent / 'dependency-completion-v1.json'
    proof = strict_json(proof_path.read_text())
    require(file_hash(request_path) == proof['training_request_sha256'], 'Training request changed')
    for relative, expected in proof['added_files'].items():
        require(file_hash(candidate / relative) == expected
                and file_hash(REPO / relative) == expected, 'Deployment dependency differs from source')
    run = ROOT / 'runs/skillrl_failure'
    require((ROOT / 'runs/readout_d/milestones/u0020/complete.json').is_file(), 'Prior arm incomplete')
    require((run / 'events/u0015/complete.json').is_file(), 'U15 evolution is not sealed')
    checkpoint = validate_full_checkpoint(run / 'checkpoints/global_step_15')
    require(checkpoint['world_size'] == 8, 'Expected eight checkpoint shards')
    require((run / 'models/u0015/phase2_export.json').is_file(), 'Missing U15 export')
    failure = run / 'logs/train-u0015-u0020.log'
    require(file_hash(failure) == proof['failed_training_log_sha256'], 'Failure evidence changed')
    code = '''
from pathlib import Path
import sys
from alfworld_rl_speed.launch import checked_request, accept
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from agent_system.environments import make_envs
from phase3 import training
from gigpo import core_gigpo
request = checked_request(Path(sys.argv[1]), sys.argv[2], 'skillrl_failure', 15)
receipt_path, receipt = accept(request)
assert Path(core_gigpo.__file__).resolve().is_relative_to(Path.cwd())
assert callable(run_ppo) and callable(training.execute) and callable(make_envs)
print('COMPLETE_ENTRYPOINT_AND_EXISTING_ACCEPTANCE_VERIFIED')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(request_path), str(run)],
        cwd=candidate, env={**os.environ, 'PYTHONPATH': str(candidate), 'CUDA_VISIBLE_DEVICES': '',
            'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1'}, text=True, capture_output=True, timeout=180)
    require(result.returncode == 0, 'Recovery preflight failed:\n' + result.stdout + result.stderr)
    require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader'], text=True).strip(), 'GPU workers still active')
    return {'utc': now(), 'checkpoint': checkpoint, 'proof_sha256': file_hash(proof_path),
            'failed_training_log_sha256': file_hash(failure), 'preflight_stdout': result.stdout,
            'execution_order': list(ARMS), 'stop_update': 20, 'completed_updates_replayed': False,
            'u15_editor_and_gate_replayed': False, 'prior_acceptance_reused': True}


def launch():
    require(not OUTPUT.exists(), 'Recovery directory already exists; inspect it before any retry')
    record = preflight()
    env = dict(os.environ)
    credential = env.get('SKILLRL_PHASE3_EDITOR_API_KEY') or getpass.getpass('Editor credential (hidden): ')
    require(bool(credential.strip()), 'Missing editor credential')
    env.update(SKILLRL_PHASE3_EDITOR_API_KEY=credential.strip(),
        ALFWORLD_DATA='/mnt/workspace/users/wangyifan/skill-RL/data/alfworld',
        XDG_CACHE_HOME=str(ROOT / 'cache'), TRITON_CACHE_DIR=str(ROOT / 'cache/triton'),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    OUTPUT.mkdir(exist_ok=False)
    write_new(OUTPUT / 'launch.json', record)
    with (OUTPUT / 'queue.log').open('xb') as log:
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--queue'],
            cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    write_new(OUTPUT / 'process.json', {'pid': child.pid, 'utc': now()})
    print(f'RESUMED_QUEUE pid={child.pid} logs={OUTPUT}', flush=True)


def queue():
    with exclusive_writer(OUTPUT):
        for branch in ARMS:
            print(f'{now()} START {branch}', flush=True)
            with (OUTPUT / f'{branch}-runner.log').open('xb') as log:
                code = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--branch', branch],
                    cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT).returncode
            if code or not (ROOT / 'runs' / branch / 'milestones/u0020/complete.json').is_file():
                write_new(OUTPUT / 'stopped.json', {'branch': branch, 'returncode': code, 'utc': now()})
                raise SystemExit(1)
            print(f'{now()} SEALED {branch} U20', flush=True)
        write_new(OUTPUT / 'complete.json', {'utc': now(), 'arms': list(ARMS), 'update': 20})


def branch_worker(branch):
    from phase3 import run
    record = strict_json((OUTPUT / 'launch.json').read_text())
    proof_path = Path(strict_json((ROOT / 'rl-speed-next-window-v1.json').read_text())['candidate']).parent / 'dependency-completion-v1.json'
    require(file_hash(proof_path) == record['proof_sha256'], 'Recovery dependency receipt changed')
    proof = strict_json(proof_path.read_text())
    for relative, expected in proof['added_files'].items():
        require(file_hash(proof_path.parent / 'candidate' / relative) == expected, 'Recovery dependency changed')
    original_command = run.command
    def command(args, log):
        target = recovery_log(args, log, run_root=ROOT / 'runs/skillrl_failure', output=OUTPUT,
            expected_failure_sha256=record['failed_training_log_sha256'])
        return original_command(args, target)
    run.command = command
    run.execute(ROOT / 'assets-launch-v1/manifest.json', ROOT / 'runs' / branch, branch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--queue', action='store_true')
    modes.add_argument('--branch', choices=ARMS)
    args = parser.parse_args()
    if args.check:
        print(preflight()['preflight_stdout'])
    elif args.queue:
        queue()
    elif args.branch:
        branch_worker(args.branch)
    else:
        launch()
