"""Optional, run-scoped boundary handoff; no active worker hot-patching."""
import hashlib
import json
import os
import sys
from pathlib import Path


def dispatch_target(request, root, branch, start):
    if branch not in request['start_updates'] or start < request['start_updates'][branch]:
        return None
    if Path(root).resolve() != Path(request['run_root']) / 'runs' / branch or start % 5 or not 0 <= start < 20:
        raise ValueError('Unregistered RL-speed run/branch/boundary')
    candidate = Path(request['candidate'])
    script = candidate / 'alfworld_rl_speed/launch.py'
    if hashlib.sha256(script.read_bytes()).hexdigest() != request['source_hashes']['alfworld_rl_speed/launch.py']:
        raise ValueError('RL-speed launcher fingerprint changed')
    return candidate


def maybe_dispatch(preparation, root, branch, bank_path, bank_sha256, start, resume_update=None):
    root = Path(root).resolve()
    path = root.parents[1] / 'rl-speed-next-window-v1.json'
    if not path.is_file():
        return
    request = json.loads(path.read_text())
    candidate = dispatch_target(request, root, branch, start)
    if candidate is None:
        return
    argv = [sys.executable, '-B', '-m', 'alfworld_rl_speed.launch', '--request', str(path),
            '--preparation', str(preparation), '--root', str(root), '--branch', branch,
            '--bank-path', str(bank_path), '--bank-sha256', bank_sha256, '--start', str(start), '--execute']
    if resume_update is not None:
        argv += ['--resume-update', str(resume_update)]
    print(f'RL_SPEED_BOUNDARY branch={branch} start={start}; independent acceptance before training', flush=True)
    env = dict(os.environ, PYTHONPATH=str(candidate), PYTHONDONTWRITEBYTECODE='1')
    os.chdir(candidate)
    os.execve(sys.executable, argv, env)
