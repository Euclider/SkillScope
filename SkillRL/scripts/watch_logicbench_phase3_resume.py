"""Wait for the frozen Phase3 GPU placement, then resume once (no retries).

Run as a module from the repository with the frozen training Python.
This outer queue changes process environment only, not registered run sources.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

from scripts.watch_sra_logicbench_gpus import gpu_snapshot, eligible_gpus, _write_json

REPO = Path(__file__).resolve().parents[1]
CACHE = Path('/home/wangyifan/.cache/skillrl-tiktoken')
CACHE_FILE = 'fb374d419588a4632f3f557e76b4b70aebbca790'
CACHE_SHA256 = '446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d'


def launch_environment(source, cache):
    result = {key: value for key, value in source.items()
              if key.lower() not in {'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'}}
    result['TIKTOKEN_CACHE_DIR'] = str(cache)
    result['PYTHONDONTWRITEBYTECODE'] = '1'
    return result


def validate_recovery(root):
    from logicbench_rl_speed.runtime import validate
    validate(root)
    if hashlib.sha256((CACHE / CACHE_FILE).read_bytes()).hexdigest() != CACHE_SHA256:
        raise ValueError('Tiktoken cache hash mismatch')
    # Verify the original failure path without permitting a cache-miss download.
    import tiktoken
    import tiktoken.load
    original = tiktoken.load.read_file
    def no_download(*args, **kwargs):
        raise RuntimeError('Offline tokenizer cache validation attempted network access')
    tiktoken.load.read_file = no_download
    try:
        tiktoken.encoding_for_model('gpt-5.5').encode('LogicBench resume check')
    finally:
        tiktoken.load.read_file = original


def wait_and_launch(state_dir, gpu_ids, command, environment, disk_check,
                    *, poll_seconds=60, before_launch=lambda: None):
    if poll_seconds <= 0 or len(gpu_ids) < 2 or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('Invalid monitor configuration')
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / 'watch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        streak = 0
        def record(status):
            status.update(updated_at=datetime.now(timezone.utc).isoformat(),
                          watcher_pid=os.getpid(), required_gpu_ids=gpu_ids)
            _write_json(state_dir / 'status.json', status)
            with (state_dir / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(status, sort_keys=True) + '\n')
        while True:
            try:
                snapshot = gpu_snapshot()
                free = eligible_gpus(snapshot, max_used_mib=2048, max_utilization=10)
                blocked = sorted(set(gpu_ids) - set(free))
                disk = disk_check()
                streak = 0 if blocked else streak + 1
                status = dict(state='waiting', blocked_gpu_ids=blocked, gpus=snapshot,
                              stable_polls=streak, disk=disk)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                streak = 0
                status = dict(state='waiting', stable_polls=0,
                              blocker=f'{type(error).__name__}: {error}')
            record(status)
            if streak >= 2:
                # Recheck immutable recovery inputs and the final GPU/disk state.
                # These are observations, not a cluster-wide GPU reservation.
                before_launch()
                try:
                    final = eligible_gpus(gpu_snapshot(), max_used_mib=2048, max_utilization=10)
                    disk_check()
                    ready = set(gpu_ids).issubset(final)
                except (OSError, ValueError, subprocess.SubprocessError):
                    ready = False
                if ready:
                    with (state_dir / 'pipeline.log').open('a') as log:
                        process = subprocess.Popen(command, cwd=REPO, env=environment,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            pass_fds=(lock.fileno(),))
                        status.update(state='running', child_pid=process.pid, command=command)
                        record(status)
                        code = process.wait()
                    status.update(state='finished' if code == 0 else 'failed', exit_code=code)
                    record(status)
                    return status  # Never retry failed training or editor requests.
                streak = 0
            time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--poll-seconds', default=60, type=float)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    environment = launch_environment(os.environ, CACHE)
    os.environ.clear()
    os.environ.update(environment)
    validate_recovery(root)
    setting = json.loads((root / 'setting.json').read_text())
    gpu_ids = json.loads((root / 'launch.json').read_text())['gpu_ids']
    if len(gpu_ids) != setting['gpu_count']:
        raise ValueError('Frozen GPU placement differs from settings')
    from logicbench_phase3_recovery.runtime import checkpoint_reserve
    from skillnet_cohort.runtime import disk_gate
    def disk_check():
        return disk_gate(root, checkpoint_reserve(root, setting),
                         minimum_free_bytes=setting['storage']['minimum_free_bytes'],
                         maximum_run_bytes=setting['storage']['maximum_run_bytes'])
    command = [str(REPO / 'scripts/run_logicbench_phase3_fast.sh'),
               '--setting', str(root / 'setting.json'), '--root', str(root),
               '--gpus', ','.join(map(str, gpu_ids)), '--execute']
    if args.check_only:
        print(json.dumps(dict(recovery_validation='passed', gpu_ids=gpu_ids,
                              disk=disk_check(), gpus=gpu_snapshot(), command=command)))
        return
    state_dir = root / 'recovery/gpu-resume-queue'
    try:
        status = wait_and_launch(state_dir, gpu_ids, command, environment, disk_check,
                                poll_seconds=args.poll_seconds,
                                before_launch=lambda: validate_recovery(root))
    except BlockingIOError:
        raise SystemExit('A resume watcher already holds the queue lock')
    raise SystemExit(status['exit_code'])


if __name__ == '__main__':
    main()
