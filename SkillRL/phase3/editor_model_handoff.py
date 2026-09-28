"""One-shot U5 handoff from the frozen, unused o3 editor to gpt-5.5.

The orchestration parent must already be SIGSTOPed while its training child is
left running. This process never signals the training child or Ray workers.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from phase1.watch_qwen35_checkpoints import validate_full_checkpoint

from .api import APIConfig
from .common import canonical, digest, require, strict_json, write_new


def process_stat(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    fields = raw[raw.rfind(')') + 2:].split()
    return {'state': fields[0], 'ppid': int(fields[1]),
            'start_time': int(fields[19]), 'exit_code': int(fields[49])}


def process_environment(pid):
    entries = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
    env = dict(item.decode().split('=', 1) for item in entries if b'=' in item)
    require(bool(env.get('SKILLRL_PHASE3_EDITOR_API_KEY')), 'Original launcher editor credential unavailable')
    return env


def migrate_unused_ledger(path):
    require(path.is_file() and not path.is_symlink(), 'Expected the original editor ledger')
    with sqlite3.connect(str(path), timeout=60, isolation_level=None) as db:
        db.execute('BEGIN IMMEDIATE')
        require(db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 0,
                'Cannot migrate an editor ledger after an API reservation')
        old_row = db.execute('SELECT value FROM profile WHERE id=1').fetchone()
        require(old_row is not None, 'Original API profile missing')
        old = strict_json(old_row[0])
        require(old['stage'] == 'editor' and old['model'] == 'o3', 'Unexpected original editor profile')
        new = {**old, 'model': 'gpt-5.5'}
        APIConfig(**new)
        stamped = datetime.now(timezone.utc).isoformat()
        db.execute('CREATE TABLE IF NOT EXISTS profile_migrations '
                   '(id INTEGER PRIMARY KEY, old_value TEXT NOT NULL, new_value TEXT NOT NULL, '
                   'checkpoint TEXT NOT NULL, migrated_utc TEXT NOT NULL)')
        require(db.execute('SELECT COUNT(*) FROM profile_migrations').fetchone()[0] == 0,
                'Editor profile was previously migrated')
        db.execute('INSERT INTO profile_migrations (old_value,new_value,checkpoint,migrated_utc) '
                   'VALUES (?,?,?,?)', (old_row[0], canonical(new), 'U5', stamped))
        db.execute('UPDATE profile SET value=? WHERE id=1', (canonical(new),))
        db.commit()
    return {'from_model': 'o3', 'to_model': 'gpt-5.5', 'migrated_utc': stamped,
            'old_profile_sha256': digest(old), 'new_profile_sha256': digest(new)}


def handoff(experiment_root, launcher_pid, parent_pid, training_pid, interval):
    root = Path(experiment_root).resolve()
    require(root.name == 'skill-scope-phase3-batched-gpu-v6-20260926', 'Unexpected experiment root')
    run = root / 'runs' / 'readout_d'
    launcher = root / 'run_six_arms.sh'
    preparation = root / 'assets-launch-v1' / 'manifest.json'
    require(launcher.is_file() and preparation.is_file(), 'Original launch artifacts missing')
    parent_stat, child_stat = process_stat(parent_pid), process_stat(training_pid)
    require(parent_stat is not None and parent_stat['state'] == 'T', 'Orchestrator is not paused')
    require(child_stat is not None and child_stat['ppid'] == parent_pid and child_stat['state'] != 'Z',
            'Training child is not running under the paused orchestrator')
    require(process_stat(launcher_pid) is not None, 'Original launcher missing')
    parent_start, child_start = parent_stat['start_time'], child_stat['start_time']
    env = process_environment(launcher_pid)
    print('HANDOFF_ARMED paused_parent=1 training_child_running=1 editor_attempts_expected=0', flush=True)
    last_notice = time.monotonic()
    while True:
        current = process_stat(training_pid)
        require(current is not None and current['start_time'] == child_start,
                'Training child disappeared before exit status could be verified')
        require(process_stat(parent_pid) is not None
                and process_stat(parent_pid)['start_time'] == parent_start
                and process_stat(parent_pid)['state'] == 'T', 'Paused parent state changed unexpectedly')
        if current['state'] == 'Z':
            require(current['exit_code'] == 0, 'Training child exited unsuccessfully')
            break
        if time.monotonic() - last_notice >= 600:
            print('WAITING_FOR_U5 training_child_untouched=1', flush=True)
            last_notice = time.monotonic()
        time.sleep(interval)
    checkpoint = run / 'checkpoints' / 'global_step_5'
    checkpoint_info = validate_full_checkpoint(checkpoint)
    require(checkpoint_info['world_size'] == 8, 'U5 checkpoint is incomplete')
    metrics = run / 'metrics' / 'u0005.json'
    require(metrics.is_file(), 'U5 metrics missing after successful training exit')
    print('U5_SEALED eight_shards=1 training_exit=0', flush=True)
    os.kill(parent_pid, signal.SIGTERM)
    os.kill(parent_pid, signal.SIGCONT)
    for _ in range(60):
        current = process_stat(parent_pid)
        if current is None or current['state'] == 'Z':
            break
        time.sleep(1)
    else:
        raise RuntimeError('Old orchestrator did not exit after U5')
    ledger = migrate_unused_ledger(run / 'editor.sqlite3')
    manifest = strict_json(preparation.read_text())
    receipt = {'schema_version': 'skillrl.phase3.editor_model_switch.v1',
               'preparation_sha256': digest(manifest), 'first_affected_branch': 'readout_d',
               'first_affected_update': 5, 'prior_editor_attempts': 0,
               'training_child_exit_code': 0, 'checkpoint_world_size': checkpoint_info['world_size'],
               'u5_metrics_sha256': digest(strict_json(metrics.read_text())), **ledger}
    write_new(root / 'editor_model_switch.json', receipt)
    log = root / 'editor-model-handoff-queue.log'
    with log.open('ab') as stream:
        proc = subprocess.Popen(['bash', str(launcher)], cwd='/mnt/workspace/users/wangyifan/skill-RL/SkillRL',
                                env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
    print(f'RELAUNCHED_QUEUE pid={proc.pid} model=gpt-5.5 key_unchanged=1', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--launcher-pid', type=int, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--interval-seconds', type=int, default=15)
    args = parser.parse_args()
    require(1 <= args.interval_seconds <= 60, 'Invalid polling interval')
    handoff(args.experiment_root, args.launcher_pid, args.parent_pid,
            args.training_pid, args.interval_seconds)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Never print credentials, provider errors, or arbitrary exception messages.
        print(f'HANDOFF_STOPPED type={type(error).__name__}', file=sys.stderr, flush=True)
        raise SystemExit(1) from None
