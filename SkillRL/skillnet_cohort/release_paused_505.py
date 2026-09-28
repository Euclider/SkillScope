"""Explicitly discard only the identity-bound paused505 in-memory process tree.

Disk results and all router ledger bytes are preserved. SIGKILL is intentional:
these processes are SIGSTOPped, so SIGTERM cannot execute without first resuming
evaluation. No process is resumed and no GPU query or environment is run here.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import time

from .common import file_hash, read_json, write_new_bytes, write_new_json
from .numerical_readout import COHORT, SOURCES, binding, paused_tree
from .first_calls_defer import gpu_users, process_identity

OUTPUT = COHORT/'numerical-readout-release-v1'
PREVIOUS = COHORT/'numerical-readout-v1/plan.json'
AUTHORITY = ('2026-09-21 用户明确要求释放暂停505的进程/显存，不再保留内存现场；'
             '404修正版重算完成后重新启动505评估，保留磁盘证据、不重跑RL')


def released(receipt):
    """Verify the original identities cannot execute; never signal recycled PIDs."""
    for expected in receipt['processes']:
        current = process_identity(expected['pid'])
        if (current is not None and current['start_ticks'] == expected['start_ticks']
                and current['state'] != 'Z'):
            raise ProcessLookupError('An explicitly released505 process is still alive')
    if set(gpu_users()) & {p['pid'] for p in receipt['processes']}:
        raise RuntimeError('Released505 still owns GPU memory')


def release():
    if (OUTPUT/'release-intent.json').exists() or (OUTPUT/'release-505.json').exists():
        raise FileExistsError('Never repeat a release attempt implicitly')
    plan = binding(PREVIOUS)
    stopped = read_json(PREVIOUS.parent/'stopped.json')
    if stopped['seed505_resume_intent_exists'] or stopped['seed505_resume_already_sent']:
        raise PermissionError('This release requires505 to have remained paused')
    tree = paused_tree(plan)
    if len(tree) != 17:
        raise PermissionError('Only the registered17 paused505 processes are authorized')
    prior = process_identity(read_json(PREVIOUS.parent/'launch.json')['pid'])
    if prior is not None and prior['state'] != 'Z':
        raise ProcessLookupError('The failed numerical supervisor must have exited')
    supervisor = read_json(plan['pause']['path'])['supervisor_pid']
    engines = {p['pid'] for p in tree if p['parent'] != supervisor and p['pid'] != supervisor}
    if set(gpu_users()) != engines:
        raise PermissionError('Unexpected GPU consumers; do not signal any other task')
    snapshots = []
    db = SOURCES[505]/'router.sqlite3'
    for source in (db, *(db.with_name(db.name+s) for s in ('-journal', '-wal', '-shm'))):
        if source.exists():
            if source.is_symlink():
                raise ValueError('Router ledger must be a regular local file')
            before = file_hash(source)
            snapshot = OUTPUT/'release-cache-snapshot'/source.name
            write_new_bytes(snapshot, source.read_bytes())
            if file_hash(source) != before or file_hash(snapshot) != before:
                raise ValueError('Paused router ledger changed during snapshot')
            snapshots.append({'path': str(source), 'snapshot': str(snapshot), 'sha256': before})
    intent = {'authority': AUTHORITY, 'approved': True, 'created_unix': time.time(),
        'previous_plan': {'path': str(PREVIOUS), 'sha256': file_hash(PREVIOUS)},
        'processes': tree, 'signal': 'SIGKILL', 'no_SIGCONT': True,
        'preserved_cache_files': snapshots, 'automatic_retry': False,
        'index_snapshot': plan['pause_verification'], 'old_disk_artifacts_deleted': False}
    # Bind all identities to kernel handles before the first signal.
    handles = []
    try:
        for expected in sorted(tree, key=lambda p: p['pid'] != supervisor):
            handles.append(os.pidfd_open(expected['pid']))
        paused_tree(plan)
        write_new_json(OUTPUT/'release-intent.json', intent)
        for handle in handles:
            signal.pidfd_send_signal(handle, signal.SIGKILL)
    finally:
        for handle in handles:
            os.close(handle)
    deadline = time.monotonic()+30
    while True:
        try:
            released(intent)
            break
        except (ProcessLookupError, RuntimeError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    for row in snapshots:
        if file_hash(row['path']) != row['sha256']:
            raise ValueError('Unexpected router ledger mutation during process release')
    progress = read_json(plan['pause_verification']['path'])
    for row in progress['endpoints']['u0000']['shards']:
        if file_hash(row['path']) != row['sha256']:
            raise ValueError('505 index changed during release')
    receipt = {**intent, 'status': 'RELEASED_BY_USER', 'released_unix': time.time(),
        'gpu_pids_after': gpu_users(), 'retained_u0_records': 810,
        'retained_u0_expected': 3636, 'process_state_recoverable': False,
        'disk_results_preserved': True}
    write_new_json(OUTPUT/'release-505.json', receipt)
    print({'state': receipt['status'], 'process_count': len(tree),
           'gpu_pids_after': receipt['gpu_pids_after'], 'retained_u0_records': 810}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute is required for the user-approved release')
    release()


if __name__ == '__main__':
    main()
