"""Race-safe capacity accounting for explicitly versioned assessment recovery.

Do not change the source-frozen legacy queue's runtime module in place. This
adapter retains its limits and link handling, adding only SQLite sidecars that
can legitimately disappear at transaction boundaries. It is not an integrity
checker: permanent evidence disappearing still raises.
"""
import os
from pathlib import Path
import re
import shutil
import stat

from .common import read_json


def transient_file(path):
    name = path.name
    if ((path.parent.name == 'forward_progress'
         and re.fullmatch(r'\.rank-\d+\.json\.[a-zA-Z0-9_\-]+', name))
            or re.fullmatch(r'\.runtime-status\.json\.[a-zA-Z0-9_\-]+', name)
            or re.fullmatch(r'\.publish-[a-zA-Z0-9_\-]+', name)):
        return True
    # phase1.archive atomically publishes continuation JSON and shard receipts.
    # Only accept these exact shapes in their registered directory layout, and
    # require the permanent replacement to exist. Do not exempt arbitrary JSON.
    atomic = re.fullmatch(r'\.([0-9a-f]{24}\.json)\.[a-zA-Z0-9_\-]+', name)
    shard = re.fullmatch(r'\.(shard-[0-7]-complete\.json)\.[a-zA-Z0-9_\-]+', name)
    if ((atomic and path.parent.parent.name == 'trajectories')
            or (shard and re.fullmatch(r'u\d{4}', path.parent.name)
                and path.parent.parent.name == 'evaluations')):
        target = path.parent/(atomic or shard).group(1)
        return stat.S_ISREG(target.lstat().st_mode)
    if name not in ('router.sqlite3-journal', 'router.sqlite3-wal', 'router.sqlite3-shm'):
        return False
    # Require the owned, permanent database to remain present (not a symlink).
    return stat.S_ISREG((path.parent/'router.sqlite3').lstat().st_mode)


def storage_bytes(root):
    used = 0
    for parent, _, names in os.walk(root, followlinks=False):
        for name in names:
            path = Path(parent)/name
            try:
                entry = path.lstat()
            except FileNotFoundError:
                if transient_file(path):
                    continue
                raise
            if not stat.S_ISLNK(entry.st_mode):
                used += entry.st_size
    return used


def disk_gate(root, required_bytes=0, *, minimum_free_bytes, maximum_run_bytes):
    root = Path(root).resolve()
    resource_path = root/'resource_limits.json'
    if resource_path.is_file():
        cohort = read_json(resource_path).get('cohort_storage_root')
        if cohort is not None:
            cohort = Path(cohort).resolve()
            if root == cohort or not root.is_relative_to(cohort) or not (cohort/'queue_launch.json').is_file():
                raise ValueError('Aggregate storage accounting requires this run\'s launched parent queue')
            root = cohort
    existing = root if root.exists() else root.parent
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    used = storage_bytes(root)
    if free-required_bytes < minimum_free_bytes or used+required_bytes > maximum_run_bytes:
        raise OSError('Storage budget would be exceeded; no automatic deletion or budget waiver')
    return {'used_bytes': used, 'free_bytes': free, 'reserved_next_bytes': required_bytes}
