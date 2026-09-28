"""Passive 30-second stage telemetry; never retries or kills on a stall warning."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

from phase1.archive import atomic_write_json


def gpu_snapshot():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
        return [dict(zip(('index', 'utilization_percent', 'used_mib', 'total_mib'), map(int, row.split(','))))
                for row in result.stdout.splitlines() if row.strip()]
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {'unavailable': type(error).__name__}


class StageWatch:
    def __init__(self, root, audit_root, stage, deadline):
        self.root, self.audit_root = Path(root), Path(audit_root)
        self.stage, self.deadline = stage, deadline
        self.started = self.last_progress = time.time()
        self.last_signature = None
        self.next_tick = 0

    def snapshot(self, children, now, event):
        forward, optimizer, files = {}, {}, []
        for path in sorted((self.audit_root/'forward_progress').glob('rank-*.json')):
            try:
                forward[path.stem] = json.loads(path.read_text())
                files.append((str(path), path.stat().st_mtime_ns, path.stat().st_size))
            except (FileNotFoundError, json.JSONDecodeError):
                continue  # An in-flight atomic diagnostic is not training corruption.
        for path in sorted((self.root/'optimizer_steps').glob('*.jsonl')):
            try:
                lines = path.read_text().splitlines()
                # Ignore only an unfinished last JSONL line; next tick reads it.
                rows = [json.loads(line) for line in lines if line.endswith('}')]
                optimizer[path.stem] = {'completed_steps': len(rows), 'last': rows[-1] if rows else None}
                files.append((str(path), path.stat().st_mtime_ns, path.stat().st_size))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        for folder, pattern in ((self.root/'rollout_progress', 'u*/step-*.json'),
                                (self.audit_root/'logs', '*.log'),
                                (self.root/'metrics', 'u*.json')):
            for path in folder.glob(pattern):
                try:
                    stat = path.stat()
                    files.append((str(path), stat.st_mtime_ns, stat.st_size))
                except FileNotFoundError:
                    continue
        evaluation = {}
        # Count completed artifact files, not rewards/outcomes. Episode work can
        # advance without the evaluator printing to stdout on every step.
        for folder, pattern in ((self.root/'evaluations', '*/shards/*/results/*.json'),
                                (self.root/'windows', '*/evaluations/u*/trajectories/*/*.json')):
            for path in folder.glob(pattern):
                try:
                    stat = path.stat()
                    files.append((str(path), stat.st_mtime_ns, stat.st_size))
                    key = str(path.parent.parent.relative_to(self.root))
                    evaluation[key] = evaluation.get(key, 0) + 1
                except FileNotFoundError:
                    continue
        signature = tuple(sorted(files))
        if signature != self.last_signature:
            self.last_progress, self.last_signature = now, signature
        elapsed_without_artifact = now-self.last_progress
        return {'updated_at': datetime.fromtimestamp(now, timezone.utc).isoformat(),
            'stage': self.stage, 'event': event, 'stage_elapsed_seconds': now-self.started,
            'remaining_budget_seconds': None if self.deadline is None else max(0., self.deadline-now),
            'children': [{'pid': p.pid, 'label': label, 'exit_code': p.poll()} for p, _, label in children],
            'forward': forward, 'optimizer': optimizer, 'evaluation_completed_files': evaluation,
            'completed_training_iterations': sorted(p.stem for p in (self.root/'metrics').glob('u*.json')),
            'gpus': gpu_snapshot(), 'free_bytes': shutil.disk_usage(self.root).free,
            'seconds_without_new_artifact': elapsed_without_artifact,
            'advisory_stall': elapsed_without_artifact >= 300,
            'stall_action': 'warn_only_no_retry_no_kill', 'automatic_retry': False}

    def tick(self, children, *, event='running', force=False):
        now = time.time()
        if not force and now < self.next_tick:
            return
        self.next_tick = now+30
        try:
            value = self.snapshot(children, now, event)
            atomic_write_json(self.audit_root/'runtime-status.json', value)
            with (self.audit_root/'runtime-history.jsonl').open('a') as stream:
                stream.write(json.dumps(value, sort_keys=True)+'\n')
            progress = {rank: f"{v['role']} {v['completed_microbatches']}/{v['total_microbatches']}"
                        for rank, v in value['forward'].items()}
            steps = {rank: v['completed_steps'] for rank, v in value['optimizer'].items()}
            print(f"WATCH {value['updated_at']} {event} {self.stage}: forward={progress}; optimizer={steps}; "
                  f"iterations={value['completed_training_iterations']}; "
                  f"evaluation={value['evaluation_completed_files']}; stall_warning={value['advisory_stall']}", flush=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            # Telemetry failure must not discard a successful expensive training stage.
            print(f'WATCH diagnostic unavailable: {type(error).__name__}: {error}', flush=True)
