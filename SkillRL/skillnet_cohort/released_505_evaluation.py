"""Restart missing505 utility after corrected404 report and corrected505 readout.

Uses the unchanged FileStore vLLM evaluator and explicit five-query local cache
adapter. Already completed trajectories and successful cache decisions are
validated and reused. Neither source reports nor frozen numerical code is edited.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

from .common import exclusive_writer, file_hash, read_json, write_new_json
from .explicit_router_resume import audit_cache, install, load_authorization
from .first_calls_recovery import RecoveryAssessment, audit_endpoint, validate_readout, verify_retained
from .first_calls_port_recovery import build_file_engine, RECORD_ENV
from . import numerical_released_run as runner
from .numerical_readout import SOURCES


def admitted(path):
    plan = runner.binding(path); runner.released_guard(plan)
    runner.verify_restart_requirements(plan)
    root = Path(plan['root'])
    intent = read_json(root/'seed505-restart-intent.json')
    if (intent['restart_mode'] != 'fresh_processes_reuse_disk_results'
            or intent['seed404_complete_sha256'] != file_hash(root/'seed-404/complete.json')
            or intent['seed505_commit_sha256'] != file_hash(root/'seed-505/committed.json')
            or intent['router_authorization_sha256'] != plan['router_resume']['sha256']):
        raise PermissionError('505 restart intent must bind completed404 and corrected505 scores')
    return plan


def evaluate(path, update, shard):
    plan = admitted(path); root = SOURCES[505]
    if update not in (0, 5) or shard not in range(8) or os.environ.get('CUDA_VISIBLE_DEVICES') != str(shard):
        raise PermissionError('Wrong505 endpoint or single-GPU shard')
    audit = Path(plan['root'])/'seed505-utility/rendezvous'
    record = audit/f'u{update:04d}-shard{shard}.json'
    ready = audit/f'u{update:04d}-shard{shard}-ready.json'
    if record.exists() or ready.exists():
        raise FileExistsError('Never retry an attempted vLLM engine implicitly')
    audit.mkdir(parents=True, exist_ok=True); os.environ[RECORD_ENV] = str(record)
    install(plan['router_resume']['path'], root, update)
    from . import vllm_backend
    original_policy = vllm_backend.VLLMPolicy
    class FilePolicy(original_policy):
        def __init__(self, checkpoint, profile):
            self.engine = build_file_engine(checkpoint, profile)
            self.tokenizer = self.engine.get_tokenizer()
            write_new_json(ready, {'state': 'ENGINE_READY', 'pid': os.getpid(),
                'plan_sha256': file_hash(path), 'rendezvous_sha256': file_hash(record),
                'update': update, 'shard': shard, 'sampling_implementation_unchanged': True})
    previous_argv = sys.argv; vllm_backend.VLLMPolicy = FilePolicy
    try:
        sys.argv = ['phase2.evaluate', '--root', str(root), '--update', str(update),
                    '--shards', '8', '--shard', str(shard)]
        from phase2.evaluate import main as evaluate_main
        evaluate_main()
    finally:
        vllm_backend.VLLMPolicy = original_policy; sys.argv = previous_argv


class ReleasedAssessment(RecoveryAssessment):
    def commands(self, jobs):
        path = Path(self.recovery['root'])/'plan.json'; admitted(path)
        expected = [(['phase2.evaluate', '--root', self.root, '--update', update,
            '--shards', 8, '--shard', rank], f'utility-u{update:04d}-shard{rank}', rank)
            for update in (0, 5) for rank in range(8)]
        if not jobs or any(job not in expected for job in jobs) or len({j[1] for j in jobs}) != len(jobs):
            raise PermissionError('Only missing505 utility shards may execute')
        authorization = load_authorization(self.recovery['router_resume']['path'])
        before = audit_cache(self.root/'router.sqlite3', authorization)
        self.active_stage = ','.join(label for _, label, _ in jobs)
        write_new_json(self.audit/'cache-admission'/(self.active_stage+'.json'), before)
        converted = []
        for args, _, rank in jobs:
            converted.extend(runner.utility_jobs(path, args[4], [rank]))
        runner.commands(path, converted)
        write_new_json(self.audit/'cache-after'/(self.active_stage+'.json'),
                       audit_cache(self.root/'router.sqlite3', authorization))


def run(path):
    from .first_calls_run import lock_prediction
    from .first_calls_reuse import import_endpoint
    from .first_calls_report import report
    from .window_storage import seal_window
    plan = admitted(path); root = SOURCES[505]; audit = Path(plan['root'])/'seed505-utility'
    if (audit/'launch.json').exists() or runner.gpu_users():
        raise PermissionError('Only one explicit505 restart on released GPUs is allowed')
    if file_hash(root/'router.sqlite3') != plan['router_at_preparation_sha256']:
        raise ValueError('505 router changed before the admitted restart')
    authorization = load_authorization(plan['router_resume']['path'])
    audit_cache(root/'router.sqlite3', authorization)
    verify_retained(plan['retained505']); validate_readout(root)
    endpoint = audit_endpoint(root, 0)
    if endpoint['completed'] != plan['endpoint505']['completed']:
        raise ValueError('Retained505 boundary changed before restart')
    if (root/'evaluations/u0005').exists() or (root/'window_signals/u0000-u0005/prediction.json').exists():
        raise ValueError('Restart requires the recorded partial-U0 boundary')
    active = ReleasedAssessment(plan['assessment_plan']['path'], audit, plan)
    active.disk(active.plan['projected_recording_upper_bytes'])
    write_new_json(audit/'launch.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'plan_sha256': file_hash(path), 'retained_u0_records': endpoint['completed'],
        'automatic_retry': False, 'training_reexecuted': False})
    try:
        with exclusive_writer(root), exclusive_writer(audit):
            active.evaluate(0)
            # Retain the legacy snapshot for the required original comparator;
            # corrected505 scores are independently committed before this stage.
            lock_prediction(root)
            import_endpoint(root, 5)
            active.evaluate(5)
            active.active_stage = 'legacy_comparator_analysis_no_report_overwrite'
            report(root); seal_window(root, 0, 5)
            verify_retained(plan['retained505'])
            cache = audit_cache(root/'router.sqlite3', authorization)
            write_new_json(audit/'cache-final.json', cache)
            write_new_json(root/'complete.json', {'status': 'complete', 'training_reexecuted': False,
                'reports_published': False, 'original_report_views_preserved': True,
                'old_evidence_preserved': True, 'explicit_recovery': str(audit)})
        write_new_json(audit/'complete.json', {'status': 'complete', 'seed': 505,
            'retained_u0_records': 810, 'completed_endpoint_records': [3636, 3636],
            'finished_unix': time.time(), 'original_reports_overwritten': False,
            'corrected_report_pending': True, 'new_RL': False})
    except BaseException as error:
        write_new_json(audit/'stopped.json', {'stage': active.active_stage,
            'error': f'{type(error).__name__}: {error}', 'automatic_retry': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--update', type=int, choices=(0, 5), required=True)
    parser.add_argument('--shard', type=int, choices=range(8), required=True)
    args = parser.parse_args()
    if not args.evaluate:
        parser.error('Only an explicitly admitted evaluation worker is supported')
    evaluate(args.plan, args.update, args.shard)


if __name__ == '__main__':
    main()
