"""Upload-gated eight-GPU Phase1/2 supervisor with a cohort-wide RMB cap.

No automatic failed-block retry, no report overwrite, no historical experiment
reuse. Native Seen monitoring remains every five updates. Full performance and
O/P/N utility endpoints are opened only after the window prediction is locked.
"""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import REPO, digest, exclusive_writer, file_hash, load_preparation, read_json, require_authorization, write_new_json
from .runtime import authorized_runtime, disk_gate, is_embedding_backend, make_runtime, router_backend


class Pipeline:
    def __init__(self, preparation, root, authorization):
        self.preparation, self.root, self.authorization = map(lambda p: Path(p).resolve(), (preparation, root, authorization))
        load_preparation(self.preparation)
        self.spec = read_json(self.preparation.parent / 'spec.json')
        self.router_backend = router_backend(self.spec)
        candidate = read_json(self.authorization)
        post_training = candidate.get('post_training_recovery')
        readout_recovery = candidate.get('readout_recovery')
        self.permit = require_authorization(self.authorization, self.preparation,
                                            'evaluation' if post_training or readout_recovery else 'training')
        for operation in (('evaluation', 'readout') if readout_recovery else ('evaluation', 'exports')):
            require_authorization(self.authorization, self.preparation, operation)
        p = self.permit
        self.post_training = p.get('post_training_recovery')
        self.readout_recovery = p.get('readout_recovery')
        if sum(bool(p.get(k)) for k in ('post_training_recovery', 'pre_optimizer_recovery', 'readout_recovery')) > 1:
            raise PermissionError('Incompatible recovery boundaries')
        self.recovery = self.readout_recovery or self.post_training or p.get('pre_optimizer_recovery')
        self.audit_root = Path(self.recovery['attempt_dir']) if self.recovery else self.root
        if self.recovery:
            if self.readout_recovery:
                from .readout_recovery import verify_binding
            elif self.post_training:
                from .post_training_recovery import verify_binding
            else:
                from .rollout_recovery import verify_binding
            verify_binding(self.recovery, self.root, self.preparation)
        self.deadline = None
        if self.spec.get('budget_profile'):
            from .day_budget import validate_admission
            validate_admission(p, self.spec, self.preparation)
            self.deadline = None if p['budget_deadline_unix'] is None else float(p['budget_deadline_unix'])
        if p.get('run_root') != str(self.root) or p.get('gpu_ids') != list(range(8)):
            raise PermissionError('Authorization must bind this exact new run and eight assigned GPUs')
        if self.router_backend == 'external_llm':
            if file_hash(p['cost_profile']) != p['cost_profile_sha256']:
                raise PermissionError('Changed money cap')
            os.environ['SKILLNET_ROUTER_COST_PROFILE'] = p['cost_profile']
        receipt = read_json(p['upload_receipt'])
        if (file_hash(p['upload_receipt']) != p['upload_receipt_sha256']
                or receipt.get('repository') != 'Euclider/SkillScope-phase3'
                or receipt.get('all_remote_blob_hashes_match') is not True):
            raise PermissionError('Verified Phase3 upload is required before local RL')
        # Read-only confirmation of the published commit, never a new push.
        remote = subprocess.run(['gh', 'api', 'repos/Euclider/SkillScope-phase3/git/ref/heads/main',
                                 '--jq', '.object.sha'], capture_output=True, text=True, check=True)
        if remote.stdout.strip() != receipt['commit']:
            raise PermissionError('Remote commit changed after verified publication')
        os.environ.update(CUDA_VISIBLE_DEVICES=','.join(map(str, p['gpu_ids'])),
                          TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                          OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        if self.readout_recovery:
            os.environ['SKILLNET_PHASE12_RECOVERY_AUTHORIZATION'] = str(self.authorization)
        else:
            os.environ.pop('SKILLNET_PHASE12_RECOVERY_AUTHORIZATION', None)
        self.active_stage = 'initialization'

    def check_deadline(self):
        if self.deadline is not None and time.time() >= self.deadline:
            raise TimeoutError('Registered hard wallclock limit reached; preserve incomplete evidence')

    def budget(self):
        if is_embedding_backend(self.router_backend):
            _, router = make_runtime(authorized_runtime(self.spec, self.permit['router_cache_path'], self.permit))
            return {**router.stats(), 'paid_router_cost': 0, 'cost_scope': 'external_api_only'}
        from agent_system.memory.router_cost_guard import CostGuard
        return CostGuard(self.permit['cost_profile']).stats()

    def command(self, arguments, label, gpu=None):
        self.commands([(arguments, label, gpu)])

    def commands(self, jobs):
        from .runtime_watch import StageWatch
        children = []
        self.active_stage = ','.join(label for _, label, _ in jobs)
        watch = StageWatch(self.root, self.audit_root, self.active_stage, self.deadline)
        self.check_deadline()
        limits = self.permit['storage']
        disk_gate(self.root, limits['checkpoint_reserve_bytes'], **{
            name: limits[name] for name in ('minimum_free_bytes', 'maximum_run_bytes')})
        try:
            for args, label, gpu in jobs:
                log = self.audit_root / 'logs' / f'{label}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                # Existing failed work is evidence; never restart it implicitly.
                stream = log.open('x')
                env = os.environ.copy()
                if gpu is not None:
                    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                process = subprocess.Popen([sys.executable, '-u', '-B', '-m', *map(str, args)],
                    cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
                children.append((process, stream, label))
            watch.tick(children, event='started', force=True)
            while any(process.poll() is None for process, _, _ in children):
                self.check_deadline()
                if any(process.poll() not in (None, 0) for process, _, _ in children):
                    raise RuntimeError('Stage failed; stop sibling jobs, preserve all records')
                watch.tick(children)
                time.sleep(1)
            if any(process.returncode != 0 for process, _, _ in children):
                raise RuntimeError('Stage stopped; inspect its retained log')
        finally:
            for process, stream, _ in children:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)  # Only a child session created above.
            for process, stream, _ in children:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                stream.close()
            watch.tick(children, event='finished' if all(p.returncode == 0 for p, _, _ in children) else 'stopped', force=True)

    def parallel(self, module, window, update, start=None):
        base = [module, '--root', window, '--update', update, '--shards', 8]
        if start is not None:
            base += ['--start-update', start]
        self.commands([(base + ['--shard', i], f'{window.name}-{module}-u{update}-shard{i}', gpu)
                       for i, gpu in enumerate(self.permit['gpu_ids'])])

    def evaluate(self, update, split, purpose, prediction=None):
        output = self.root / 'evaluations' / f'u{update:04d}-{split}-{purpose}'
        if not (output / 'completion.json').exists():
            args = ['skillnet_cohort.evaluate', '--preparation', self.preparation,
                    '--checkpoint', self.root / 'models' / f'u{update:04d}', '--update', update,
                    '--split', split, '--purpose', purpose, '--output', output,
                    '--authorization', self.authorization, '--execute']
            if prediction is not None:
                args += ['--prediction', prediction]
            shards = self.spec['evaluation'].get('parallel_workers', 1)
            if shards == 1:
                self.command(args, output.name, gpu=0)
            else:
                if shards != len(self.permit['gpu_ids']):
                    raise ValueError('Full evaluation must use exactly the registered GPU shards')
                self.commands([(args + ['--shards', shards, '--shard', i], f'{output.name}-shard{i}', gpu)
                               for i, gpu in enumerate(self.permit['gpu_ids'])])
                self.command(args + ['--shards', shards, '--collect-shards'], f'{output.name}-collect')
        return output

    def run(self):
        from .support import build_support, register_window
        from .window_storage import seal_window, reclaim_proven_rows
        from phase2.protocol import signal_directory
        from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
        if self.router_backend == 'external_llm' and not os.environ.get('SKILLNET_ROUTER_API_KEY'):
            raise PermissionError('Missing dedicated runtime credential')
        if (self.root / 'launch.json').exists() and not self.recovery:
            raise FileExistsError('This launch is new-run only; reconcile/resume explicitly after a stop')
        self.check_deadline()
        horizon = int(self.spec['training']['iterations'])
        readout_recovery = getattr(self, 'readout_recovery', None)
        if (self.post_training or readout_recovery) and horizon != 5:
            raise PermissionError('Post-training continuation is bound to the completed U0-U5 window')
        if (self.audit_root / 'launch.json').exists():
            raise FileExistsError('Recovery was already attempted; no automatic failed retry')
        write_new_json(self.audit_root / 'launch.json', {'preparation_sha256': file_hash(self.preparation),
            'authorization_sha256': file_hash(self.authorization), 'gpu_ids': self.permit['gpu_ids'],
            'global_horizon': horizon, 'window_horizon': 5, 'seed': self.spec['seed'],
            'router_backend': self.router_backend,
            'environment_block_seed': f"{self.spec['seed']}+16*(start_update//5)",
            'budget': self.budget(), 'full_run_guaranteed_with_budget': False})
        if not self.recovery:
            self.command(['skillnet_cohort.checkpoints', '--preparation', self.preparation,
                '--target', self.root / 'models/u0000', '--authorization', self.authorization, '--execute'], 'export-u0000')
        previous_prediction = None
        for start in range(0, horizon, 5):
            end = start + 5
            if not self.post_training and not readout_recovery:
                self.command(['skillnet_cohort.segmented_training', '--preparation', self.preparation,
                    '--root', self.root, '--authorization', self.authorization, '--start', start, '--execute'],
                    f'train-u{start:04d}-u{end:04d}')
            native = self.root / 'checkpoints' / f'global_step_{end}'
            if validate_full_checkpoint(native)['world_size'] != 8 or not (self.root / 'metrics' / f'u{end:04d}.json').is_file():
                raise ValueError('Missing complete native training boundary')
            if not readout_recovery:
                self.command(['skillnet_cohort.checkpoints', '--preparation', self.preparation,
                    '--target', self.root / 'models' / f'u{end:04d}', '--native-checkpoint', native,
                    '--authorization', self.authorization, '--execute'], f'export-u{end:04d}')
            windows, predictions = [], []
            if start == 0 and not readout_recovery:
                for split in self.spec['evaluation']['splits']:
                    self.evaluate(start, split, 'performance')
            # Both split predictions must be locked before either target gold is opened.
            for split in self.spec['evaluation'].get('utility_splits', self.spec['evaluation']['splits']):
                window = self.root / 'windows' / f'u{start:04d}-u{end:04d}-{split}'
                if readout_recovery:
                    if str(window) not in readout_recovery['windows']:
                        raise PermissionError('Window is outside the audited readout continuation')
                else:
                    source = self.evaluate(start, split, 'anchors', previous_prediction)
                    support_dir = self.root / 'support' / f'u{start:04d}-{split}'
                    support = build_support(self.preparation, source, support_dir)
                    if not support['anchor_sets']:
                        write_new_json(support_dir / 'abstention.json', {'reason': 'insufficient_natural_support', 'all_candidates': 37})
                        continue
                    register_window(self.preparation, support_dir / 'manifest.json', self.root, window, start, self.authorization)
                    self.parallel('phase2.evaluate', window, start)
                    self.parallel('phase2.measure', window, end, start)
                self.command(['phase2.aggregate', '--root', window, '--update', end, '--start-update', start, '--shards', 8],
                             f'{window.name}-aggregate')
                self.command(['phase2.window_forecast', '--root', window, '--update', end, '--start-update', start],
                             f'{window.name}-prediction')
                predictions.append(signal_directory(window, end, start) / 'prediction.json')
                windows.append(window)
            if not predictions:
                raise ValueError('No naturally supported utility window; stop with explicit abstention, never force skill calls')
            for window in windows:
                self.parallel('phase2.evaluate', window, end)
                self.command(['phase2.window_report', '--root', window], f'{window.name}-report')
                seal_window(window, start, end)
            previous_prediction = predictions[0]
            for split in self.spec['evaluation']['splits']:
                self.evaluate(end, split, 'performance', previous_prediction)
            cleanup = reclaim_proven_rows(self.root, windows, start, end, self.permit)
            write_new_json(self.root / 'completed_windows' / f'u{start:04d}-u{end:04d}.json',
                {'start': start, 'end': end, 'windows': [str(w) for w in windows], 'budget': self.budget(),
                 'deleted_temporary_rows': len(cleanup['deleted']), 'unproven_rows_retained': len(cleanup['retained'])})
        self.check_deadline()
        if self.spec['readout'].get('capture_scope') == 'window_start_old_only_v1':
            self.active_stage = 'final_reports'
            from .reports import seed_reports
            seed_reports(self.root, self.spec)
        write_new_json(self.root / 'complete.json', {'endpoint': horizon, 'budget': self.budget(), 'status': 'complete'})


def main():
    def stop_owned_pipeline(signum, frame):
        raise TimeoutError('Supervisor termination requested; preserve evidence and stop owned children')
    signal.signal(signal.SIGTERM, stop_owned_pipeline)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preparation', 'root', 'authorization'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--prompt-key', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        raise PermissionError('Use --execute only for the authorized new run')
    pipeline = Pipeline(args.preparation, args.root, args.authorization)
    if args.prompt_key:
        if pipeline.router_backend != 'external_llm':
            raise ValueError('Embedding router never requests API credentials')
        os.environ['SKILLNET_ROUTER_API_KEY'] = getpass.getpass('Router credential (hidden, process environment only): ')
    if args.detach:
        args.root.mkdir(parents=True, exist_ok=True)
        with (args.root / 'supervisor.log').open('x') as stream:
            child = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'skillnet_cohort.run',
                '--preparation', str(args.preparation), '--root', str(args.root),
                '--authorization', str(args.authorization), '--execute'], cwd=REPO,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_new_json(args.root / 'supervisor_launch.json', {'pid': child.pid, 'key_persisted': False})
        print({'pid': child.pid, 'root': str(args.root), 'status': 'launched_not_complete'})
        return
    with exclusive_writer(args.root):
        try:
            pipeline.run()
        except BaseException as error:
            # Do not serialize arbitrary exceptions: remote providers may echo inputs.
            write_new_json(pipeline.audit_root / 'stopped.json', {'status': 'stopped_incomplete', 'stage': pipeline.active_stage,
                'exception_type': type(error).__name__, 'budget': pipeline.budget(),
                'automatic_retry': False, 'all_existing_evidence_retained': True})
            raise SystemExit('Stopped safely; see stopped.json and stage logs. No automatic retry.') from None


if __name__ == '__main__':
    main()
