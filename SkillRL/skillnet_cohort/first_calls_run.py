"""No-RL assessment extension: all observed skills, all first-call anchors."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import REPO, exclusive_writer, file_hash, read_json, require_authorization, write_new_json


def verify_legacy_seal(plan):
    binding = plan['legacy_seal']; path = Path(binding['path']); root = path.parent
    if file_hash(path) != binding['sha256']:
        raise ValueError('Original sealed evidence changed')
    for item in read_json(path)['files']:
        source = root/item['path']
        if not source.resolve().is_relative_to(root.resolve()) or file_hash(source) != item['sha256']:
            raise ValueError('Original sealed artifact changed: '+item['path'])


def lock_prediction(root):
    """Direct scores only; explicitly NOT a new prospective seed404 study."""
    import pandas as pd
    from phase1.archive import utc_now
    from phase2.protocol import validate_extended, signal_directory
    from phase2.ranking import score_snapshot
    from phase2.utilities import margins, read_evaluations
    from phase2.window_forecast import features, KEYS
    root = Path(root); config = read_json(root/'protocol.json'); validate_extended(config, REPO)
    if (root/'evaluations/u0005').exists():
        raise ValueError('Lock amendment scores before opening this variant target results')
    directory = signal_directory(root, 5, 0); commit = read_json(directory/'committed.json')
    if commit['features_sha256'] != file_hash(directory/'skill_context_features.parquet'):
        raise ValueError('Changed score features')
    current = features(root, config, {'start': 0, 'end': 5}, margins(read_evaluations(root)))
    rows = [{**{k: r[k] for k in KEYS}, 'supported': bool(r.supported),
        'P_int': None if pd.isna(r.P_int) else float(r.P_int),
        'D': None if pd.isna(r.D_contribution) else float(r.D_contribution), 'predicted_delta_zero': 0.}
        for _, r in current.iterrows()]
    value = {'created_at': utc_now(), 'start_update': 0, 'global_update': 5, 'window_role': 'test',
        'rl_path_id': config['rl_path_id'], 'target_gold_read': False,
        'target_gold_read_scope': 'this scoring computation; earlier legacy seed404 labels exist',
        'coverage_amendment': config['coverage_amendment'],
        'protocol_sha256': file_hash(root/'protocol.json'), 'features_sha256': commit['features_sha256'],
        'predictions': rows, 'models': {}, 'ranking_plan': config['ranking'],
        'ranking_scores': score_snapshot(current, config['ranking']),
        'ranking_primary': True, 'fitted_regressor_required': False,
        'status': 'fixed_direct_scores_retrospective_coverage_amendment'}
    write_new_json(directory/'prediction.json', value)
    return directory/'prediction.json'


class Assessment:
    def __init__(self, path):
        self.path = Path(path).resolve(); self.plan = read_json(path)
        self.root = Path(self.plan['jobs'][0]['run_root']); self.active_stage = 'admission'
        self.permit = require_authorization(self.path.parent/'permit.json',
            self.plan['jobs'][0]['preparation'], 'evaluation')
        require_authorization(self.path.parent/'permit.json', self.plan['jobs'][0]['preparation'], 'readout')
        if self.permit['run_root'] != str(self.root) or set(self.permit['operations']) != {'evaluation', 'readout'}:
            raise PermissionError('This assessment cannot train, export, edit skills or resample performance')

    def disk(self, additional=0):
        from .runtime import disk_gate
        limits = self.permit['storage']
        return disk_gate(self.root, limits['checkpoint_reserve_bytes']+additional,
            minimum_free_bytes=limits['minimum_free_bytes'], maximum_run_bytes=limits['maximum_run_bytes'])

    def commands(self, jobs):
        from .runtime_watch import StageWatch
        from .seed_queue import verify_sources
        verify_sources(self.plan); self.disk()
        self.active_stage = ','.join(label for _, label, _ in jobs)
        watch = StageWatch(self.root, self.root, self.active_stage, None)
        children = []; next_disk = 0
        try:
            for arguments, label, gpu in jobs:
                path = self.root/'logs'/(label+'.log'); path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.open('x'); env = os.environ.copy()
                env.update(TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                    PYTHONPATH=str(REPO)+os.pathsep+env.get('PYTHONPATH', ''))
                env.pop('SKILLNET_PHASE12_RECOVERY_AUTHORIZATION', None)
                try:
                    child = subprocess.Popen([sys.executable, '-u', '-B', '-m', *map(str, arguments)],
                        cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    stream.close(); raise
                children.append((child, stream, label))
            watch.tick(children, event='started', force=True)
            while any(p.poll() is None for p, _, _ in children):
                if any(p.poll() not in (None, 0) for p, _, _ in children):
                    raise RuntimeError('Assessment child failed; preserve evidence, no automatic retry')
                if time.time() >= next_disk:
                    self.disk(); next_disk = time.time()+60
                watch.tick(children)
                time.sleep(1)
            if any(p.returncode != 0 for p, _, _ in children):
                raise RuntimeError('Assessment stage failed; see retained logs')
        finally:
            for p, _, _ in children:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)  # Only newly created assessment child sessions.
            for p, stream, _ in children:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait()
                stream.close()
            watch.tick(children, event='finished' if all(p.returncode == 0 for p, _, _ in children) else 'stopped', force=True)

    def run(self):
        from .assets import model_inventory
        from .first_calls_defer import gpu_users, same_process, process_identity
        from .first_calls_reuse import import_endpoint
        from .first_calls_report import report, publish
        from .seed_queue import verify_sources
        from .window_storage import seal_window
        from phase2.protocol import validate_extended
        verify_sources(self.plan)
        if (self.root/'launch.json').exists():
            raise FileExistsError('Do not implicitly retry a failed assessment')
        handoff = read_json(self.path.parent/'handoff.json')
        if (handoff['status'] != 'ready' or handoff['plan_sha256'] != file_hash(self.path)
                or handoff['boundary']['exit_code'] != 0 or gpu_users()):
            raise PermissionError('The registered successful post-RL idle-GPU handoff is required')
        for binding in handoff['paused_coordinators']:
            if not same_process(binding) or process_identity(binding['pid'])['state'] not in ('T', 't'):
                raise PermissionError('Legacy coordinators must remain safely deferred during assessment')
        if not self.plan['wait_for_seed505_RL']:
            registration = Path(self.plan['policy_registration'])
            if file_hash(registration) != self.plan['policy_registration_sha256']:
                raise ValueError('The earlier frozen coverage rule changed')
            boundary = handoff['boundary']
            receipt = read_json(boundary['legacy_queue_finished'])
            if (file_hash(boundary['legacy_queue_finished']) != boundary['sha256']
                    or receipt['status'] != 'complete' or receipt['completed_seeds'] != [404, 505, 606]):
                raise PermissionError('Followups require the completed, source-frozen legacy queue')
        for update in (0, 5):
            if model_inventory(self.root/'models'/f'u{update:04d}') != self.plan['model_inventory'][f'u{update:04d}']:
                raise ValueError('Retained endpoint weights changed')
        verify_legacy_seal(self.plan)
        validate_extended(read_json(self.root/'protocol.json'), REPO)
        self.disk(self.plan['projected_recording_upper_bytes'])
        write_new_json(self.root/'launch.json', {'plan_sha256': file_hash(self.path), 'started_unix': time.time(),
            'training_executed': False, 'performance_resampled': False, 'automatic_retry': False})
        try:
            # Exact old decisions are retained; only missing skill readouts are added.
            self.commands([(['skillnet_cohort.first_calls_measure', '--root', self.root, '--shard', rank],
                f'readout-shard{rank}', rank) for rank in range(8)])
            self.commands([(['phase2.aggregate', '--root', self.root, '--update', 5,
                '--start-update', 0, '--shards', 8], 'readout-aggregate', None)])
            import_endpoint(self.root, 0)
            self.evaluate(0)
            lock_prediction(self.root)
            import_endpoint(self.root, 5)
            self.evaluate(5)
            self.active_stage = 'analysis_and_publication'
            report(self.root)
            seal_window(self.root, 0, 5)
            publish(self.root)
            write_new_json(self.root/'complete.json', {'status': 'complete', 'training_reexecuted': False,
                'reports_published': True, 'old_evidence_preserved': True})
        except BaseException as error:
            write_new_json(self.root/'stopped.json', {'stage': self.active_stage,
                'error': f'{type(error).__name__}: {error}', 'automatic_retry': False})
            raise

    def evaluate(self, update):
        self.commands([(['phase2.evaluate', '--root', self.root, '--update', update,
            '--shards', 8, '--shard', rank], f'utility-u{update:04d}-shard{rank}', rank) for rank in range(8)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        print({'state': 'NOT_STARTED', 'training_executed': False}); return
    def stop(signum, frame):
        raise InterruptedError('Assessment termination requested; preserve all evidence')
    signal.signal(signal.SIGTERM, stop)
    assessment = Assessment(a.plan)
    with exclusive_writer(assessment.root):
        assessment.run()


if __name__ == '__main__':
    main()
