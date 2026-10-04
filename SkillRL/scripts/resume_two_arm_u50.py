"""Approved ordered continuation: SkillRL U50, then reward readout U20->U50."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import getpass
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v7-20260927')
OUTPUT = ROOT / 'recovery-two-arm-u50-finite-exp-v2'
AUTHORITY = OUTPUT / 'authorization.json'
ORDER = ('skillrl_failure', 'readout_d')
STARTS = {'skillrl_failure': 25, 'readout_d': 20}


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_window(branch, start):
    from phase3.common import require
    require(branch in STARTS and type(start) is int
            and STARTS[branch] <= start < 50 and start % 5 == 0,
            'Unregistered branch/window; completed updates must not be replayed')


def milestone_ready(root, branch):
    from phase3.common import strict_json
    p = Path(root) / 'runs' / branch / 'milestones/u0050'
    if not all((p / name).is_file() for name in
               ('complete.json', 'valid_seen/complete.json', 'valid_unseen/complete.json')):
        return False
    r = strict_json((p / 'complete.json').read_text())
    return r.get('branch') == branch and r.get('endpoint') == 50


def authority():
    from phase3.common import strict_json, require
    from skillnet_cohort.common import file_hash
    r = strict_json(AUTHORITY.read_text())
    require(r['run_root'] == str(ROOT) and r['order'] == list(ORDER)
            and r['starts'] == STARTS and r['stop_update'] == 50, 'Foreign continuation authority')
    for name, sha in r['sources'].items():
        require(file_hash(name) == sha, 'Registered continuation source changed: ' + name)
    for name, sha in r['evidence'].items():
        require(file_hash(name) == sha, 'Registered acceptance/evidence changed: ' + name)
    receipt = strict_json(Path(r['numeric_receipt']).read_text())
    require(receipt['passed'], 'GPU regression not accepted')
    for relative, sha in receipt['source_hashes'].items():
        require(file_hash(Path(r['candidate']) / relative) == sha,
                'Accepted training source changed: ' + relative)
    for relative, sha in r['additional_dependencies'].items():
        require(file_hash(Path(r['candidate']) / relative) == sha, 'Added dependency changed')
    return r


def train(rest):
    from phase3 import training
    from phase3.common import require, write_new
    from skillnet_cohort.common import file_hash
    from alfworld_rl_speed.runtime import apply_options
    parser = argparse.ArgumentParser()
    for name in ('preparation', 'root', 'branch', 'bank-path', 'bank-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--resume-update', type=int)
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args(rest)
    r = authority()
    validate_window(a.branch, a.start)
    require(a.execute and a.resume_update is None and Path(a.root) == ROOT / 'runs' / a.branch
            and Path(a.preparation) == Path(r['preparation']), 'Wrong training boundary')
    require(Path(training.__file__).resolve().is_relative_to(Path(r['candidate'])), 'Wrong trainer deployment')
    original = training.configuration
    def configuration(*args, **kwargs):
        cfg = apply_options(original(*args, **kwargs), r['speed_options'])
        cfg.phase3.speed_receipt = r['numeric_receipt']
        cfg.phase3.speed_receipt_sha256 = file_hash(r['numeric_receipt'])
        cfg.phase3.numerical_repair = 'saturated_exp_finite_backward_v2'
        cfg.actor_rollout_ref.actor.speed_audit_root = str(Path(a.root) / 'speed-audits')
        cfg.actor_rollout_ref.actor.speed_receipt_sha256 = file_hash(r['numeric_receipt'])
        require(cfg.algorithm.adv_estimator == 'grpo' and cfg.actor_rollout_ref.actor.optim.lr == 1e-6
                and cfg.data.train_batch_size * cfg.env.rollout.n == 128
                and cfg.trainer.total_training_steps == 150
                and cfg.trainer.resume_from_path == str(Path(a.root) / 'checkpoints' / f'global_step_{a.start}'),
                'Frozen GRPO recipe or checkpoint changed')
        return cfg
    training.configuration = configuration
    write_new(Path(a.root) / 'speed-profiles' / f'u{a.start:04d}-u{a.start+5:04d}.json',
        {'receipt': r['numeric_receipt'], 'sha256': file_hash(r['numeric_receipt']),
         'options': r['speed_options'], 'start_update': a.start,
         'continuation_authority_sha256': file_hash(AUTHORITY), 'previous_windows_unchanged': True})
    print(f'FINITE_EXP_CONTINUATION branch={a.branch} start={a.start} stop=50', flush=True)
    training.execute(a.preparation, a.root, a.branch, a.bank_path, a.bank_sha256, a.start, True)


def predict(rest):
    from phase3.common import require, strict_json
    from phase3.parallel_predict import source_hashes, run_parallel
    r = authority()
    require(strict_json(Path(r['readout_receipt']).read_text())['source_hashes'] == source_hashes(),
            'Unaccepted readout source')
    p = argparse.ArgumentParser()
    for name in ('bank', 'old-path', 'new-path', 'batch', 'output', 'calibration', 'identity'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--bank-sha256', required=True)
    p.add_argument('--parity-atol', type=float, required=True)
    a = p.parse_args(rest)
    identity = strict_json(a.identity.read_text())
    validate_window(identity['branch_id'], identity['start'])
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3,4,5,6,7', 'Wrong scoring allocation')
    run_parallel(a, list(range(8)))


def stage(branch):
    from phase3 import run
    from phase3.api import JSONClient
    from phase3.common import require, write_new
    r = authority()
    if branch == 'readout_d':
        require(milestone_ready(ROOT, 'skillrl_failure'), 'SkillRL U50 evaluation must finish first')
    original = run.command
    def command(args, log):
        args = list(map(str, args))
        if args[0] not in ('phase3.training', 'phase3.predict'):
            return original(args, log)
        mode = '--train' if args[0] == 'phase3.training' else '--predict'
        target = Path(log)
        if target == ROOT / 'runs/skillrl_failure/logs/train-u0025-u0030.log':
            target = OUTPUT / 'train-skillrl-u0025-u0030-recovery.log'
        require(not target.exists(), 'Prior subprocess log exists; explicit reconciliation required')
        target.parent.mkdir(parents=True, exist_ok=True)
        cwd = r['candidate'] if mode == '--train' else str(REPO)
        with target.open('x') as stream:
            code = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), mode, *args[1:]],
                cwd=cwd, env={**os.environ, 'PYTHONPATH': cwd, 'PYTHONDONTWRITEBYTECODE': '1'},
                stdout=stream, stderr=subprocess.STDOUT).returncode
        require(code == 0, f'Continuation subprocess failed; see {target}')
    def editor(config, ledger_path, **kwargs):
        client = JSONClient(config, ledger_path, **kwargs)
        record = client.authorize_transport_timeout(timeout_seconds=600)
        write_new(OUTPUT / f'editor-timeout-{branch}.json', record)
        return client
    run.command, run.JSONClient = command, editor
    run.execute(r['preparation'], ROOT / 'runs' / branch, branch, stop_update=50)


def archive_incomplete(r):
    """Preserve only the incomplete U26 attempt; never move completed evidence."""
    from phase3.common import require, write_new
    from skillnet_cohort.common import file_hash
    run = ROOT / 'runs/skillrl_failure'
    require(not (run / 'metrics/u0026.json').exists() and not (run / 'checkpoints/global_step_30').exists(),
            'U26 advanced; cannot replay')
    for relative, sha in r['failed_artifacts'].items():
        require(file_hash(run / relative) == sha, 'Failed attempt changed: ' + relative)
    targets = ('direction_batches/u0026.pt', 'direction_batches/u0026.json',
               'episodes/u0026', 'segments/u0025-u0030.json', 'speed-profiles/u0025-u0030.json')
    destination = OUTPUT / 'preserved-incomplete-u26'
    require(not destination.exists(), 'Failed attempt already archived; inspect before retry')
    write_new(OUTPUT / 'archive-intent.json', {'targets': list(targets), 'sha256': r['failed_artifacts'],
        'reason': 'U26 optimizer attempt failed; no completed U26 update/checkpoint; restore sealed U25'})
    for relative in targets:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        (run / relative).rename(target)
    write_new(OUTPUT / 'archive-complete.json', {'targets': list(targets), 'utc': now()})


def launch():
    from phase3.common import require, write_new
    from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
    r = authority()
    require(not (OUTPUT / 'launch.json').exists(), 'Already launched; inspect records')
    for branch, start in STARTS.items():
        run = ROOT / 'runs' / branch
        require(validate_full_checkpoint(run / 'checkpoints' / f'global_step_{start}')['world_size'] == 8,
                'Incomplete resume checkpoint')
        require((run / 'events' / f'u{start:04d}/complete.json').is_file()
                and (run / 'milestones/u0020/complete.json').is_file(), 'Unsealed starting endpoint')
    require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader'], text=True).strip(), 'GPU workers still active')
    env = dict(os.environ)
    key = env.get('SKILLRL_PHASE3_EDITOR_API_KEY') or getpass.getpass('Editor credential (hidden): ')
    require(bool(key.strip()), 'Missing credential')
    env.update(SKILLRL_PHASE3_EDITOR_API_KEY=key.strip(), PYTHONPATH=str(REPO),
        ALFWORLD_DATA='/mnt/workspace/users/wangyifan/skill-RL/data/alfworld',
        XDG_CACHE_HOME=str(ROOT / 'cache'), TRITON_CACHE_DIR=str(ROOT / 'cache/triton'),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    archive_incomplete(r)
    write_new(OUTPUT / 'launch.json', {'utc': now(), 'order': list(ORDER), 'starts': STARTS,
        'stop_update': 50, 'completed_results_replayed': False, 'retry_incomplete_update': 26})
    with (OUTPUT / 'queue.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'],
            cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    write_new(OUTPUT / 'process.json', {'pid': child.pid, 'utc': now()})
    print(f'TWO_ARM_U50_QUEUE_STARTED pid={child.pid} output={OUTPUT}', flush=True)


def worker():
    from phase3.common import write_new
    from skillnet_cohort.common import exclusive_writer
    with exclusive_writer(OUTPUT):
        for branch in ORDER:
            print(f'{now()} START {branch} target=U50', flush=True)
            with (OUTPUT / f'{branch}-runner.log').open('x') as log:
                code = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--stage', branch],
                    cwd=REPO, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
            if code or not milestone_ready(ROOT, branch):
                write_new(OUTPUT / 'stopped.json', {'branch': branch, 'returncode': code, 'utc': now()})
                raise SystemExit(1)
            print(f'{now()} SEALED {branch} U50 seen+unseen', flush=True)
            for _ in range(24):
                if not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                        '--format=csv,noheader'], text=True).strip():
                    break
                time.sleep(5)
            else:
                write_new(OUTPUT / 'stopped.json', {'branch': branch, 'reason': 'GPU workers remain', 'utc': now()})
                raise SystemExit(1)
        write_new(OUTPUT / 'complete.json', {'utc': now(), 'arms': list(ORDER), 'update': 50})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--train', action='store_true')
    modes.add_argument('--predict', action='store_true')
    modes.add_argument('--stage', choices=ORDER)
    modes.add_argument('--worker', action='store_true')
    a, rest = p.parse_known_args()
    if not a.train:
        sys.path.insert(0, str(REPO))
    if a.train:
        train(rest)
    elif a.predict:
        predict(rest)
    elif rest:
        p.error('Unexpected arguments')
    elif a.stage:
        stage(a.stage)
    elif a.worker:
        worker()
    else:
        launch()
