"""Authorized SkillRL-only continuation; frozen training implementation reused."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v7-20260927')
OUTPUT = ROOT / 'recovery-skillrl-u50-v1'
AUTHORITY = OUTPUT / 'authorization.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def training_parser():
    parser = argparse.ArgumentParser()
    for name in ('preparation', 'root', 'branch', 'bank-path', 'bank-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--resume-update', type=int)
    parser.add_argument('--execute', action='store_true')
    return parser


def validate_extension(record, root, branch, start, preparation):
    from phase3.common import require
    require(record['schema'] == 'phase3.skillrl_u50.v1'
            and record['branch'] == branch == 'skillrl_failure'
            and record['start_update'] == 20 and record['stop_update'] == 50,
            'Unregistered branch or continuation horizon')
    require(Path(root).resolve() == Path(record['run_root']) / 'runs' / branch
            and Path(preparation).resolve() == Path(record['preparation'])
            and type(start) is int and 20 <= start < 50 and start % 5 == 0,
            'Unregistered continuation window or preparation')


def training_route(args, record):
    args = list(map(str, args))
    if args[0] != 'phase3.training':
        return None
    parsed = training_parser().parse_args(args[1:])
    from phase3.common import require
    require(parsed.execute, 'Continuation training requires explicit execution')
    validate_extension(record, parsed.root, parsed.branch, parsed.start, parsed.preparation)
    return [sys.executable, '-B', str(Path(__file__).resolve()), '--train', *args[1:]]


def checked_authority():
    from phase3.common import require, strict_json
    from skillnet_cohort.common import file_hash
    record = strict_json(AUTHORITY.read_text())
    validate_extension(record, ROOT / 'runs/skillrl_failure', 'skillrl_failure', 20, record['preparation'])
    require(record['run_root'] == str(ROOT) and record['editor_timeout_seconds'] == 600,
            'Foreign continuation authority')
    for name, expected in record['source_hashes'].items():
        require(file_hash(name) == expected, 'Continuation source changed: ' + name)
    require(file_hash(record['base_training_request']) == record['base_training_request_sha256'],
            'Base acceleration request changed')
    return record


def checked_training_base(record):
    """Validate original U15 acceptance, then separate U20–U50 authorization.

    The old launcher's U20 bound is not patched. This entrypoint checks its
    original registered U15 boundary only to verify that deployment receipt;
    actual future windows are independently constrained by validate_extension.
    """
    from alfworld_rl_speed.launch import checked_request, accept
    from phase3.common import require, strict_json
    from skillnet_cohort.common import file_hash
    base = checked_request(Path(record['base_training_request']),
        ROOT / 'runs/skillrl_failure', 'skillrl_failure', 15)
    dependency = strict_json(Path(record['dependency_receipt']).read_text())
    require(file_hash(record['dependency_receipt']) == record['dependency_receipt_sha256'],
            'Dependency completion receipt changed')
    for name, expected in dependency['added_files'].items():
        require(file_hash(Path(base['candidate']) / name) == expected, 'Training dependency changed')
    receipt_path, receipt = accept(base)
    require(file_hash(receipt_path) == record['acceptance_sha256'], 'Accepted acceleration changed')
    return base, receipt_path, receipt


def configure_with_receipt(cfg, receipt_path, receipt, root):
    from alfworld_rl_speed.runtime import apply_options
    from skillnet_cohort.common import file_hash
    cfg = apply_options(cfg, receipt['options'])
    cfg.phase3.speed_receipt = str(receipt_path)
    cfg.phase3.speed_receipt_sha256 = file_hash(receipt_path)
    cfg.actor_rollout_ref.actor.speed_audit_root = str(Path(root) / 'speed-audits')
    cfg.actor_rollout_ref.actor.speed_receipt_sha256 = file_hash(receipt_path)
    return cfg


def train(argv):
    from phase3.common import require, write_new
    from skillnet_cohort.common import file_hash
    from phase3 import training
    args = training_parser().parse_args(argv)
    record = checked_authority()
    require(args.execute, 'Missing training execution authorization')
    validate_extension(record, args.root, args.branch, args.start, args.preparation)
    base, receipt_path, receipt = checked_training_base(record)
    original = training.configuration
    def configuration(*a, **kw):
        return configure_with_receipt(original(*a, **kw), receipt_path, receipt, args.root)
    training.configuration = configuration
    write_new(Path(args.root) / 'speed-profiles' / f'u{args.start:04d}-u{args.start+5:04d}.json',
        {'receipt': str(receipt_path), 'sha256': file_hash(receipt_path), 'options': receipt['options'],
         'start_update': args.start, 'source_commit': base['upstream_commit'],
         'previous_windows_unchanged': True, 'continuation_authority_sha256': file_hash(AUTHORITY)})
    print(f'ACCEPTED_SPEED_CONTINUATION start={args.start} end={args.start+5} stop=50', flush=True)
    training.execute(args.preparation, args.root, args.branch, args.bank_path,
                     args.bank_sha256, args.start, True, args.resume_update)


def check_training():
    from phase3.common import require, strict_json
    from phase3.training import configuration
    from verl.trainer.main_ppo import run_ppo
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from gigpo import core_gigpo
    record = checked_authority()
    _, receipt_path, receipt = checked_training_base(record)
    root = ROOT / 'runs/skillrl_failure'
    bank_hash = strict_json((root / 'events/u0015/complete.json').read_text())['selected_bank_sha256']
    for start in (20, 45):
        cfg = configure_with_receipt(configuration(record['preparation'], root, 'skillrl_failure',
            root / 'banks' / f'{bank_hash}.json', bank_hash, start), receipt_path, receipt, root)
        require(cfg.phase3.segment_start == start and cfg.phase3.segment_end == start + 5
                and cfg.trainer.total_training_steps == 150
                and cfg.actor_rollout_ref.actor.optim.lr == 1e-6
                and cfg.data.train_batch_size * cfg.env.rollout.n == 128
                and cfg.algorithm.adv_estimator == 'grpo'
                and cfg.trainer.resume_from_path == str(root / 'checkpoints' / f'global_step_{start}'),
                'Continuation changed frozen training recipe')
    print('U20_AND_U45_CONFIG_AND_FULL_TRAINING_IMPORT_VERIFIED', flush=True)


def stage(stop):
    from phase3 import run
    from phase3.api import JSONClient
    from phase3.common import require, write_new
    record = checked_authority()
    original = run.command
    def command(args, log):
        argv = training_route(args, record)
        if argv is None:
            return original(args, log)
        require(not log.exists(), 'Prior training log exists; reconcile before retry')
        log.parent.mkdir(parents=True, exist_ok=True)
        candidate = json.loads(Path(record['base_training_request']).read_text())['candidate']
        with log.open('x') as stream:
            code = subprocess.run(argv, cwd=candidate,
                env={**os.environ, 'PYTHONPATH': candidate, 'PYTHONDONTWRITEBYTECODE': '1'},
                stdout=stream, stderr=subprocess.STDOUT).returncode
        require(code == 0, f'Continuation training failed; inspect {log}')
    def editor(config, ledger_path, **kwargs):
        require(config.stage == 'editor' and Path(ledger_path) == ROOT / 'runs/skillrl_failure/editor.sqlite3',
                'Timeout extension is restricted to this SkillRL ledger')
        client = JSONClient(config, ledger_path, **kwargs)
        receipt = client.authorize_transport_timeout(timeout_seconds=600)
        write_new(OUTPUT / 'editor-transport-timeout.json', receipt)
        return client
    run.command, run.JSONClient = command, editor
    run.execute(record['preparation'], ROOT / 'runs/skillrl_failure', 'skillrl_failure',
                stop_update=stop, retry_editor_request=record['retry_request'] if stop == 20 else None)


def preflight():
    from phase3.common import require, strict_json, digest
    from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
    record = checked_authority()
    root = ROOT / 'runs/skillrl_failure'
    require(validate_full_checkpoint(root / 'checkpoints/global_step_20')['world_size'] == 8,
            'Missing eight-rank U20 checkpoint')
    require((root / 'models/u0020/phase2_export.json').is_file()
            and (root / 'predictions/u0015-u0020/complete.json').is_file(), 'U20 export/readout incomplete')
    with sqlite3.connect(f'file:{root}/editor.sqlite3?mode=ro', uri=True) as db:
        row = db.execute('SELECT request,result FROM attempts WHERE key=?', (record['retry_request'],)).fetchone()
        require(row is not None, 'U20 timeout request missing')
        request, result = map(strict_json, row)
        require(digest(request) == record['retry_request'] and request['identity']['event_id'] == 'u0020'
                and result['status'] == 'failed' and result['failure']['type'] == 'APITimeoutError',
                'Not the registered U20 timeout')
        require(db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 4,
                'Editor ledger advanced; reconcile before launching')
    candidate = json.loads(Path(record['base_training_request']).read_text())['candidate']
    result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--check-training'],
        cwd=candidate, env={**os.environ, 'PYTHONPATH': candidate, 'CUDA_VISIBLE_DEVICES': '',
            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
        text=True, capture_output=True, timeout=180)
    require(result.returncode == 0, result.stdout + result.stderr)
    require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader'], text=True).strip(), 'Another GPU job is active')
    return {'utc': now(), 'training_preflight': result.stdout, 'stop_update': 50,
            'branch': 'skillrl_failure', 'u20_training_and_readout_replayed': False,
            'other_arms_started': False, 'editor_timeout_seconds': 600}


def launch():
    from phase3.common import require, write_new
    require(not (OUTPUT / 'launch.json').exists(), 'Continuation already launched; inspect its records')
    record = preflight()
    env = dict(os.environ)
    credential = env.get('SKILLRL_PHASE3_EDITOR_API_KEY') or getpass.getpass('Editor credential (hidden): ')
    require(bool(credential.strip()), 'Missing editor credential')
    env.update(SKILLRL_PHASE3_EDITOR_API_KEY=credential.strip(), PYTHONPATH=str(REPO),
        ALFWORLD_DATA='/mnt/workspace/users/wangyifan/skill-RL/data/alfworld',
        XDG_CACHE_HOME=str(ROOT / 'cache'), TRITON_CACHE_DIR=str(ROOT / 'cache/triton'),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    write_new(OUTPUT / 'launch.json', record)
    with (OUTPUT / 'queue.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'],
            cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    write_new(OUTPUT / 'process.json', {'pid': child.pid, 'utc': now()})
    print(f'SKILLRL_U50_QUEUE_STARTED pid={child.pid} output={OUTPUT}', flush=True)


def worker():
    from phase3.common import write_new
    from skillnet_cohort.common import exclusive_writer
    with exclusive_writer(OUTPUT):
        for stop in (20, 50):
            print(f'{now()} START skillrl_failure stop={stop}', flush=True)
            with (OUTPUT / f'runner-u{stop:04d}.log').open('x') as log:
                code = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--stage', str(stop)],
                    cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT).returncode
            if code or not (ROOT / f'runs/skillrl_failure/milestones/u{stop:04d}/complete.json').is_file():
                write_new(OUTPUT / 'stopped.json', {'stop': stop, 'returncode': code, 'utc': now()})
                raise SystemExit(1)
            print(f'{now()} SEALED skillrl_failure U{stop}', flush=True)
        write_new(OUTPUT / 'complete.json', {'utc': now(), 'branch': 'skillrl_failure', 'update': 50})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--check-training', action='store_true')
    modes.add_argument('--train', action='store_true')
    modes.add_argument('--stage', type=int, choices=(20, 50))
    modes.add_argument('--worker', action='store_true')
    args, remaining = parser.parse_known_args()
    if args.train or args.check_training:
        base = json.loads((ROOT / 'rl-speed-next-window-v1.json').read_text())
        sys.path.insert(0, base['candidate'])
    else:
        sys.path.insert(0, str(REPO))
    if args.train:
        train(remaining)
    elif remaining:
        parser.error('Unexpected arguments')
    elif args.check_training:
        check_training()
    elif args.check:
        print(preflight())
    elif args.stage:
        stage(args.stage)
    elif args.worker:
        worker()
    else:
        launch()
