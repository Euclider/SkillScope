"""Boundary-only local acceptance, then native training with a sealed profile."""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from phase3.common import require, strict_json, write_new
from skillnet_cohort.common import file_hash
from .runtime import acceptable, apply_options, choose_options

CODE = Path(__file__).resolve().parents[1]


def fingerprint():
    roots = ('alfworld_rl_speed', 'verl', 'phase1', 'phase2', 'phase3',
             'skillnet_cohort', 'agent_system', 'scripts', 'configs')
    return {str(path.relative_to(CODE)): file_hash(path) for root in roots
            for path in sorted((CODE / root).rglob('*'))
            if path.is_file() and path.suffix in ('.py', '.json', '.yaml', '.yml')}


def dependencies():
    return {p: importlib.metadata.version(p) for p in
            ('torch', 'transformers', 'vllm', 'ray', 'numpy', 'omegaconf', 'tensordict')}


def hardware():
    return subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()


def checked_request(path, root, branch, start):
    request = strict_json(path.read_text())
    require(request['schema'] == 'alfworld.phase3.rl_speed.request.v1', 'Foreign speed request')
    require(str(CODE) == request['candidate'], 'Foreign candidate path')
    require(Path(root).resolve() == Path(request['run_root']) / 'runs' / branch, 'Foreign run/branch')
    require(branch in request['start_updates'] and start >= request['start_updates'][branch]
            and start % 5 == 0 and start < 20, 'Unregistered speed boundary')
    require(request['source_hashes'] == fingerprint(), 'Speed source fingerprint changed')
    require(request['dependencies'] == dependencies(), 'Speed dependencies changed')
    require(request['hardware'] == hardware(), 'Speed GPU/driver identity changed')
    for name, expected in request['inputs'].items():
        require(file_hash(name) == expected, f'Speed input changed: {name}')
    return request


def run_probe(request, mode, *, gpu_ref=False):
    output = Path(request['acceptance_root']) / mode
    # Never retry/overwrite a failed probe implicitly. Its failure remains part
    # of the local receipt; a failed optional mode stays disabled.
    if (output / 'result.json').exists():
        return strict_json((output / 'result.json').read_text())
    if output.exists():
        return {}
    output.mkdir(parents=True)
    argv = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc-per-node=8', '-m', 'alfworld_rl_speed.probe', '--mode', mode,
        '--batch', request['batch'], '--config', request['baseline_config'],
        '--native', request['checkpoint'], '--output', str(output),
        '--baseline', str(Path(request['acceptance_root']) / 'baseline')]
    if gpu_ref:
        argv.append('--gpu-reference')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(CODE), TOKENIZERS_PARALLELISM='false')
    env.update(CUDA_DEVICE_MAX_CONNECTIONS='1', NCCL_CUMEM_ENABLE='0',
               VLLM_ALLREDUCE_USE_SYMM_MEM='0', NCCL_DEBUG='WARN', VLLM_LOGGING_LEVEL='WARN')
    # Dedicated worker group and bounded probe; never signal the live queue.
    with (output / 'probe.log').open('x') as stream:
        proc = subprocess.Popen(argv, cwd=CODE, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=5400)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            code = -1
    write_new(output / 'process.json', {'returncode': code, 'mode': mode})
    if code or not (output / 'result.json').is_file():
        return {}
    return strict_json((output / 'result.json').read_text())


def accept(request):
    acceptance = Path(request['acceptance_root'])
    receipt_path = acceptance / 'receipt.json'
    if receipt_path.exists():
        receipt = strict_json(receipt_path.read_text())
        require(receipt['source_hashes'] == fingerprint() and receipt['dependencies'] == dependencies(),
                'Receipt no longer matches source/environment')
        for name, expected in receipt['probe_hashes'].items():
            require(file_hash(acceptance / name) == expected, 'Probe evidence changed')
        require(receipt['options'] == choose_options(receipt['reports']), 'Receipt options mismatch')
        return receipt_path, receipt
    require(Path(request['checkpoint']).is_dir(), 'Native boundary checkpoint missing')
    for name, expected in request['probe_inputs'].items():
        require(file_hash(name) == expected, 'Probe model/batch input changed')
    # Called synchronously by phase3.training, after readout/editor/gate have
    # exited. Refuse unexpected GPU users rather than competing with them.
    for _ in range(24):
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if not active:
            break
        time.sleep(5)  # Let just-exited gate workers release their CUDA contexts.
    require(not active, 'GPUs still in use at training acceptance boundary')
    baseline = run_probe(request, 'baseline')
    require(acceptable(baseline), 'Baseline local native checkpoint/minibatch probe failed')
    reports = {'suffix': run_probe(request, 'suffix')}
    require(acceptable(reports['suffix']), 'Response-only local gradient/parity probe failed')
    reports['reference_gpu_root'] = run_probe(request, 'reference_gpu_root')
    gpu_ref = acceptable(reports['reference_gpu_root'])
    if gpu_ref:
        require(all(r.get('vllm_colocation_passed') for r in reports['reference_gpu_root']['ranks']),
                'GPU reference lacks vLLM colocation acceptance')
    reports['no_sync'] = run_probe(request, 'no_sync', gpu_ref=gpu_ref)
    options = choose_options(reports)
    receipt = dict(schema='alfworld.phase3.rl_speed.receipt.v1', status='offline_minibatch_accepted',
        options=options, reports=reports, baseline=baseline, source_hashes=fingerprint(), dependencies=dependencies(),
        probe_hashes={str(p.relative_to(acceptance)): file_hash(p) for p in sorted(acceptance.rglob('*'))
                      if p.is_file()}, hardware=request['hardware'],
        input_hashes=request['probe_inputs'],
        first_full_rl_update='pending; per-rank optimizer counters/memory will be audited at runtime',
        numerical_contract='Not bitwise equivalent. LP/entropy <=1e-5; full preclip gradient relL2<=0.03 and cosine>=0.9995.',
        scope='Training only; no padding trimming, batch/normalization/sample/optimizer-boundary change.')
    write_new(receipt_path, receipt)
    return receipt_path, receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', type=Path, required=True)
    args, training_argv = p.parse_known_args()
    training_parser = argparse.ArgumentParser()
    for name in ('preparation', 'root', 'branch', 'bank-path', 'bank-sha256'):
        training_parser.add_argument('--' + name, required=True)
    training_parser.add_argument('--start', type=int, required=True)
    training_parser.add_argument('--resume-update', type=int)
    training_parser.add_argument('--execute', action='store_true')
    train = training_parser.parse_args(training_argv)
    require(train.execute, 'Cannot run acceptance/training without --execute')
    request = checked_request(args.request, train.root, train.branch, train.start)
    receipt_path, receipt = accept(request)
    # Configuration/restore follow the existing implementation. Only the
    # explicitly accepted fields are overlaid before recording the segment.
    from phase3 import training
    original = training.configuration
    def configuration(*args, **kwargs):
        cfg = apply_options(original(*args, **kwargs), receipt['options'])
        cfg.phase3.speed_receipt = str(receipt_path)
        cfg.phase3.speed_receipt_sha256 = file_hash(receipt_path)
        cfg.actor_rollout_ref.actor.speed_audit_root = str(Path(train.root) / 'speed-audits')
        cfg.actor_rollout_ref.actor.speed_receipt_sha256 = file_hash(receipt_path)
        return cfg
    training.configuration = configuration
    write_new(Path(train.root) / 'speed-profiles' / f'u{train.start:04d}-u{train.start+5:04d}.json',
        {'receipt': str(receipt_path), 'sha256': file_hash(receipt_path), 'options': receipt['options'],
         'start_update': train.start, 'source_commit': request['upstream_commit'],
         'previous_windows_unchanged': True})
    training.execute(train.preparation, train.root, train.branch, train.bank_path,
                     train.bank_sha256, train.start, True, train.resume_update)


if __name__ == '__main__':
    main()
