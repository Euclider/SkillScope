"""No-update native GPU regression for the finite-exponential repair."""
import os
from pathlib import Path
import subprocess
import sys
import torch

BASE = Path('/mnt/workspace/users/wangyifan/phase3-speed-742f1aa-BlbHzX')
CANDIDATE = BASE / 'candidate-finite-exp-v2'
RUN = Path('/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v7-20260927/runs/skillrl_failure')
OUTPUT = BASE / 'finite-exp-acceptance-v2'
sys.path.insert(0, str(CANDIDATE))
from phase3.common import require, strict_json, write_new
from skillnet_cohort.common import file_hash
from alfworld_rl_speed.runtime import acceptable
from alfworld_rl_speed.launch import fingerprint


def main():
    OUTPUT.mkdir(exist_ok=False)
    frozen = fingerprint()
    source = RUN / 'direction_batches/u0026.pt'
    metadata = strict_json(source.with_suffix('.json').read_text())
    require(file_hash(source) == metadata['sha256'], 'Changed captured U26 batch')
    b = torch.load(source, weights_only=True, map_location='cpu', mmap=True)
    t = b['tensors']
    severity = t['old_log_probs'].masked_fill(~t['actual_loss_mask'].bool(), float('inf')).min(1).values
    indices = severity.argsort()[:128]
    small = {**b, 'tensors': {k: v[indices].clone() for k, v in t.items()},
             'metadata': [b['metadata'][i] for i in indices.tolist()]}
    batch = OUTPUT / 'extreme-old-logprob-batch.pt'
    with batch.open('xb') as stream:
        torch.save(small, stream)
    write_new(OUTPUT / 'subset.json', {'source_sha256': metadata['sha256'],
        'batch_sha256': file_hash(batch), 'indices': indices.tolist(),
        'engineering_only': True, 'no_optimizer_updates': True})
    del b, t, small
    reports = {}
    for mode in ('baseline', 'no_sync'):
        directory = OUTPUT / mode
        directory.mkdir()
        cmd = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
            '-m', 'alfworld_rl_speed.probe', '--batch', str(batch),
            '--config', str(RUN / 'segments/u0025-u0030.json'),
            '--native', str(RUN / 'checkpoints/global_step_25'), '--output', str(directory),
            '--baseline', str(OUTPUT / 'baseline'), '--mode', mode]
        if mode == 'no_sync':
            cmd.append('--gpu-reference')
        env = {**os.environ, 'PYTHONPATH': str(CANDIDATE), 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
            'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1',
            'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'NCCL_CUMEM_ENABLE': '0',
            'VLLM_ALLREDUCE_USE_SYMM_MEM': '0', 'NCCL_DEBUG': 'WARN', 'VLLM_LOGGING_LEVEL': 'WARN'}
        print('PROBE_START', mode, flush=True)
        with (directory / 'probe.log').open('x') as log:
            code = subprocess.run(cmd, cwd=CANDIDATE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        require(code == 0, f'GPU regression failed: {mode}; preserved log {directory}')
        reports[mode] = strict_json((directory / 'result.json').read_text())
        require(acceptable(reports[mode]), f'GPU regression acceptance failed: {mode}')
        print('PROBE_PASSED', mode, flush=True)
    require(fingerprint() == frozen, 'Source changed during GPU regression')
    write_new(OUTPUT / 'receipt.json', {'passed': True, 'source_hashes': frozen,
        'reports': reports, 'batch_sha256': file_hash(batch),
        'scope': 'U25 native checkpoint; 128 low-probability U26 rows; no optimizer step/checkpoint writes',
        'not_a_reproduction_of_the_full_failed_U26_update': True})
    print('FINITE_EXP_GPU_REGRESSION_PASSED', flush=True)


def final_tail_regression():
    """Recheck final saturation algebra; reuse the sealed numerical baseline."""
    output = BASE / 'finite-exp-acceptance-v3'
    output.mkdir(exist_ok=False)
    previous = strict_json((OUTPUT / 'receipt.json').read_text())
    frozen = fingerprint()
    require(previous['passed'] and set(previous['source_hashes']) == set(frozen), 'Foreign baseline')
    require([k for k in frozen if frozen[k] != previous['source_hashes'][k]] ==
            ['verl/trainer/ppo/core_algos.py'], 'Unexpected source change since baseline')
    batch = OUTPUT / 'extreme-old-logprob-batch.pt'
    require(file_hash(batch) == previous['batch_sha256'], 'Changed probe rows')
    cmd = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
        '-m', 'alfworld_rl_speed.probe', '--batch', str(batch), '--config', str(RUN / 'segments/u0025-u0030.json'),
        '--native', str(RUN / 'checkpoints/global_step_25'), '--output', str(output / 'no_sync'),
        '--baseline', str(OUTPUT / 'baseline'), '--mode', 'no_sync', '--gpu-reference']
    env = {**os.environ, 'PYTHONPATH': str(CANDIDATE), 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
        'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1',
        'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'NCCL_CUMEM_ENABLE': '0',
        'VLLM_ALLREDUCE_USE_SYMM_MEM': '0', 'NCCL_DEBUG': 'WARN', 'VLLM_LOGGING_LEVEL': 'WARN'}
    with (output / 'probe.log').open('x') as log:
        code = subprocess.run(cmd, cwd=CANDIDATE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    require(code == 0, 'Final eight-GPU regression failed')
    report = strict_json((output / 'no_sync/result.json').read_text())
    require(acceptable(report) and fingerprint() == frozen, 'Final source/parity not accepted')
    write_new(output / 'receipt.json', {'passed': True, 'source_hashes': frozen,
        'reports': {'baseline': previous['reports']['baseline'], 'no_sync': report},
        'baseline_receipt_sha256': file_hash(OUTPUT / 'receipt.json'),
        'batch_sha256': file_hash(batch), 'scope': 'Final saturated K3 algebra; no parameter updates',
        'not_a_reproduction_of_the_full_failed_U26_update': True})
    print('FINAL_FINITE_EXP_GPU_REGRESSION_PASSED', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--final-tail']:
        final_tail_regression()
    else:
        main()
