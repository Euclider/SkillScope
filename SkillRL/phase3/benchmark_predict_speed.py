"""Engineering-only replay: compare serial and paired-shard readout outputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from .common import digest, require, strict_json, write_new
from .parallel_predict import run_parallel, source_hashes
from skillnet_cohort.common import file_hash


def subset(source, target, count):
    import torch
    metadata = strict_json(source.with_suffix('.json').read_text())
    require(file_hash(source) == metadata['sha256'], 'Changed probe batch')
    batch = torch.load(source, weights_only=False, map_location='cpu')
    buckets, seen = {}, set()
    for i, row in enumerate(batch['metadata']):
        if row['decision_id'] in seen:
            continue
        seen.add(row['decision_id'])
        mask = batch['tensors']['actual_loss_mask'][i].bool()
        if not mask.any():
            continue
        adv = float(batch['tensors']['advantages'][i, mask][0])
        key = (int(adv > 0) - int(adv < 0), min(int(mask.sum()) // 16, 3))
        buckets.setdefault(key, []).append(i)
    indices = []
    while len(indices) < count and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(indices) < count:
                indices.append(buckets[key].pop(0))
    require(len(indices) == count, 'Too few probe decisions')
    selected = torch.tensor(indices)
    small = {**batch, 'metadata': [batch['metadata'][i] for i in indices],
             'tensors': {key: value.index_select(0, selected) for key, value in batch['tensors'].items()}}
    with target.open('xb') as stream:
        torch.save(small, stream)
    write_new(target.with_suffix('.json'), {**metadata, 'sha256': file_hash(target),
        'engineering_subset_only': True, 'source_batch_sha256': metadata['sha256'], 'row_indices': indices})


def benchmark(*, root, output, decisions=48):
    frozen_sources = source_hashes()
    output.mkdir(exist_ok=False, parents=True)
    run = root / 'runs/skillrl_failure'
    records = []
    # These are scoring tests, never formal update windows. For FP32 use
    # the retained U15 endpoint with the captured U0 batch because U5 was
    # already retired. Both implementations receive identical endpoints.
    for mode, update, start in [('bf16', 11, 10), ('fp32', 1, 0)]:
        directory = output / mode
        directory.mkdir()
        batch = directory / 'batch.pt'
        subset(run / f'direction_batches/u{update:04d}.pt', batch, decisions)
        source_identity = strict_json((run / f'events/u{start+5:04d}/identity.json').read_text())
        retained_identity = strict_json((run / 'events/u0015/identity.json').read_text())
        identity = {**source_identity, 'new_policy_sha256': retained_identity['new_policy_sha256']}
        write_new(directory / 'identity.json', identity)
        write_new(directory / 'scope.json', {'engineering_only': True,
            'not_a_formal_utility_or_evolution_window': True,
            'initial_batch': update, 'old_checkpoint_update': start, 'new_checkpoint_update': 15})
        calibration = strict_json((run / 'calibration-fp32-autocast-v2.json').read_text())
        write_new(directory / 'calibration.json', calibration)
        old = run / 'models/u0010' if start else Path('/mnt/workspace/users/wangyifan/model/Qwen3.5-4B')
        common = dict(bank=run / 'banks' / f"{identity['bank_sha256']}.json",
            bank_sha256=identity['bank_sha256'], old_path=old, new_path=run / 'models/u0015',
            batch=batch, calibration=directory / 'calibration.json', identity=directory / 'identity.json',
            parity_atol=.03)
        cmd = [sys.executable, '-B', '-m', 'phase3.predict']
        for key, value in common.items():
            cmd += ['--' + key.replace('_', '-'), str(value)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='2,3', OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1',
                   HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        print(f'{mode} SERIAL_START decisions={decisions}', flush=True)
        t0 = time.monotonic()
        with (directory / 'serial.log').open('x') as stream:
            subprocess.run(cmd + ['--output', str(directory / 'serial')], cwd=Path(__file__).resolve().parents[1],
                           env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        serial_seconds = time.monotonic() - t0
        print(f'{mode} PARALLEL_START serial_seconds={serial_seconds:.3f}', flush=True)
        t0 = time.monotonic()
        run_parallel(SimpleNamespace(**common, output=directory / 'parallel'), [2, 3, 4, 5, 6, 7])
        parallel_seconds = time.monotonic() - t0
        a = strict_json((directory / 'serial/readout.json').read_text())
        b = strict_json((directory / 'parallel/readout.json').read_text())
        ca = strict_json((directory / 'serial/complete.json').read_text())
        cb = strict_json((directory / 'parallel/complete.json').read_text())
        require(a == b, 'Parallel readout differs from serial: do not activate')
        for key in ('decisions', 'forward_calls', 'forward_input_tokens', 'chosen_logprob_max_abs_error'):
            require(ca[key] == cb[key], f'Scoring accounting/parity changed: {key}')
        row = dict(mode=mode, decisions=decisions, gpu_ids=[2, 3, 4, 5, 6, 7],
                   exact_bundle_equal=True, serial_seconds=serial_seconds,
                   parallel_seconds=parallel_seconds, measured_speedup=serial_seconds/parallel_seconds,
                   bundle_sha256=digest(a), chosen_logprob_max_abs_error=ca['chosen_logprob_max_abs_error'])
        write_new(directory / 'result.json', row)
        records.append(row)
        print(f'{mode} PASSED {row}', flush=True)
    require(source_hashes() == frozen_sources, 'Source changed during acceptance: do not activate')
    receipt = {'schema': 'phase3.readout.speed.acceptance.v1', 'passed': True,
               'source_hashes': frozen_sources, 'comparisons': records,
               'no_environment_rollouts_or_editor_calls': True,
               'scope': 'Identical per-decision kernels; independent identical GPU pairs; exact ordered merge.'}
    write_new(output / 'receipt.json', receipt)
    print('ALL_PARITY_CHECKS_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decisions', type=int, default=48)
    args = parser.parse_args()
    benchmark(root=args.root, output=args.output, decisions=args.decisions)
