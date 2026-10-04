"""Independent GPU pairs; exact ordered merge of compact decision scalars."""
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import time

from .bank import Bank
from .common import digest, require, strict_json, write_new
from .readout import CompactReadout, WindowIdentity

SOURCES = ('phase3/predict.py', 'phase3/parallel_predict.py', 'phase3/readout.py',
           'phase3/fast_direction.py', 'phase2/stable_direction.py',
           'skillnet_cohort/reward_variants.py', 'phase2/measure.py', 'skillnet_cohort/assets.py')


def partition(n, index, count):
    require(type(count) is int and type(index) is int and 1 <= count <= n
            and 0 <= index < count, 'Invalid or empty prediction shard')
    return n * index // count, n * (index + 1) // count


def source_hashes():
    from skillnet_cohort.common import file_hash
    repo = Path(__file__).resolve().parents[1]
    return {name: file_hash(repo / name) for name in SOURCES}


def maybe_dispatch(args):
    root = args.output.resolve().parent.parent
    request_path = root.parents[1] / 'readout-speed-next-window-v1.json'
    if not request_path.exists():
        return False
    request = strict_json(request_path.read_text())
    require(request['schema'] == 'phase3.readout.speed.v1'
            and str(root) == str(Path(request['run_root']) / 'runs' / root.name),
            'Foreign readout speed request')
    identity = strict_json(args.identity.read_text())
    start = request['start_updates'].get(identity['branch_id'])
    if start is None or identity['start'] < start:
        return False
    from skillnet_cohort.common import file_hash
    receipt_path = Path(request['acceptance_receipt'])
    require(file_hash(receipt_path) == request['acceptance_sha256'], 'Changed readout acceptance')
    receipt = strict_json(receipt_path.read_text())
    require(receipt['passed'] is True and receipt['source_hashes'] == request['source_hashes'] == source_hashes(),
            'Readout speed source is not accepted')
    require(request['gpu_ids'] == list(range(8))
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3,4,5,6,7',
            'Wrong readout GPU allocation')
    run_parallel(args, request['gpu_ids'])
    return True


def run_parallel(args, gpu_ids):
    import torch
    from skillnet_cohort.common import exclusive_writer, file_hash
    require(len(gpu_ids) in (2, 4, 6, 8) and len(set(gpu_ids)) == len(gpu_ids)
            and all(type(x) is int and x >= 0 for x in gpu_ids), 'Invalid GPU pairs')
    bank = Bank.load(args.bank, args.bank_sha256)
    identity = WindowIdentity(**strict_json(args.identity.read_text()))
    require(identity.bank_sha256 == bank.manifest_sha256 and identity.branch_id == bank.branch_id,
            'Foreign parallel readout bank')
    output = Path(args.output)
    with exclusive_writer(output):
        from .predict import endpoint_forward_precision
        metadata = strict_json(args.batch.with_suffix('.json').read_text())
        require(file_hash(args.batch) == metadata['sha256'], 'Changed parallel direction batch')
        _, backend = endpoint_forward_precision(identity.start)
        expected_source = {'identity': asdict(identity), 'batch_sha256': metadata['sha256'],
            'backend': backend, 'numerical_version': 'fp64_zero_sum_readout_v1',
            'score': 'shadow_only' if identity.branch_id == 'skillrl_failure' else 'all_registered_readouts',
            'decision_use': identity.branch_id != 'skillrl_failure', 'parity_atol': args.parity_atol}
        if (output / 'complete.json').exists():
            source = strict_json((output / 'source.json').read_text())
            require(source == expected_source, 'Changed completed parallel source')
            result = strict_json((output / 'readout.json').read_text())
            complete = strict_json((output / 'complete.json').read_text())
            require(result['identity'] == strict_json(args.identity.read_text())
                    and complete['readout_sha256'] == digest(result)
                    and complete['source_sha256'] == digest(source), 'Changed completed parallel readout')
            return result
        batch = torch.load(args.batch, weights_only=False, map_location='cpu')
        seen, ids = set(), []
        for row, meta in enumerate(batch['metadata']):
            if meta['decision_id'] not in seen:
                seen.add(meta['decision_id'])
                if batch['tensors']['actual_loss_mask'][row].bool().any():
                    ids.append(meta['decision_id'])
        ids.sort()
        del batch
        count = len(gpu_ids) // 2
        slices = [partition(len(ids), i, count) for i in range(count)]
        write_new(output / 'execution.json', {'schema': 'phase3.readout.parallel.v1',
            'gpu_ids': gpu_ids, 'partition': 'contiguous_sorted_decision_ids',
            'ranges': slices, 'decision_ids_sha256': digest(ids), 'source_hashes': source_hashes(),
            'single_decision_forward_unchanged': True, 'fp64_arithmetic_unchanged': True,
            'unused_legacy_computation_omitted': True})
        processes, started = [], time.monotonic()
        try:
            for i in range(count):
                directory = output / 'shards' / f'{i:02d}'
                if (directory / 'complete.json').exists():
                    continue
                log = output / 'logs' / f'pair-{i:02d}.log'
                require(not log.exists(), 'Prior prediction worker log exists; explicit reconciliation required')
                log.parent.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, '-B', '-m', 'phase3.predict', '--bank', str(args.bank),
                    '--bank-sha256', args.bank_sha256, '--old-path', str(args.old_path),
                    '--new-path', str(args.new_path), '--batch', str(args.batch), '--output', str(directory),
                    '--calibration', str(args.calibration), '--identity', str(args.identity),
                    '--parity-atol', str(args.parity_atol), '--shard-index', str(i),
                    '--shard-count', str(count), '--stable-only']
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpu_ids[2*i:2*i+2])),
                           PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                           OPENBLAS_NUM_THREADS='1')
                with log.open('x') as stream:
                    processes.append(subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[1],
                        env=env, stdout=stream, stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in processes):
                require(all(p.poll() in (None, 0) for p in processes),
                        'Prediction worker failed; see preserved pair logs')
                time.sleep(1)
            require(all(p.returncode == 0 for p in processes), 'Prediction worker failed')
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.terminate()
            for proc in processes:
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        calibration = strict_json(args.calibration.read_text())
        aggregate = CompactReadout(identity, bank.active_versions, tau_delta=calibration['tau_delta'])
        counts, tokens, calls, parity = 0, 0, 0, 0.
        sources, control = [], None
        for i, (lo, hi) in enumerate(slices):
            directory = output / 'shards' / f'{i:02d}'
            complete = strict_json((directory / 'complete.json').read_text())
            compact = strict_json((directory / 'compact.json').read_text())
            source = strict_json((directory / 'source.json').read_text())
            controls = strict_json((directory / 'controls.json').read_text())
            require(complete['compact_sha256'] == digest(compact)
                    and complete['source_sha256'] == digest(source)
                    and compact['seen'] == ids[lo:hi] and complete['decisions'] == hi - lo
                    and complete['calibration_sha256'] == digest(calibration), 'Incomplete/changed prediction shard')
            require(source.pop('partition') == {'index': i, 'count': count}, 'Wrong shard index')
            require(source == expected_source, 'Foreign shard source')
            sources.append(source)
            if control is None:
                control = controls
            require(controls == control and source == sources[0], 'Shard protocols differ')
            aggregate.merge_state(compact)
            counts += complete['decisions']
            tokens += complete['forward_input_tokens']
            calls += complete['forward_calls']
            parity = max(parity, complete['chosen_logprob_max_abs_error'])
        require(sorted(aggregate.seen) == ids and counts == len(ids), 'Incomplete merged decisions')
        result = aggregate.bundle()
        write_new(output / 'source.json', sources[0])
        write_new(output / 'controls.json', control)
        write_new(output / 'readout.json', result)
        write_new(output / 'complete.json', {'readout_sha256': digest(result), 'source_sha256': digest(sources[0]),
            'decisions': counts, 'forward_calls': calls, 'forward_input_tokens': tokens,
            'wall_seconds': time.monotonic() - started, 'chosen_logprob_max_abs_error': parity,
            'calibration_sha256': digest(calibration), 'full_vocab_saved': False, 'utility_gold_computed': False,
            'execution_profile': 'independent-GPU-pairs-ordered-merge-v1', 'worker_pairs': count})
        return result
