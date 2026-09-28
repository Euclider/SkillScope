"""One no-retry GPU shard over the immutable seed404 actual U0 batch."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import time

from .common import REPO, file_hash, read_json, write_new_bytes, write_new_json
from .realized_reward import FIELDS, KEYS, assert_same_scalars, vector_signals


@contextmanager
def adapter(config):
    import torch
    from phase2 import measure as base
    from phase2.stable_direction import token_signals
    from . import first_calls_measure as original
    old_forward, old_signals = original.forward, original.token_signals
    calls = []

    def forward(model, *args):
        with torch.cuda.device(next(model.parameters()).device):
            values, hidden = base.forward(model, *args)
        return values.to('cuda:0'), hidden

    def score(*args, **kwargs):
        values = token_signals(*args, tau_c=config['signals']['tau_C'],
            epsilon=config['signals']['epsilon'], tau_delta=config['_frozen_tau_delta'], include_legacy=True)
        # Only primary live-old rows need new geometry, not matched-backend audit calls.
        if len(calls) % 2 == 0:
            values.update(vector_signals(*args, epsilon=config['signals']['epsilon']))
        calls.append(values)
        return values

    original.forward, original.token_signals = forward, score
    try:
        yield calls
    finally:
        original.forward, original.token_signals = old_forward, old_signals


def measure(output, shard):
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from phase2.protocol import controls_by_skill
    from . import first_calls_measure as original
    from .numerical_readout import ordered_decisions, write_new_tensor
    from .realized_reward_run import binding, gpu_pair, storage_gate

    output = Path(output).resolve()
    plan = binding(output)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_pair(shard):
        raise PermissionError('Wrong registered GPU pair')
    if (output/f'attempt-shard-{shard}.json').exists():
        raise FileExistsError('No automatic retry of a started shard')
    storage_gate(plan)
    torch.set_num_threads(1)
    for gpu in range(2):
        if torch.cuda.mem_get_info(gpu)[0] < plan['minimum_free_gpu_mib']*2**20:
            raise MemoryError('Registered free-memory floor failed; never evict another process')
    write_new_json(output/f'attempt-shard-{shard}.json', {'pid': os.getpid(), 'shard': shard,
        'started_unix': time.time(), 'plan_sha256': file_hash(output/'plan.json'), 'automatic_retry': False})
    root, source = Path(plan['trajectory_root']), Path(plan['source'])
    config = read_json(root/'protocol.json')
    config['_frozen_tau_delta'] = read_json(root/'signals/calibration.json')['tau_delta']
    batch = torch.load(root/'batches/u0001/training_batch.pt', map_location='cpu', weights_only=False)
    controls = controls_by_skill(config, REPO)
    rows, seen = [], set()
    for row, meta in enumerate(batch['metadata']):
        if meta['decision_id'] in seen:
            continue
        seen.add(meta['decision_id'])
        if meta['info'].get('selected_skill_id') in controls:
            rows.append((row, meta))
    selected = rows[shard::8]
    recorded = pd.read_parquet(source/'token_signals.parquet')
    by_id = {key: frame for key, frame in recorded.groupby('decision_id', sort=False)}
    if set(by_id) != {m['decision_id'] for _, m in rows}:
        raise ValueError('Actual batch support differs from the frozen numerical report')
    tokenizer = AutoTokenizer.from_pretrained(root/'models/u0000', local_files_only=True)
    models = [AutoModelForCausalLM.from_pretrained(root/'models'/endpoint,
        dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to(f'cuda:{rank}').eval()
        for rank, endpoint in enumerate(('u0000', 'u0005'))]
    frames, checks = {}, []
    expected_old = {r['row_index']: r['sha256'] for r in plan['old_tensor_files']}
    for count, (index, (row, meta)) in enumerate(ordered_decisions(selected, by_id)):
        old_hash = file_hash(root/'old_logprobs/u0001'/f'row-{row:06d}.pt')
        if old_hash != expected_old[row]:
            raise ValueError('Archived OLD probability row changed before scoring')
        with adapter(config) as calls:
            fresh, _, witness, _ = original.score_decision((root, *models), tokenizer,
                batch['tensors'], row, meta, controls[meta['info']['selected_skill_id']],
                float(batch['meta_info']['temperature']), repeat=False)
        if len(calls) != 4:
            raise ValueError('Both primary and matched-backend checks required for two controls')
        assert_same_scalars(fresh, by_id[meta['decision_id']])
        if not (fresh.P_centering_abs_error <= 1e-9+1e-11*fresh.P_int.abs()).all():
            raise ValueError('Existing stable-P identity failed')
        frames[index] = fresh[KEYS+list(FIELDS)].copy()
        checks.append({'decision_id': meta['decision_id'], 'row_index': row,
            'old_tensor_sha256': old_hash,
            'tokens': len(fresh), 'stable_scalars_exact': True})
        # Small audit samples, not another full-vocabulary archive.
        if index == 0:
            witness = {k: (v[:16].clone() if isinstance(v, torch.Tensor) else v)
                       for k, v in witness.items()}
            write_new_tensor(output/f'witness-shard-{shard}.pt', witness)
        print(f'seed404 realized shard{shard} {count+1}/{len(selected)} original_row={row} '
              f'tokens={len(fresh)} stable_exact=True', flush=True)
    result = pd.concat([frames[i] for i in range(len(selected))], ignore_index=True)
    write_new_bytes(output/f'tokens-shard-{shard}.parquet', result.to_parquet(index=False))
    write_new_json(output/f'input-audit-shard-{shard}.json', checks)
    write_new_json(output/f'shard-{shard}.json', {'seed': 404, 'shard': shard, 'decisions': len(selected),
        'tokens': len(result), 'tokens_sha256': file_hash(output/f'tokens-shard-{shard}.parquet'),
        'input_audit_sha256': file_hash(output/f'input-audit-shard-{shard}.json'),
        'witness_sha256': file_hash(output/f'witness-shard-{shard}.pt'),
        'all_stable_scalar_signals_exact': True, 'target_labels_read': False,
        'finished_unix': time.time(), 'peak_gpu_allocated_bytes': [torch.cuda.max_memory_allocated(i) for i in range(2)]})
