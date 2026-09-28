"""Reuse complete live OLD rows; witness dtype/values before the first optimizer.

No inferred reference-policy output and no fabricated native entropy diagnostic.
All masked OLD entries are canonical zero; only scored tokens enter the PPO loss.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
from pathlib import Path

from .common import file_hash, read_json, write_new_bytes, write_new_json


def build(source_manifest, output):
    import torch
    from .lossless_tensor import load
    source_manifest, output = Path(source_manifest).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Never overwrite a recovery cache')
    old = read_json(source_manifest)
    root = Path(old['source_root'])
    if any((root/name).exists() for name in ('optimizer_steps', 'checkpoints', 'batches', 'metrics')):
        raise ValueError('Complete-OLD reuse is only for the pre-optimizer U1 failure')
    batch_path = source_manifest.parent/'rollout_batch.pt'
    if file_hash(batch_path) != old['batch_sha256']:
        raise ValueError('Source recovered batch changed')
    payload = torch.load(batch_path, map_location='cpu', weights_only=False)
    responses = payload['tensors']['responses']
    masks = payload['tensors']['attention_mask'][:, -responses.shape[-1]:].bool()
    chosen = torch.zeros_like(responses, dtype=torch.float32)
    rows = len(responses)
    paths = sorted((root/'old_logprobs/u0001').glob('row-*.pt'))
    if [p.name for p in paths] != [f'row-{i:06d}.pt' for i in range(rows)]:
        raise ValueError('OLD capture must be complete, without missing or extra rows')
    def check(item):
        i, path = item
        value = load(path)
        if (value['row_index'] != i or value['rank'] != i//(rows//8)
                or not torch.equal(value['token_ids'], responses[i, masks[i]])
                or not torch.equal(value['token_positions'], masks[i].nonzero().flatten())
                or value['trainer_chosen_log_probs'].dtype != torch.float32
                or not torch.isfinite(value['trainer_chosen_log_probs']).all()):
            raise ValueError(f'Invalid retained OLD row {i}')
        return i, value['trainer_chosen_log_probs'], {'row': i, 'sha256': file_hash(path)}
    hashes = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, values, audit in executor.map(check, enumerate(paths)):
            chosen[i, masks[i]] = values
            hashes.append(audit)
            if (i+1) % 512 == 0:
                print(f'OLD cache verified {i+1}/{rows}', flush=True)
    stream = io.BytesIO()
    torch.save({'trainer_chosen_log_probs_fp32': chosen, 'mask': masks}, stream)
    compact = output/'chosen-old.pt'
    write_new_bytes(compact, stream.getvalue())
    manifest = {**old, 'schema_version': 'skillnet.phase12.complete_old_recovery.v2',
        'batch_path': str(batch_path), 'old_probability_cache': str(compact),
        'old_probability_cache_sha256': file_hash(compact), 'matched_original_old_rows': hashes,
        'complete_old_rows': rows, 'reuses_exact_trainer_chosen': True,
        'native_old_entropy_diagnostic': 'unavailable; omitted, not approximated',
        'masked_old_entries': 'canonical_zero_not_observed',
        'dtype_rule': 'one original row per rank witnessed by native forward before reuse',
        'prior_manifest': str(source_manifest), 'prior_manifest_sha256': file_hash(source_manifest)}
    write_new_json(output/'manifest.json', manifest)
    print({'status': 'COMPLETE_OLD_CACHE_VERIFIED', 'rows': rows, 'masked_tokens': int(masks.sum()),
           'manifest': str(output/'manifest.json')}, flush=True)


def load_old_with_witness(batch, binding, actor_group):
    import torch
    from verl import DataProto
    from .rollout_recovery import _verified_manifest
    manifest = _verified_manifest(binding['manifest'], binding['sha256'])
    path = Path(manifest['old_probability_cache'])
    if file_hash(path) != manifest['old_probability_cache_sha256'] or len(batch) != manifest['complete_old_rows']:
        raise ValueError('Complete OLD cache identity changed')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    masks = batch.batch['response_mask'].bool()
    if not torch.equal(saved['mask'], masks):
        raise ValueError('Recovered OLD response mask mismatch')
    world = actor_group.world_size
    indices = list(range(0, len(batch), len(batch)//world))
    witness = batch.select_idxs(indices)
    witness.meta_info = deepcopy(batch.meta_info)
    witness.meta_info['phase2_capture'].update(full_vocab=False, stage='old_reuse_witness')
    witness.meta_info['phase2_capture'].pop('reuse_verified_old', None)
    live = actor_group.compute_log_prob(witness).batch['old_log_probs'].cpu()
    selected = saved['trainer_chosen_log_probs_fp32'][indices]
    valid = masks[indices]
    if not torch.equal(live.float()[valid], selected[valid]):
        raise ValueError('Native OLD witness differs from retained actual trainer probabilities')
    restored = saved['trainer_chosen_log_probs_fp32'].to(live.dtype)
    if not torch.equal(restored.float()[masks], saved['trainer_chosen_log_probs_fp32'][masks]):
        raise ValueError('Native OLD dtype cannot exactly represent all retained chosen probabilities')
    write_new_json(Path(binding['attempt_dir'])/'old-reuse-witness.json', {
        'rows': len(batch), 'witness_rows': indices, 'native_dtype': str(live.dtype),
        'witness_bitwise_match': True, 'all_valid_chosen_values_exact_after_dtype_restore': True,
        'entropy_diagnostic_omitted': True, 'extra_environment_rollouts': 0})
    print(f'RECOVERY: {len(batch)} live OLD rows reused; native {live.dtype} witness PASS.', flush=True)
    return DataProto.from_dict({'old_log_probs': restored}, meta_info={'temperature': batch.meta_info.get('temperature', 1.0)})


def save_stage_outputs(batch, attempt_dir, update, stage):
    """Persist small chosen-probability outputs immediately after each forward."""
    import torch
    fields = {'old': ['old_log_probs'], 'reference': ['ref_log_prob']}[stage]
    stream = io.BytesIO()
    torch.save({k: batch.batch[k].detach().cpu() for k in fields}, stream)
    path = Path(attempt_dir)/'forward_outputs'/f'u{update:04d}-{stage}.pt'
    write_new_bytes(path, stream.getvalue())
    write_new_json(path.with_suffix('.json'), {'stage': stage, 'update': update, 'rows': len(batch),
        'sha256': file_hash(path), 'dtypes': {k: str(batch.batch[k].dtype) for k in fields}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    build(a.source_manifest, a.output)
