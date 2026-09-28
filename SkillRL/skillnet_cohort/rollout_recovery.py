"""Explicit recovery of the interrupted U1 batch, before ANY optimizer update.

Never samples a replacement trajectory. Archived response token IDs are used
verbatim, while prompts are reconstructed by the unchanged tokenizer/template.
All surviving live OLD rows must match the reconstructed balanced row layout.
Missing vLLM sampling logprobs are diagnostic-only and are NOT fabricated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path

from .common import file_hash, read_json, write_new_bytes, write_new_json


def save_pre_forward(batch, root, update):
    """Durable exact batch BEFORE forwards, so another I/O failure loses no RNG/tokens."""
    import numpy as np
    import torch
    target = Path(root) / 'pre_forward_batches' / f'u{update:04d}.pt'
    buffer = io.BytesIO()
    torch.save({'tensors': {k: v.detach().cpu() for k, v in batch.batch.items()},
                'non_tensor_batch': batch.non_tensor_batch, 'meta_info': batch.meta_info,
                'numpy_state': np.random.get_state(), 'torch_rng': torch.get_rng_state()}, buffer)
    write_new_bytes(target, buffer.getvalue())


def verify_old_reuse(path, payload, capture):
    from .lossless_tensor import load, bitwise_equal
    from phase1.archive import append_jsonl_idempotent
    binding = capture['reuse_verified_old']
    manifest = _verified_manifest(binding['manifest'], binding['sha256'])
    expected = {item['row']: item['sha256'] for item in manifest['matched_original_old_rows']}
    row = payload['row_index']
    if (row not in expected or file_hash(path) != expected[row]
            or not bitwise_equal(load(path), payload)):
        raise ValueError(f'Recovered OLD row differs from retained live evidence: row {row}')
    append_jsonl_idempotent(Path(capture['progress_root']) / f'reused-old-rank{payload["rank"]}.jsonl',
        [{'row_index': row, 'sha256': expected[row], 'bitwise_equal_to_new_native_forward': True}], ('row_index',))


from functools import lru_cache


@lru_cache(maxsize=4)
def _verified_manifest(path, expected):
    if file_hash(path) != expected:
        raise ValueError('Recovery manifest changed')
    return read_json(path)


def verify_binding(binding, root, preparation):
    root = Path(root).resolve()
    manifest = _verified_manifest(binding['manifest'], binding['sha256'])
    if (Path(manifest['source_root']).resolve() != root or manifest['optimizer_steps_before_failure'] != 0
            or manifest['trajectory_count'] != 128 or manifest['group_count'] != 16
            or file_hash(root/'launch.json') != binding['original_launch_sha256']
            or file_hash(root/'stopped.json') != binding['original_stopped_sha256']
            or read_json(root/'launch.json')['preparation_sha256'] != file_hash(preparation)
            or not any(Path(binding['attempt_dir']).resolve() == root.parent/name/'seed-404'
                       for name in ('recovery-v1', 'recovery-v2'))):
        raise PermissionError('Recovery does not bind this failed pre-optimizer run')
    for name in ('optimizer_steps', 'checkpoints', 'batches', 'metrics'):
        if (root/name).exists():
            raise PermissionError('Never repeat any completed optimizer work')
    if file_hash(root/'segments/u0000-u0005.json') != manifest['source_config_sha256']:
        raise ValueError('Original segment configuration changed')
    for source in manifest['source_journal']:
        if file_hash(source['path']) != source['sha256']:
            raise ValueError('Original rollout journal changed')
    if file_hash(root/'rollout_summaries/u0001-train.json') != manifest['source_summary_sha256']:
        raise ValueError('Original rollout summary changed')
    return manifest


def records(root):
    root = Path(root)
    if any((root / name).exists() for name in ('optimizer_steps', 'checkpoints', 'batches', 'metrics')):
        raise ValueError('Recovery is limited to a pre-optimizer U1 failure')
    stopped = read_json(root / 'stopped.json')
    if stopped.get('stage') != 'train-u0000-u0005':
        raise ValueError('Not the authorized interrupted initial training block')
    steps = sorted((root / 'rollout_progress/u0001').glob('step-*.json'))
    summary = read_json(root / 'rollout_summaries/u0001-train.json')
    config = read_json(root / 'segments/u0000-u0005.json')
    if not steps or len(steps) > config['env']['max_steps']:
        raise ValueError('Missing rollout journal')
    by_id, order = {}, []
    for step, path in enumerate(steps):
        value = read_json(path)
        if value['step'] != step or value['update'] != 1:
            raise ValueError('Discontinuous initial rollout journal')
        for row in value['records']:
            tid = row['trajectory_id']
            if tid not in by_id:
                if step != 0:
                    raise ValueError('Trajectory appeared after the initial state')
                by_id[tid] = []
                order.append(tid)
            if row['environment_step'] != len(by_id[tid]) or len(row['response_token_ids']) != row['response_tokens']:
                raise ValueError('Missing states or sampled tokens')
            by_id[tid].append(row)
    if order != [r['trajectory_id'] for r in summary['trajectories']] or len(order) != 128:
        raise ValueError('Complete original 16 x 8 trajectory order is required')
    for item in summary['trajectories']:
        rows = by_id[item['trajectory_id']]
        if (len(rows) != item['environment_steps'] or sum(r['reward'] for r in rows) != item['episode_return']
                or any(r['info']['extra.gamefile'] != item['game_id'] for r in rows)):
            raise ValueError('Journal and completed trajectory summary disagree')
        if len(rows) < config['env']['max_steps'] and not rows[-1]['info']['won']:
            raise ValueError('Incomplete trajectory cannot be silently shortened')
    groups = Counter(by_id[t][0]['group_id'] for t in order)
    if len(groups) != 16 or set(groups.values()) != {8}:
        raise ValueError('Original GRPO group membership was lost')
    return config, summary, order, by_id, steps


def reconstruct(root, tokenizer):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.utils.torch_functional import tokenize_and_postprocess_data
    from verl.utils.model import compute_position_id_with_mask
    from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions
    from agent_system.multi_turn_rollout import adjust_batch
    config, summary, order, by_id, steps = records(root)
    tensors = {k: [] for k in ('prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids')}
    non_tensor = {k: [] for k in ('uid', 'traj_uid', 'phase2_decision_id', 'phase2_metadata',
        'rewards', 'active_masks', 'is_action_valid', 'episode_rewards', 'episode_lengths',
        'tool_callings', 'success_rate', 'data_source', 'index', 'anchor_obs')}
    success = np.mean([float(by_id[t][-1]['info']['won']) for t in order])
    width = int(config['actor_rollout_ref']['rollout']['response_length'])
    for index, item in enumerate(summary['trajectories']):
        for row in by_id[item['trajectory_id']]:
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': row['info']['prompt_text']}],
                tokenize=False, add_generation_prompt=True, **config['data']['apply_chat_template_kwargs'])
            ids, mask = tokenize_and_postprocess_data(prompt=prompt, tokenizer=tokenizer,
                max_length=config['data']['max_prompt_length'], pad_token_id=tokenizer.pad_token_id,
                left_pad=True, truncation='error')
            positions = compute_position_id_with_mask(mask)[0]
            response = torch.full((width,), tokenizer.pad_token_id, dtype=ids.dtype)
            n = row['response_tokens']
            if not 0 < n <= width:
                raise ValueError('Response exceeds original cap')
            response[:n] = torch.tensor(row['response_token_ids'], dtype=ids.dtype)
            response_mask = torch.zeros((width,), dtype=mask.dtype)
            response_mask[:n] = 1
            values = {'prompts': ids[0], 'responses': response,
                'input_ids': torch.cat([ids[0], response]),
                'attention_mask': torch.cat([mask[0], response_mask]),
                'position_ids': torch.cat([positions, positions[-1:] + torch.arange(1, width + 1)])}
            for key, value in values.items():
                tensors[key].append(value)
            identity = {k: row[k] for k in ('decision_id', 'global_update', 'environment_step', 'trajectory_id', 'group_id', 'info')}
            values = {'uid': row['group_id'], 'traj_uid': row['trajectory_id'],
                'phase2_decision_id': row['decision_id'], 'phase2_metadata': json.dumps(identity, ensure_ascii=False, sort_keys=True),
                'rewards': row['reward'], 'active_masks': True, 'is_action_valid': bool(row['info']['is_action_valid']),
                'episode_rewards': np.float32(item['episode_return']), 'episode_lengths': np.float32(item['environment_steps']),
                'tool_callings': np.float32(0), 'success_rate': success, 'data_source': 'alfworld', 'index': index,
                'anchor_obs': row['info']['observation']}
            for key, value in values.items():
                non_tensor[key].append(value)
    raw_rows = len(tensors['responses'])
    batch = DataProto.from_dict({k: torch.stack(v) for k, v in tensors.items()},
                                {k: np.asarray(v, dtype=object) for k, v in non_tensor.items()})
    # Only padding to the native eight-rank divisor uses driver NumPy RNG before
    # the failed OLD forward. Verify the resulting layout against live captures;
    # do not search seeds or choose another trajectory if this does not match.
    previous = np.random.get_state()
    try:
        np.random.seed(config['skillnet_cohort']['seed'])
        batch = adjust_batch(OmegaConf.create(config), batch)
        next_numpy_state = np.random.get_state()
    finally:
        np.random.set_state(previous)
    lengths = batch.batch['attention_mask'].sum(-1).tolist()
    partitions = get_seqlen_balanced_partitions(lengths, k_partitions=8, equal_size=True)
    batch.reorder(torch.tensor([i for part in partitions for i in part]))
    return batch, {'raw_rows': raw_rows, 'rows': len(batch), 'trajectory_count': len(order),
        'group_count': 16, 'response_tokens': int(batch.batch['attention_mask'][:, -width:].sum()),
        'initial_game_ids': [by_id[t][0]['info']['extra.gamefile'] for t in order],
        'source_journal': [{'path': str(p.resolve()), 'sha256': file_hash(p)} for p in steps],
        'numpy_state_after_adjust': next_numpy_state}


def build(root, output):
    import torch
    from transformers import AutoTokenizer
    from .lossless_tensor import load
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Never overwrite a recovery reconstruction')
    config = read_json(root / 'segments/u0000-u0005.json')
    tokenizer = AutoTokenizer.from_pretrained(config['alignment_run']['model_path'], local_files_only=True)
    batch, audit = reconstruct(root, tokenizer)
    width = batch.batch['responses'].shape[-1]
    paths = sorted((root / 'old_logprobs/u0001').glob('row-*.pt'))
    def check(path):
        value = load(path)
        index = int(path.stem.split('-')[1])
        mask = batch.batch['attention_mask'][index, -width:].bool()
        if (value['row_index'] != index or value['rank'] != index // (len(batch)//8)
                or not torch.equal(value['token_ids'], batch.batch['responses'][index, mask])
                or not torch.equal(value['token_positions'], mask.nonzero().flatten())):
            raise ValueError(f'Original live row layout does not match reconstruction: {path.name}')
        return {'row': index, 'sha256': file_hash(path)}
    with ThreadPoolExecutor(max_workers=8) as executor:
        matched = list(executor.map(check, paths))
    if not matched:
        raise ValueError('No live row witnesses for the reconstructed layout')
    old_rows = {r['row'] for r in matched}
    per_rank = len(batch) // 8
    prefix = 0
    while prefix < per_rank and all(rank * per_rank + prefix in old_rows for rank in range(8)):
        prefix += 1
    numpy_state = audit.pop('numpy_state_after_adjust')
    payload = {'tensors': dict(batch.batch.items()), 'non_tensor_batch': batch.non_tensor_batch,
               'meta_info': batch.meta_info, 'numpy_state_after_adjust': numpy_state}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    write_new_bytes(output / 'rollout_batch.pt', buffer.getvalue())
    write_new_json(output / 'manifest.json', {**audit,
        'schema_version': 'skillnet.phase12.pre_optimizer_recovery.v1',
        'source_root': str(root), 'batch_sha256': file_hash(output / 'rollout_batch.pt'),
        'matched_original_old_rows': matched, 'common_completed_microbatch_prefix': prefix,
        'sampling_logprobs': 'unavailable after crash; diagnostic omitted, never fabricated',
        'rng_boundary': 'vLLM and worker RNG restarted at registered seed; not bitwise uninterrupted continuation',
        'optimizer_steps_before_failure': 0, 'environment_rollouts_repeated': 0,
        'source_summary_sha256': file_hash(root/'rollout_summaries/u0001-train.json'),
        'source_config_sha256': file_hash(root/'segments/u0000-u0005.json')})
    print({'status': 'RECONSTRUCTED', 'matched_rows': len(matched), 'common_prefix': prefix,
           'rows': len(batch), 'output': str(output)}, flush=True)


def load_for_training(binding, envs):
    import numpy as np
    import torch
    from verl import DataProto
    manifest_path = Path(binding['manifest'])
    if file_hash(manifest_path) != binding['sha256']:
        raise ValueError('Recovery manifest changed')
    value = read_json(manifest_path)
    path = Path(value.get('batch_path', manifest_path.parent / 'rollout_batch.pt'))
    if file_hash(path) != value['batch_sha256']:
        raise ValueError('Recovered batch changed')
    # Advance each original TextWorld game iterator by exactly its already-used
    # first reset. No action, generation, router call or reward evaluation.
    _, _, infos = envs.envs.reset()
    games = [item['extra.gamefile'] for item in infos]
    if games != value['initial_game_ids']:
        raise ValueError('Environment iterator does not match the retained first games')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    np.random.set_state(payload['numpy_state_after_adjust'])
    print('RECOVERY: reusing complete U1 rollout, no resampling; vLLM RNG restart recorded.', flush=True)
    return DataProto.from_dict(payload['tensors'], payload['non_tensor_batch'], payload['meta_info'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    build(a.root, a.output)
