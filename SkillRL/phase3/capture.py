"""Compact observed episodes plus one actual direction batch per fixed window."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from phase1.archive import jsonable
from skillnet_cohort.common import file_hash, write_new_bytes
from .common import digest, require, write_new


def attach_decisions(batch, infos, *, config, update, step):
    metadata = []
    for index, info in enumerate(infos):
        trajectory = str(batch.non_tensor_batch['traj_uid'][index])
        metadata.append(json.dumps({'decision_id': f'{config.branch_id}:u{update}:{trajectory}:s{step}',
            'global_update': update, 'environment_step': step, 'trajectory_id': trajectory,
            'group_id': str(batch.non_tensor_batch['uid'][index]), 'info': jsonable(info)}, sort_keys=True))
    batch.non_tensor_batch['phase3_metadata'] = np.asarray(metadata, dtype=object)


def archive_batch(batch, *, update, config):
    # No full-vocabulary probabilities, activations, gradients or utility gold.
    # The failure-driven arm also captures the batch for a passive, non-decision
    # action-bias audit. Its edit selector never consumes the resulting readout.
    if update != config.phase3.segment_start + 1:
        return
    import torch
    fields = ('input_ids', 'attention_mask', 'position_ids', 'responses', 'advantages', 'old_log_probs')
    tensors = {key: batch.batch[key].detach().cpu().clone() for key in fields}
    length = tensors['responses'].shape[-1]
    source = 'loss_mask' if config.actor_rollout_ref.rollout.multi_turn.enable else 'attention_mask'
    tensors['actual_loss_mask'] = batch.batch[source][:, -length:].detach().cpu().clone()
    require(tensors['actual_loss_mask'].shape == tensors['advantages'].shape, 'Actual advantage/mask mismatch')
    require(torch.isfinite(tensors['advantages'][tensors['actual_loss_mask'].bool()]).all().item(), 'Non-finite advantages')
    metadata = [json.loads(str(value)) for value in batch.non_tensor_batch['phase3_metadata']]
    for item in metadata:
        require(item['global_update'] == update and item['info']['bank_sha256'] == config.phase3.bank_sha256,
                'Foreign direction batch')
    value = {'schema_version': 'skillrl.phase3.direction_batch.v1', 'tensors': tensors, 'metadata': metadata,
             'temperature': float(batch.meta_info['temperature']), 'bank_sha256': config.phase3.bank_sha256,
             'branch_id': config.phase3.branch_id, 'global_update': update}
    target = Path(config.phase3.root) / 'direction_batches' / f'u{update:04d}.pt'
    require(not target.exists(), 'Refusing to overwrite an already captured direction batch')
    buffer = io.BytesIO()
    torch.save(value, buffer)
    write_new_bytes(target, buffer.getvalue())
    write_new(target.with_suffix('.json'), {'sha256': file_hash(target), 'row_count': len(metadata),
        'loss_tokens': int(tensors['actual_loss_mask'].sum()), 'bank_sha256': config.phase3.bank_sha256,
        'capture_point': 'actual GRPO advantages before optimizer update', 'full_vocab_saved': False})


def archive_episodes(*, config, is_train, update, batches, infos, trajectory_ids, episode_rewards):
    root = Path(config.root)
    for index, (batch_steps, info_steps) in enumerate(zip(batches, infos)):
        steps, active_infos = [], []
        for step, (batch, info) in enumerate(zip(batch_steps, info_steps)):
            if not bool(jsonable(batch['active_masks'])):
                continue
            require(info['bank_sha256'] == config.bank_sha256, 'Foreign rollout bank')
            mask = jsonable(batch['attention_mask'])
            response_length = len(jsonable(batch['responses']))
            steps.append({**{key: jsonable(info.get(key)) for key in (
                'observation', 'next_observation', 'raw_model_output', 'admissible_actions',
                'selected_skill_id', 'skill_version_sha256', 'skill_router_api',
                'candidate_skill_ids', 'skill_router_version', 'skill_router_scores')},
                'step_index': step, 'action': info['projected_action'],
                'is_action_valid': bool(jsonable(info['is_action_valid'])),
                'prompt_tokens': sum(mask[:-response_length]), 'completion_tokens': sum(mask[-response_length:]),
                'reward': jsonable(batch['rewards'])})
            active_infos.append(info)
        require(active_infos, 'Empty rollout trajectory')
        game = Path(active_infos[0]['extra.gamefile']).resolve()
        data_root = Path(config.data_root).resolve()
        require(game.is_relative_to(data_root), 'Foreign rollout game')
        game_id = game.relative_to(data_root).as_posix()
        trajectory = str(trajectory_ids[index])
        record = {'trajectory_id': trajectory, 'branch_id': config.branch_id, 'bank_sha256': config.bank_sha256,
            'global_update': update, 'split': 'train' if is_train else 'valid_seen', 'game_id': game_id,
            'task': active_infos[0]['task_description'], 'success': bool(active_infos[-1]['won']),
            'episode_return': jsonable(episode_rewards[index]), 'steps': steps}
        write_new(root / 'episodes' / f'u{update:04d}' / record['split'] / f'{trajectory}.json', record)


def archive_metrics(config, update, metrics):
    write_new(Path(config.root) / 'metrics' / f'u{update:04d}.json',
              {'global_update': update, 'branch_id': config.branch_id, 'bank_sha256': config.bank_sha256,
               'metrics': jsonable(metrics)})
