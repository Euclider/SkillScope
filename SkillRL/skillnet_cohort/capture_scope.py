"""Versioned endpoint evidence selection; never changes the GRPO update batch."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .common import write_new_json


WINDOW_START = 'window_start_old_only_v1'


def full_capture(settings, update):
    scope = settings.get('capture_scope', 'every_update_old_new_v1')
    if scope == 'every_update_old_new_v1':
        return True
    if scope != WINDOW_START:
        raise ValueError('Unknown capture scope; no implicit evidence reduction')
    updates = list(settings.get('capture_updates', []))
    if not updates or any(type(u) is not int or u < 1 for u in updates):
        raise ValueError('Explicit window-start batch updates required')
    return int(update) in updates


def capture_post(settings):
    # Validate even when the selected start batch is not the current update.
    full_capture(settings, 1)
    return settings.get('capture_scope') != WINDOW_START


def archive_rollout_summary(root, update, is_train, infos, rewards, lengths, trajectory_ids):
    """Small per-trajectory accounting, not a replacement for start-state evidence."""
    from phase1.archive import jsonable
    rows = []
    for i, trajectory in enumerate(trajectory_ids):
        active = infos[i][:int(lengths[i])]
        skills = Counter(info['selected_skill_id'] for info in active if info.get('selected_skill_id'))
        rows.append({'trajectory_id': str(trajectory), 'episode_return': jsonable(rewards[i]),
                     'environment_steps': int(lengths[i]), 'skill_selection_counts': dict(skills),
                     'game_id': next((info['extra.gamefile'] for info in active if info.get('extra.gamefile')), None)})
    split = 'train' if is_train else 'monitor'
    write_new_json(Path(root) / 'rollout_summaries' / f'u{update:04d}-{split}.json', {
        'schema_version': 'skillnet.phase12.compact_rollout.v1', 'update': int(update),
        'split': split, 'trajectories': rows, 'raw_states_archived': False})
