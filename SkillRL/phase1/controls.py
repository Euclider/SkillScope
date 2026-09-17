from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def shuffle_trajectory_reward_tensor(token_rewards, response_mask, group_ids: Sequence, trajectory_ids: Sequence, seed: int):
    """Shuffle episode returns within GRPO groups while keeping rollout states fixed."""
    shuffled = token_rewards.clone()
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row, (group_id, trajectory_id) in enumerate(zip(group_ids, trajectory_ids)):
        groups[str(group_id)][str(trajectory_id)].append(row)

    rng = np.random.default_rng(seed)
    mapping = []
    for group_id, trajectories in groups.items():
        ids = sorted(trajectories)
        if len(ids) < 2:
            continue
        totals = {
            trajectory_id: float(token_rewards[rows].sum().item())
            for trajectory_id, rows in trajectories.items()
        }
        permuted = list(ids)
        while permuted == ids:
            rng.shuffle(permuted)
        for destination, source in zip(ids, permuted):
            rows = trajectories[destination]
            shuffled[rows] = 0
            final_row = rows[-1]
            valid_positions = response_mask[final_row].nonzero(as_tuple=False).flatten()
            if len(valid_positions) == 0:
                raise ValueError(f"Trajectory {destination} has no valid response token")
            shuffled[final_row, valid_positions[-1]] = totals[source]
            mapping.append({
                "group_id": group_id,
                "destination_trajectory_id": destination,
                "source_trajectory_id": source,
                "assigned_return": totals[source],
            })
    return shuffled, mapping


def apply_shuffled_reward_control(batch, *, seed: int):
    shuffled, mapping = shuffle_trajectory_reward_tensor(
        batch.batch["token_level_rewards"],
        batch.batch["response_mask"],
        batch.non_tensor_batch["uid"],
        batch.non_tensor_batch["traj_uid"],
        seed,
    )
    batch.batch["token_level_rewards"] = shuffled
    return batch, mapping

