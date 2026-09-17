import torch

from phase1.controls import shuffle_trajectory_reward_tensor


def test_shuffled_reward_preserves_group_return_multiset():
    rewards = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    mask = torch.ones_like(rewards, dtype=torch.bool)
    shuffled, mapping = shuffle_trajectory_reward_tensor(
        rewards, mask,
        group_ids=["g", "g", "g", "g"],
        trajectory_ids=["t1", "t1", "t2", "t2"],
        seed=7,
    )
    before = sorted([rewards[:2].sum().item(), rewards[2:].sum().item()])
    after = sorted([shuffled[:2].sum().item(), shuffled[2:].sum().item()])
    assert before == after
    assert all(item["source_trajectory_id"] != item["destination_trajectory_id"] for item in mapping)

