import json

from phase1.archive import archive_rollout_batch


def test_archive_counts_distinct_injected_skills(tmp_path):
    batch = [[
        {"active_masks": True, "rewards": 0, "attention_mask": [1, 1], "responses": [1]},
        {"active_masks": True, "rewards": 10, "attention_mask": [1], "responses": [1, 2]},
    ]]
    infos = [[
        {
            "retrieved_skill_ids": ["a", "b"], "candidate_skill_ids": ["a", "b"],
            "injected_skill_ids": ["a"], "selected_skill_id": "a",
            "skill_router_version": "router-v1", "is_action_valid": True,
        },
        {
            "retrieved_skill_ids": ["a", "b"], "candidate_skill_ids": ["a", "b"],
            "injected_skill_ids": ["b"], "selected_skill_id": "b",
            "skill_router_version": "router-v1", "is_action_valid": True,
        },
    ]]
    assert archive_rollout_batch(
        output_dir=tmp_path, run_id="run", split="train", global_step=1,
        total_batch_list=batch, total_infos=infos, episode_rewards=[10],
        episode_lengths=[2], trajectory_ids=["trajectory"],
    ) == 1
    data = json.loads((tmp_path / "trajectories/run/trajectory.json").read_text())
    assert data["unique_skill_count"] == 2
    assert data["injected_skill_ids"] == ["a", "b"]
    assert data["schema_version"] == "phase1.trajectory.v2"
    assert data["skill_selection_counts"] == {"a": 1, "b": 1}
    assert data["steps"][0]["candidate_skill_ids"] == ["a", "b"]
