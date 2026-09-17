from phase1.summarize_step_routing import summarize_trajectories


def test_step_routing_summary_counts_skills_and_checks_invariants():
    summary = summarize_trajectories([{
        "trajectory_id": "t1",
        "context_id": "clean",
        "episode_return": 10,
        "steps": [
            {
                "step_index": 0, "selected_skill_id": "cle_006",
                "injected_skill_ids": ["cle_006"],
                "candidate_skill_ids": ["cle_006", "gen_002"],
                "skill_router_version": "router-v1",
            },
            {
                "step_index": 1, "selected_skill_id": "gen_002",
                "injected_skill_ids": ["gen_002"],
                "candidate_skill_ids": ["cle_006", "gen_002"],
                "skill_router_version": "router-v1",
            },
        ],
    }])
    assert summary["success_rate"] == 1.0
    assert summary["mean_distinct_selected_skills"] == 2.0
    assert summary["skill_selection_counts"] == {"cle_006": 1, "gen_002": 1}
    assert summary["routing_invariant_violation_count"] == 0
    assert all(summary["pilot_gate"].values())
