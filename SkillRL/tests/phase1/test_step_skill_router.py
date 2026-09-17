from agent_system.memory import FrozenStepSkillRouter, SkillsOnlyMemory

FULL_BANK = "memory_data/alfworld/claude_style_skills.json"


def clean_bundle():
    memory = SkillsOnlyMemory(FULL_BANK, task_specific_top_k=None)
    return memory, memory.retrieve("put a clean apple in the fridge", top_k=12)


def test_router_is_deterministic_and_injects_exactly_one_skill():
    _, bundle = clean_bundle()
    router = FrozenStepSkillRouter()
    kwargs = {
        "task_description": "put a clean apple in the fridge",
        "current_observation": "On countertop 1, you see an apple 1.",
        "admissible_actions": ["take apple 1 from countertop 1", "go to sinkbasin 1"],
        "history": [],
        "step_index": 1,
    }
    first = router.route(bundle, **kwargs)
    second = router.route(bundle, **kwargs)
    assert first["selected_skill_id"] == second["selected_skill_id"] == "gen_002"
    assert first["injected_skill_ids"] == ["gen_002"]
    assert len(first["general_skills"] + first["task_specific_skills"]) == 1
    assert len(first["candidate_skill_ids"]) == 18


def test_router_changes_skill_across_observable_task_phases():
    _, bundle = clean_bundle()
    router = FrozenStepSkillRouter()
    task = "put a clean apple in the fridge"
    states = [
        ("You are in the middle of a room.", ["go to countertop 1"], []),
        (
            "On countertop 1, you see an apple 1.",
            ["take apple 1 from countertop 1"],
            [{"observation": "room", "action": "go to countertop 1"}],
        ),
        (
            "You pick up the apple 1.",
            ["clean apple 1 with sinkbasin 1", "put apple 1 in/on countertop 1"],
            [{"observation": "countertop", "action": "take apple 1 from countertop 1"}],
        ),
        (
            "You clean the apple 1.",
            ["put apple 1 in/on fridge 1"],
            [{"observation": "sink", "action": "clean apple 1 with sinkbasin 1"}],
        ),
    ]
    selected = [
        router.route(
            bundle,
            task_description=task,
            current_observation=observation,
            admissible_actions=actions,
            history=history,
            step_index=index,
        )["selected_skill_id"]
        for index, (observation, actions, history) in enumerate(states)
    ]
    assert selected == ["cle_006", "gen_002", "cle_003", "cle_005"]


def test_minus_skill_preserves_retrieval_provenance_and_forces_fallback():
    memory, _ = clean_bundle()
    router = FrozenStepSkillRouter()
    with memory.temporarily_disabled("gen_002"):
        bundle = memory.retrieve("put a clean apple in the fridge", top_k=12)
    routed = router.route(
        bundle,
        task_description="put a clean apple in the fridge",
        current_observation="On countertop 1, you see an apple 1.",
        admissible_actions=["take apple 1 from countertop 1"],
        history=[],
        step_index=1,
    )
    assert "gen_002" in routed["retrieved_skill_ids"]
    assert "gen_002" not in routed["candidate_skill_ids"]
    assert routed["selected_skill_id"] != "gen_002"


def test_upstream_pick_two_skill_is_available_without_separate_category():
    memory = SkillsOnlyMemory(FULL_BANK, task_specific_top_k=None)
    bundle = memory.retrieve("put two mugs in cabinet 1", top_k=12)
    assert bundle["task_type"] == "pick_two"
    assert {item["skill_id"] for item in bundle["task_specific_skills"]} == {
        "pic_001", "pic_002", "pic_003", "pic_004", "pic_005"
    }
