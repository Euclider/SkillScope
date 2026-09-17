from agent_system.memory.skills_only_memory import SkillsOnlyMemory


def test_context_manager_restores_mask(skill_bank):
    memory = SkillsOnlyMemory(str(skill_bank))
    memory.set_disabled_skill_ids({"gen_001"})
    with memory.temporarily_disabled("cle_001"):
        assert memory.disabled_skill_ids == {"gen_001", "cle_001"}
    assert memory.disabled_skill_ids == {"gen_001"}


def test_minus_skill_removes_only_target(skill_bank):
    memory = SkillsOnlyMemory(str(skill_bank))
    with memory.temporarily_disabled("cle_001"):
        result = memory.retrieve("clean the plate", top_k=1)
    assert result["retrieved_skill_ids"] == ["gen_001", "cle_001", "err_001"]
    assert result["injected_skill_ids"] == ["gen_001", "err_001"]
    assert "Clean" not in memory.format_for_prompt(result)
    assert "Explore" in memory.format_for_prompt(result)


def test_no_skill_injects_no_memory_text(skill_bank):
    memory = SkillsOnlyMemory(str(skill_bank))
    with memory.temporarily_disabled_all():
        result = memory.retrieve("clean the plate", top_k=1)
    assert result["injected_skill_ids"] == []
    assert memory.format_for_prompt(result) == "No relevant skills found for this task."


def test_pick_two_routes_independently(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text("""{
      "general_skills": [],
      "task_specific_skills": {
        "pick_and_place": [{"skill_id": "one", "title": "One", "principle": "one", "when_to_apply": "one"}],
        "pick_two": [{"skill_id": "two", "title": "Two", "principle": "two", "when_to_apply": "two"}]
      },
      "common_mistakes": []
    }""")
    memory = SkillsOnlyMemory(str(bank), task_specific_top_k=1)
    result = memory.retrieve("find two mugs and put them in cabinet 1", top_k=1)
    assert result["task_type"] == "pick_two"
    assert result["injected_skill_ids"] == ["two"]


def test_alfworld_annotation_synonyms_route_to_canonical_context():
    memory = SkillsOnlyMemory("phase1/config/frozen_alfworld_skills.json", task_specific_top_k=1)
    examples = {
        "Put a cooked tomato on the counter.": "heat",
        "Place a chilled potato in a microwave.": "cool",
        "Put a cleaned rag in a drawer.": "clean",
        "Hold a bowl while turning a lamp on.": "look_at_obj_in_light",
    }
    for task, expected in examples.items():
        assert memory.retrieve(task, top_k=1)["task_type"] == expected
