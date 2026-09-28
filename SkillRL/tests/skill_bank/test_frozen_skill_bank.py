"""Offline contract tests: no models, environments, upstream scripts, or API calls."""

import copy
import hashlib
import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from agent_system.memory.frozen_skill_bank import (
    RENDERER_VERSION,
    SKILLNET37_COMMIT,
    SKILLNET37_MANIFEST,
    SKILLNET37_MANIFEST_SHA256,
    FrozenBankError,
    FrozenSkillBank,
    FrozenSkillBankMemory,
    load_skillnet37,
    main,
)
from phase1.first_invocation import (
    PayloadArm,
    intervention_payload,
    selected_payload_text,
)


CONTENT_SHA256 = "a3fdd5f265f4a9926533686aef315163a6fb6b6f6ba5fc3f2153a9240dc662ab"
TASK_GOALS = [
    ("pick_and_place_simple", "put a book in a desk"),
    ("pick_two_obj_and_place", "put two apples in a bowl"),
    ("look_at_obj_in_light", "look at a book under a desklamp"),
    ("pick_heat_then_place_in_recep", "put a hot potato in a fridge"),
    ("pick_cool_then_place_in_recep", "put a cool apple on a table"),
    ("pick_clean_then_place_in_recep", "put a clean mug in a cabinet"),
]


@pytest.fixture(scope="module")
def bank():
    return load_skillnet37()


@pytest.fixture
def memory(bank):
    return FrozenSkillBankMemory(bank)


@pytest.fixture
def copied_manifest(tmp_path):
    directory = tmp_path / "skillnet37"
    shutil.copytree(SKILLNET37_MANIFEST.parent / "upstream", directory / "upstream")
    manifest = directory / "manifest.json"
    shutil.copyfile(SKILLNET37_MANIFEST, manifest)
    return manifest


def repin_for_adversarial_test(manifest_path, mutate):
    """Only pin a mutated test copy to reach checks beyond the outer digest."""
    data = json.loads(manifest_path.read_bytes())
    mutate(data)
    raw = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest_path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_snapshot_identity_and_exact_counts(bank):
    summary = bank.verification_summary()
    assert len(bank) == 37
    assert bank.bank_id == "skillnet-alfworld-37-5c472b36d2a4"
    assert bank.manifest_sha256 == SKILLNET37_MANIFEST_SHA256
    assert bank.content_sha256 == CONTENT_SHA256
    assert summary["source_commit"] == SKILLNET37_COMMIT
    assert summary["counts"] == {
        "skills": 37, "main_files": 37, "support_files": 45,
        "upstream_files": 83, "upstream_utf8_bytes": 152902,
    }
    assert summary["candidate_scope"] == "all_tasks"
    assert summary["maximum_payload_utf8_bytes"] == 7653
    assert summary["executed_source_scripts"] is False
    assert summary["router_implemented"] is False
    assert [skill.name for skill in bank.skills] == sorted(skill.name for skill in bank.skills)
    assert len(set(bank.skill_ids)) == 37


def test_bank_setting_agrees_with_loaded_snapshot_and_cannot_launch_an_experiment(bank):
    setting = json.loads((SKILLNET37_MANIFEST.parent / "setting.json").read_bytes())
    assert setting["status"] == "bank_frozen_router_pending"
    assert setting["runnable_experiment"] is False
    assert setting["bank"]["bank_id"] == bank.bank_id
    assert setting["bank"]["manifest_sha256"] == bank.manifest_sha256
    assert setting["bank"]["content_sha256"] == bank.content_sha256
    assert setting["bank"]["skill_count"] == len(bank)
    assert setting["bank"]["upstream_commit"] == SKILLNET37_COMMIT
    assert setting["bank"]["candidate_scope"] == "all_tasks"
    assert setting["bank"]["merged_with_skillrl_bank"] is False
    assert setting["adapter"]["renderer"] == RENDERER_VERSION
    assert setting["adapter"]["historical_runtime_defaults_changed"] is False
    assert setting["router"]["implementation_status"] == "not_implemented"
    assert setting["router"]["provider"] is None
    assert setting["router"]["model_snapshot"] is None
    assert setting["experiment"]["new_seed"] is None
    assert setting["experiment"]["launch_authorized_by_this_setting"] is False


@pytest.mark.parametrize("task_type,goal", TASK_GOALS, ids=[item[0] for item in TASK_GOALS])
def test_all_six_task_types_receive_the_identical_complete_pool(memory, bank, task_type, goal):
    # task_type is a label for this synthetic goal, never a bank partition.
    bundle = memory.retrieve(goal)
    assert bundle == memory.retrieve("unrecognized task with no matching words")
    assert bundle["candidate_skill_ids"] == list(bank.skill_ids)
    assert bundle["retrieved_skill_ids"] == list(bank.skill_ids)
    assert bundle["selected_skill_id"] is None
    assert bundle["injected_skill_ids"] == []
    assert bundle["task_type"] == "all_tasks"
    assert len(bundle["general_skills"]) == 37
    assert bundle["task_specific_skills"] == []
    assert bundle["mistakes_to_avoid"] == []
    assert all(set(item) == {"skill_id", "name", "description", "title"}
               for item in bundle["general_skills"])
    assert memory.retrieve(goal, top_k=37) == bundle


@pytest.mark.parametrize("top_k", [0, 1, 5, 36, 38, True, "37", 37.0])
def test_candidate_truncation_is_rejected(memory, top_k):
    with pytest.raises(FrozenBankError, match="complete candidate pool"):
        memory.retrieve("clean a mug", top_k=top_k)


@pytest.mark.parametrize("filters", [
    {"task_type": "clean"}, {"disabled_skill_ids": []}, {"observation": "text"},
])
def test_no_hidden_candidate_filter_arguments(memory, filters):
    with pytest.raises(FrozenBankError, match="candidate filters"):
        memory.retrieve("goal", **filters)


def test_full_bank_cannot_be_rendered_as_a_policy_payload(memory):
    with pytest.raises(FrozenBankError, match="full-bank injection is forbidden"):
        memory.format_for_prompt(memory.retrieve("goal"))


@pytest.mark.parametrize("skill_index", range(37))
def test_every_selected_package_keeps_all_raw_text(memory, bank, skill_index):
    skill = bank.skills[skill_index]
    bundle = memory.selected_bundle(skill.skill_id)
    payload = memory.format_for_prompt(bundle)
    assert bundle["candidate_skill_ids"] == list(bank.skill_ids)
    assert bundle["selected_skill_id"] == skill.skill_id
    assert bundle["injected_skill_ids"] == [skill.skill_id]
    assert len(bundle["general_skills"]) == 1
    assert bundle["payload_renderer"] == RENDERER_VERSION
    assert payload == selected_payload_text(memory, bundle) == skill.payload
    assert payload.startswith(f"### Frozen Skill: {skill.skill_id}\n\n")
    expected = f"### Frozen Skill: {skill.skill_id}\n\n" + skill.files[0].text
    assert skill.files[0].path == f"{skill.directory}/SKILL.md"
    for file in skill.files[1:]:
        relative = Path(file.path).relative_to(skill.directory).as_posix()
        expected += f"\n\n#### Supporting file: {relative}\n\n" + file.text
    assert payload == expected
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == bundle["payload_sha256"]
    for file in skill.files:
        assert file.content == (SKILLNET37_MANIFEST.parent / "upstream" / file.path).read_bytes()
        assert hashlib.sha256(file.content).hexdigest() == file.sha256


def test_null_selection_preserves_pool_and_renders_nothing(memory, bank):
    bundle = memory.selected_bundle(None)
    assert bundle["candidate_skill_ids"] == list(bank.skill_ids)
    assert bundle["general_skills"] == []
    assert bundle["injected_skill_ids"] == []
    assert bundle["payload_sha256"] == hashlib.sha256(b"").hexdigest()
    assert memory.format_for_prompt(bundle) == selected_payload_text(memory, bundle) == ""


@pytest.mark.parametrize("skill_id", ["missing", "", "skillnet:missing", [], 37])
def test_unknown_or_malformed_selection_fails(memory, skill_id):
    with pytest.raises(FrozenBankError, match="Unknown skill ID"):
        memory.selected_bundle(skill_id)


def test_catalog_and_bundle_edits_do_not_modify_canonical_data(memory, bank):
    skill = bank.skills[0]
    catalog = bank.router_catalog()
    catalog[0]["description"] = "changed outside bank"
    catalog.pop()
    assert bank.router_catalog()[0]["description"] == skill.description
    assert len(bank.router_catalog()) == 37
    bundle = memory.selected_bundle(skill.skill_id)
    bundle["general_skills"][0]["description"] = "changed outside bank"
    bundle["general_skills"][0]["principle"] = "replacement payload must be ignored"
    assert memory.format_for_prompt(bundle) == skill.payload
    with pytest.raises(FrozenInstanceError):
        skill.payload = "changed"
    with pytest.raises(FrozenInstanceError):
        skill.files[0].content = b"changed"


@pytest.mark.parametrize("key,value", [
    ("bank_id", "wrong-bank"),
    ("bank_manifest_sha256", "0" * 64),
    ("bank_content_sha256", "0" * 64),
    ("selected_skill_id", "wrong-id"),
    ("injected_skill_ids", []),
    ("payload_renderer", "wrong-renderer"),
    ("payload_sha256", "0" * 64),
])
def test_inconsistent_bundle_identity_is_rejected(memory, bank, key, value):
    bundle = memory.selected_bundle(bank.skill_ids[0])
    bundle[key] = value
    with pytest.raises(FrozenBankError):
        memory.format_for_prompt(bundle)


@pytest.mark.parametrize("method", [
    "store", "add_skills", "remove_skill", "save_skills", "set_disabled_skill_ids",
])
def test_mutation_methods_are_unavailable(memory, method):
    with pytest.raises(FrozenBankError):
        getattr(memory, method)(None)


def test_stateless_legacy_memory_lifecycle(memory, bank):
    assert len(memory) == 37
    assert memory.reset(batch_size=4) is None
    assert memory.fetch(0) is None
    assert memory[0] == memory[3] == memory.retrieve("")
    assert memory.bank is bank


def test_existing_payload_arms_change_only_the_selected_payload(memory, bank):
    target = bank.skills[0]
    routed = memory.selected_bundle(target.skill_id)
    before = copy.deepcopy(routed)
    expected = {
        PayloadArm.ORIGINAL: (target.payload, [target.skill_id]),
        PayloadArm.PLACEBO: ("offline unrelated placeholder", [target.skill_id]),
        PayloadArm.NULL: ("", []),
    }
    for arm in PayloadArm:
        result = intervention_payload(
            memory=memory, routed=routed, arm=arm,
            target_skill_id=target.skill_id, placebo_text="offline unrelated placeholder",
        )
        assert result == expected[arm]
        assert routed == before
        other = bank.skills[1]
        assert intervention_payload(
            memory=memory, routed=memory.selected_bundle(other.skill_id), arm=arm,
            target_skill_id=target.skill_id, placebo_text="offline unrelated placeholder",
        ) == (other.payload, [other.skill_id])
    # This tests transport only, not tokenizer-based PLACEBO length matching.
    assert bank.get(target.skill_id).payload == target.payload


def test_original_license_and_missing_final_newline_are_preserved(bank):
    license_bytes = (SKILLNET37_MANIFEST.parent / "upstream/LICENSE").read_bytes()
    assert license_bytes.startswith(b"MIT License")
    assert hashlib.sha256(license_bytes).hexdigest() == "379c3ae903e21a26b5280f83aedffbe6b98a2ecc95b7918952efa3e86c698a1b"
    regulator = bank.get("skillnet:alfworld-temperature-regulator")
    action_spec = next(file for file in regulator.files if file.path.endswith("/references/action_spec.md"))
    assert len(action_spec.content) == 565
    assert not action_spec.content.endswith(b"\n")


def test_manifest_change_fails_before_parsing(copied_manifest):
    copied_manifest.write_bytes(copied_manifest.read_bytes() + b" ")
    with pytest.raises(FrozenBankError, match="Manifest SHA-256 mismatch"):
        load_skillnet37(copied_manifest)


@pytest.mark.parametrize("expected_hash", [None, "", "a" * 63, "G" * 64, "0" * 64])
def test_caller_must_supply_correct_pinned_digest(expected_hash):
    with pytest.raises(FrozenBankError):
        FrozenSkillBank(SKILLNET37_MANIFEST, expected_manifest_sha256=expected_hash)


@pytest.mark.parametrize("target", ["main", "support", "license"])
def test_source_tampering_is_detected_at_the_same_byte_length(copied_manifest, target):
    data = json.loads(copied_manifest.read_bytes())
    if target == "main":
        relative = data["skills"][0]["main_file"]
    elif target == "support":
        relative = next(row["files"][1] for row in data["skills"] if len(row["files"]) > 1)
    else:
        relative = "LICENSE"
    path = copied_manifest.parent / "upstream" / relative
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(FrozenBankError, match="Source SHA-256 mismatch"):
        load_skillnet37(copied_manifest)


@pytest.mark.parametrize("change", ["missing", "extra", "length", "symlink"])
def test_inventory_and_file_type_changes_are_rejected(copied_manifest, change):
    root = copied_manifest.parent / "upstream"
    path = root / "LICENSE"
    if change == "missing":
        path.unlink()
    elif change == "extra":
        (root / "extra.txt").write_text("unexpected", encoding="utf-8")
    elif change == "length":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        (root / "alias").symlink_to(path)
    with pytest.raises(FrozenBankError):
        load_skillnet37(copied_manifest)


@pytest.mark.parametrize("change", [
    "schema", "renderer", "candidate_filtering", "count", "content_digest",
    "description", "payload_digest", "omitted_support", "path_traversal", "absolute_path",
])
def test_semantic_manifest_inconsistencies_fail_even_with_a_test_only_repin(copied_manifest, change):
    def mutate(data):
        if change == "schema":
            data["schema_version"] = "unsupported"
        elif change == "renderer":
            data["content_policy"]["payload_renderer"] = "unsupported"
        elif change == "candidate_filtering":
            data["content_policy"]["task_type_filtering"] = True
        elif change == "count":
            data["counts"]["skills"] = 36
        elif change == "content_digest":
            data["content_sha256"] = "0" * 64
        elif change == "description":
            data["skills"][0]["description"] = "rewritten"
        elif change == "payload_digest":
            data["skills"][0]["payload_sha256"] = "0" * 64
        elif change == "omitted_support":
            next(row for row in data["skills"] if len(row["files"]) > 1)["files"].pop()
        elif change == "path_traversal":
            data["files"][0]["path"] = "../LICENSE"
        elif change == "absolute_path":
            data["files"][0]["path"] = "/LICENSE"
    digest = repin_for_adversarial_test(copied_manifest, mutate)
    with pytest.raises(FrozenBankError):
        FrozenSkillBank(copied_manifest, expected_manifest_sha256=digest)
    with pytest.raises(FrozenBankError, match="Manifest SHA-256 mismatch"):
        load_skillnet37(copied_manifest)


def test_cli_verification_is_offline_and_json(capsys, monkeypatch):
    import socket
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("Bank verification must not use network or execute scripts")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert main([]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["counts"]["skills"] == 37
    assert data["manifest_sha256"] == SKILLNET37_MANIFEST_SHA256


def test_cli_fails_closed_for_missing_snapshot(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--manifest", str(tmp_path / "absent.json")])
    assert error.value.code == 1
    assert "verification failed" in capsys.readouterr().err
