"""Snapshot and split-boundary checks for the LogicBench Phase 1/2 pilot."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from agent_system.memory.frozen_skill_bank import FrozenBankError, FrozenSkillBankMemory
from agent_system.memory.sra_logicbench_bank import (
    EXPECTED_IDS,
    SRA_LOGICBENCH19_COMMIT,
    SRA_LOGICBENCH19_MANIFEST,
    SRA_LOGICBENCH19_MANIFEST_SHA256,
    FrozenSraLogicBenchBank,
    load_sra_logicbench19,
)

ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "data/logicbench/sra19"


def test_frozen_bank_identity_and_unmodified_content():
    bank = load_sra_logicbench19()
    setting = json.loads((SRA_LOGICBENCH19_MANIFEST.parent / "setting.json").read_bytes())
    assert bank.manifest_sha256 == SRA_LOGICBENCH19_MANIFEST_SHA256
    assert bank.skill_ids == EXPECTED_IDS
    assert len(bank) == 19
    assert bank.verification_summary()["source_commit"] == SRA_LOGICBENCH19_COMMIT
    assert bank.verification_summary()["router_implemented"] is True
    assert bank.verification_summary()["evaluation_instances_in_bank"] is False
    assert setting["runnable_experiment"] is False
    assert setting["bank"]["manifest_sha256"] == bank.manifest_sha256
    assert setting["bank"]["skills_json_sha256"] == bank.content_sha256
    source = json.loads((SRA_LOGICBENCH19_MANIFEST.parent / "skills.json").read_bytes())
    assert all(set(row) == {"skill_id", "name", "description", "content"} for row in source)
    for row in source:
        skill = bank.get(row["skill_id"])
        assert skill.content == row["content"]
        assert skill.payload == f"### Frozen Skill: {row['skill_id']}\n\n{row['content']}"
    assert bank.router_catalog() == [
        {key: row[key] for key in ("skill_id", "name", "description")} for row in source
    ]


def test_memory_only_injects_selected_skill_and_remains_read_only():
    memory = FrozenSkillBankMemory(load_sra_logicbench19())
    candidates = memory.retrieve("unknown question")
    assert candidates["candidate_skill_ids"] == list(EXPECTED_IDS)
    assert candidates["selected_skill_id"] is None
    assert candidates["injected_skill_ids"] == []
    selected = memory.selected_bundle("logicbench_000")
    assert selected["injected_skill_ids"] == ["logicbench_000"]
    assert memory.format_for_prompt(selected) == memory.bank.get("logicbench_000").payload
    assert memory.format_for_prompt(memory.selected_bundle(None)) == ""
    with pytest.raises(FrozenBankError, match="at most one skill"):
        memory.format_for_prompt(candidates)
    with pytest.raises(FrozenBankError, match="cannot store"):
        memory.store({"answer": "yes"})


def test_modified_source_or_manifest_is_rejected(tmp_path):
    copied = tmp_path / "bank"
    shutil.copytree(SRA_LOGICBENCH19_MANIFEST.parent, copied)
    skills = copied / "skills.json"
    skills.write_bytes(skills.read_bytes() + b" ")
    with pytest.raises(FrozenBankError, match="skills.json SHA-256 mismatch"):
        FrozenSraLogicBenchBank(copied / "manifest.json")
    shutil.copyfile(SRA_LOGICBENCH19_MANIFEST.parent / "skills.json", skills)
    manifest = copied / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(FrozenBankError, match="manifest SHA-256 mismatch"):
        FrozenSraLogicBenchBank(manifest)


def test_eval_asset_is_separate_and_pinned():
    manifest = json.loads((EVAL_ROOT / "eval_manifest.json").read_bytes())
    setting = json.loads((SRA_LOGICBENCH19_MANIFEST.parent / "setting.json").read_bytes())
    eval_bytes = (EVAL_ROOT / manifest["file"]).read_bytes()
    assert hashlib.sha256(eval_bytes).hexdigest() == manifest["sha256"]
    assert manifest["sha256"] == setting["evaluation_data"]["eval_sha256"]
    assert manifest["may_train_policy_on_this_file"] is False
    assert manifest["may_supply_router_with_gold_annotation"] is False
    assert manifest["may_use_answers_as_readout_inputs"] is False
    instances = json.loads(eval_bytes)
    assert len(instances) == manifest["instances"] == 760
    assert len({row["instance_id"] for row in instances}) == 760
    counts = Counter(row["skill_annotations"][0] for row in instances)
    assert set(counts) == set(EXPECTED_IDS)
    assert set(counts.values()) == {40}
    assert all(len(row["skill_annotations"]) == 1 for row in instances)
    assert Counter(row["eval_data"]["task_type"] for row in instances) == manifest["task_type_counts"]
    for row in instances:
        answer, kind = row["eval_data"]["answer"], row["eval_data"]["task_type"]
        assert answer in ({"yes", "no"} if kind == "BQA" else {f"choice_{i}" for i in range(1, 5)})
