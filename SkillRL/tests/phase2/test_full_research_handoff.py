"""The public research handoff must include records without runtime debris."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _file(root: Path, name: str, content: bytes = b"fixture") -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_inventory_includes_research_records_but_not_raw_runtime(tmp_path):
    project = tmp_path / "project"
    included = {
        "README.md", "2026-09-28-results.md", ".gitignore", "PROJECT_SHA256SUMS",
        "deploy/5090/start.sh", "SkillRL/phase2/direction.py",
        "SkillRL/docs/experiments/protocol.json", "SkillRL/figs/overview.png",
        "SkillRL/agent_system/environments/layout.npy",
        "SkillRL/artifacts/phase12/seed/reports/ranking.csv",
        "SkillRL/artifacts/metrics/seed/summary.json",
        "SkillRL/artifacts/phase12/seed/skill_scores.csv",
        "SkillRL/artifacts/phase12/seed/analysis.md",
    }
    excluded = {
        "SkillRL/artifacts/phase12/seed/trajectories/turn.jsonl",
        "SkillRL/artifacts/phase12/seed/router.sqlite3",
        "SkillRL/artifacts/phase12/seed/weights.pt",
        "SkillRL/artifacts/phase12/seed/progress.lock",
        "SkillRL/artifacts/phase12/seed/sign_null_draws.csv",
        "SkillRL/docs/experiments/datasets/train.parquet",
        "SkillRL/tests/e2e/model.safetensors", "SkillRL/.env",
        "SkillRL/__pycache__/direction.pyc",
        "SkillRL/agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/chromedriver",
    }
    for name in included | excluded:
        _file(project, name)

    handoff = importlib.import_module("scripts.package_full_research_handoff")
    actual = {name for _, name in handoff.inventory(project)}
    assert actual == included


def test_package_hashes_every_included_file_and_refuses_overwrite(tmp_path):
    project = tmp_path / "project"
    _file(project, "README.md", b"research report\n")
    _file(project, "SkillRL/phase2/direction.py", b"VALUE = 1\n")
    _file(project, "SkillRL/artifacts/phase12/reports/summary.csv", b"score\n1\n")
    _file(project, "SkillRL/artifacts/phase12/models/u5.pt", b"model")
    handoff = importlib.import_module("scripts.package_full_research_handoff")
    output, archive = tmp_path / "staged", tmp_path / "handoff.tar.gz"

    result = handoff.make_package(project, output, archive)
    manifest = json.loads((output / "RELEASE_MANIFEST.json").read_text())
    assert result["source_files"] == 3
    assert {row["path"] for row in manifest["files"]} == {
        "README.md", "SkillRL/phase2/direction.py",
        "SkillRL/artifacts/phase12/reports/summary.csv",
    }
    assert not (output / "SkillRL/artifacts/phase12/models/u5.pt").exists()
    assert archive.is_file()
    with pytest.raises(ValueError, match="new package paths"):
        handoff.make_package(project, output, archive)


def test_package_fails_closed_on_credentials_without_printing_them(tmp_path):
    project = tmp_path / "project"
    marker = "sk-" + "z" * 32
    _file(project, "README.md", marker.encode())
    handoff = importlib.import_module("scripts.package_full_research_handoff")

    with pytest.raises(ValueError) as exc:
        handoff.make_package(project, tmp_path / "staged", tmp_path / "handoff.tar.gz")
    assert marker not in str(exc.value)


def test_known_vendor_doc_is_redacted_only_in_public_copy(tmp_path):
    project = tmp_path / "project"
    marker = "sk-" + "z" * 32
    name = "SkillRL/agent_system/environments/env_package/webshop/webshop/README_INSTALL_ARM-MAC.md"
    _file(project, name, ("Install with " + marker + "\n").encode())
    handoff = importlib.import_module("scripts.package_full_research_handoff")
    output = tmp_path / "staged"

    handoff.make_package(project, output, tmp_path / "handoff.tar.gz")
    published = (output / name).read_text()
    manifest = json.loads((output / "RELEASE_MANIFEST.json").read_text())
    assert marker not in published
    assert "[REDACTED CREDENTIAL]" in published
    assert marker in (project / name).read_text()
    assert manifest["redacted_paths"] == [name]


def test_publisher_uses_verified_package_and_detects_changed_source(tmp_path):
    project = tmp_path / "project"
    _file(project, "README.md", b"first version\n")
    handoff = importlib.import_module("scripts.package_full_research_handoff")
    output = tmp_path / "staged"
    handoff.make_package(project, output, tmp_path / "handoff.tar.gz")
    publisher = importlib.import_module("scripts.publish_research_handoff")

    entries = publisher.verified_entries(project, output)
    assert entries["README.md"] == b"first version\n"
    assert "FULL_RESEARCH_RELEASE_MANIFEST.json" in entries
    _file(project, "README.md", b"changed after package\n")
    with pytest.raises(ValueError, match="changed since package creation"):
        publisher.verified_entries(project, output)
