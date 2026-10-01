"""Offline, hash-pinned SRA-Bench LogicBench-19 bank.

The 19 source records are a subset of the upstream skill corpus. They remain
independent of the benchmark instance file and are never edited by this bank.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .frozen_skill_bank import FrozenBankError

SRA_LOGICBENCH19_COMMIT = "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f"
SRA_LOGICBENCH19_MANIFEST_SHA256 = "d361ac5cf5048d0a1627806be850a3f854cd2a8319b2c884ce44d6a6961875bf"
SRA_LOGICBENCH19_MANIFEST = (
    Path(__file__).resolve().parents[2] / "memory_data/logicbench/sra19/manifest.json"
)
SRA_LOGICBENCH19_RENDERER = "skillscope.sra_logicbench_markdown.v1"
EXPECTED_IDS = tuple(f"logicbench_{index:03d}" for index in range(19))


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class FrozenSraLogicSkill:
    skill_id: str
    name: str
    description: str
    content: str
    content_sha256: str
    payload: str
    payload_sha256: str


class FrozenSraLogicBenchBank:
    """Verify the entire 19-skill snapshot against an externally pinned manifest."""

    def __init__(
        self,
        manifest_path: str | Path = SRA_LOGICBENCH19_MANIFEST,
        *,
        expected_manifest_sha256: str = SRA_LOGICBENCH19_MANIFEST_SHA256,
    ):
        path = Path(manifest_path)
        try:
            if path.is_symlink() or not path.is_file():
                raise FrozenBankError("Missing or symlinked SRA manifest")
            manifest_bytes = path.read_bytes()
            if _digest(manifest_bytes) != expected_manifest_sha256:
                raise FrozenBankError("SRA manifest SHA-256 mismatch")
            manifest = json.loads(manifest_bytes)
            if manifest["schema_version"] != "skillscope.sra_logicbench_bank.v1":
                raise FrozenBankError("Unsupported SRA bank schema")
            if manifest["source"]["commit"] != SRA_LOGICBENCH19_COMMIT:
                raise FrozenBankError("Unexpected SRA upstream commit")
            policy = manifest["content_policy"]
            if (policy["payload_renderer"] != SRA_LOGICBENCH19_RENDERER
                or policy["source_skill_text_unchanged"] is not True
                or policy["read_only"] is not True
                or policy["task_answers_excluded"] is not True):
                raise FrozenBankError("Invalid SRA bank content policy")
            root = path.parent
            if root.is_symlink():
                raise FrozenBankError("Symlinked SRA bank directory")
            actual_files = {item.name for item in root.iterdir()}
            source_files = {"manifest.json", "skills.json", "LICENSE"}
            if actual_files not in (source_files, source_files | {"setting.json"}):
                raise FrozenBankError("SRA bank file inventory mismatch")
            for name in ("skills.json", "LICENSE"):
                if (root / name).is_symlink() or not (root / name).is_file():
                    raise FrozenBankError(f"Missing or symlinked SRA bank file: {name}")
            license_bytes = (root / "LICENSE").read_bytes()
            if _digest(license_bytes) != manifest["source"]["license_sha256"]:
                raise FrozenBankError("SRA LICENSE SHA-256 mismatch")
            skill_bytes = (root / "skills.json").read_bytes()
            if _digest(skill_bytes) != manifest["skills_json_sha256"]:
                raise FrozenBankError("SRA skills.json SHA-256 mismatch")
            rows = json.loads(skill_bytes)
            entries = manifest["skills"]
            if (not isinstance(rows, list) or not isinstance(entries, list)
                or len(rows) != len(entries) or len(rows) != 19
                or [row["skill_id"] for row in rows] != list(EXPECTED_IDS)
                or [entry["skill_id"] for entry in entries] != list(EXPECTED_IDS)
                or manifest["counts"] != {"skills": 19, "source_records": 19}):
                raise FrozenBankError("SRA skill inventory mismatch")
            skills = []
            for row, entry in zip(rows, entries):
                if (set(row) != {"skill_id", "name", "description", "content"}
                    or set(entry) != {"skill_id", "name", "description", "content_sha256", "payload_sha256", "payload_utf8_bytes"}
                    or any(not isinstance(row[key], str) or not row[key] for key in row)
                    or row["name"] != entry["name"]
                    or row["description"] != entry["description"]):
                    raise FrozenBankError("Malformed SRA skill record")
                content = row["content"]
                payload = f"### Frozen Skill: {row['skill_id']}\n\n{content}"
                if (not content.startswith("# ")
                    or _digest(content.encode("utf-8")) != entry["content_sha256"]
                    or _digest(payload.encode("utf-8")) != entry["payload_sha256"]
                    or len(payload.encode("utf-8")) != entry["payload_utf8_bytes"]):
                    raise FrozenBankError("SRA skill content or payload mismatch")
                skills.append(FrozenSraLogicSkill(
                    row["skill_id"], row["name"], row["description"], content,
                    entry["content_sha256"], payload, entry["payload_sha256"],
                ))
        except FrozenBankError:
            raise
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
            raise FrozenBankError(f"Cannot load SRA LogicBench bank: {error}") from error
        self._manifest = manifest
        self._manifest_sha256 = expected_manifest_sha256
        self._content_sha256 = _digest(skill_bytes)
        self._skills = tuple(skills)
        self._by_id = MappingProxyType({skill.skill_id: skill for skill in skills})

    @property
    def bank_id(self) -> str:
        return self._manifest["bank_id"]

    @property
    def payload_renderer(self) -> str:
        return SRA_LOGICBENCH19_RENDERER

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @property
    def content_sha256(self) -> str:
        return self._content_sha256

    @property
    def skills(self) -> tuple[FrozenSraLogicSkill, ...]:
        return self._skills

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return EXPECTED_IDS

    def __len__(self) -> int:
        return len(self._skills)

    def get(self, skill_id: str) -> FrozenSraLogicSkill:
        try:
            return self._by_id[skill_id]
        except (KeyError, TypeError) as error:
            raise FrozenBankError(f"Unknown SRA skill ID: {skill_id!r}") from error

    def router_catalog(self) -> list[dict[str, str]]:
        return [
            {"skill_id": skill.skill_id, "name": skill.name, "description": skill.description}
            for skill in self._skills
        ]

    def verification_summary(self) -> dict[str, Any]:
        return {
            "status": "verified_local_snapshot",
            "bank_id": self.bank_id,
            "manifest_sha256": self.manifest_sha256,
            "skills_json_sha256": self.content_sha256,
            "source_commit": self._manifest["source"]["commit"],
            "skill_count": len(self),
            "payload_renderer": self.payload_renderer,
            "router_implemented": True,
            "evaluation_instances_in_bank": False,
        }


def load_sra_logicbench19() -> FrozenSraLogicBenchBank:
    return FrozenSraLogicBenchBank()
