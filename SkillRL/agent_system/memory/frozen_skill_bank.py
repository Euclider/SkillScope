"""Hash-pinned, read-only skill packages; no retrieval model or script execution.

The SkillNet-37 snapshot is an opt-in bank, not a replacement for historical
SkillsOnlyMemory settings.  This module does not choose skills: an independent
router must supply the selected ID.  Source scripts are rendered as text only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

from .base import BaseMemory


SCHEMA_VERSION = "skillrl.frozen_skill_bank.v1"
RENDERER_VERSION = "skillrl.raw_skill_package.v1"
SKILLNET37_COMMIT = "5c472b36d2a435001fdae3bc8439886d8050645a"
SKILLNET37_MANIFEST_SHA256 = "0767ff7578b1e997119a36b5f636fec6ebc1d0ac600ffb40b1e599065b7fe514"
SKILLNET37_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "memory_data/alfworld/skillnet37/manifest.json"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FrozenBankError(ValueError):
    """The frozen bank, its manifest, or its use violates the bank contract."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenBankError(message)


def _relative_path(value: str) -> PurePosixPath:
    _require(isinstance(value, str) and bool(value), "Expected a nonempty relative path")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and "\\" not in value
        and ":" not in value
        and ".." not in path.parts
        and path.as_posix() == value
        and value != ".",
        f"Unsafe or noncanonical source path: {value!r}",
    )
    return path


def _front_matter(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    _require(bool(lines) and lines[0] == "---", "Missing SKILL.md YAML header")
    try:
        end = lines.index("---", 1)
        header = yaml.safe_load("\n".join(lines[1:end]))
    except (ValueError, yaml.YAMLError) as error:
        raise FrozenBankError("Invalid SKILL.md YAML header") from error
    _require(isinstance(header, dict), "SKILL.md header must be a mapping")
    return header


@dataclass(frozen=True)
class FrozenSkillFile:
    """Unmodified UTF-8 source bytes, addressed relative to the upstream repo."""

    path: str
    sha256: str
    content: bytes

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")


@dataclass(frozen=True)
class FrozenSkill:
    skill_id: str
    name: str
    description: str
    directory: str
    files: tuple[FrozenSkillFile, ...]
    payload: str
    payload_sha256: str


def render_raw_package(skill_id: str, directory: str, files: Sequence[FrozenSkillFile]) -> str:
    """Preserve each file's text, including whitespace; add only file headers."""
    _require(bool(files), "Cannot render an empty skill package")
    _require(files[0].path == f"{directory}/SKILL.md", "SKILL.md must be first")
    parts = [f"### Frozen Skill: {skill_id}\n\n", files[0].text]
    for file in files[1:]:
        relative = PurePosixPath(file.path).relative_to(directory).as_posix()
        parts.extend([f"\n\n#### Supporting file: {relative}\n\n", file.text])
    return "".join(parts)


class FrozenSkillBank:
    """Load only when manifest, exact inventory, raw bytes and payloads match.

    The caller must supply a manifest digest from a separately frozen protocol.
    Hashes inside an unpinned manifest alone would not establish a fixed bank.
    No executable source is imported, evaluated, or run during loading/rendering.
    """

    def __init__(self, manifest_path: str | Path, *, expected_manifest_sha256: str):
        _require(
            isinstance(expected_manifest_sha256, str)
            and bool(_SHA256.fullmatch(expected_manifest_sha256)),
            "A pinned SHA-256 manifest digest is required",
        )
        manifest_path = Path(manifest_path)
        try:
            raw = manifest_path.read_bytes()
            _require(_digest(raw) == expected_manifest_sha256, "Manifest SHA-256 mismatch")
            manifest = json.loads(raw)
            self._load(manifest_path, manifest, expected_manifest_sha256)
        except FrozenBankError:
            raise
        except (OSError, UnicodeError, KeyError, TypeError, ValueError) as error:
            raise FrozenBankError(f"Cannot load frozen bank: {error}") from error

    def _load(self, manifest_path: Path, manifest: dict[str, Any], manifest_sha256: str) -> None:
        _require(manifest["schema_version"] == SCHEMA_VERSION, "Unsupported manifest schema")
        policy = manifest["content_policy"]
        _require(policy["payload_renderer"] == RENDERER_VERSION, "Unsupported payload renderer")
        for key in ("read_only", "upstream_bytes_unchanged", "include_all_skill_directories"):
            _require(policy[key] is True, f"Frozen bank requires {key}=true")
        for key in ("semantic_deduplication", "instruction_rewriting", "task_type_filtering", "execute_bundled_scripts"):
            _require(policy[key] is False, f"Frozen bank requires {key}=false")

        root = manifest_path.parent / "upstream"
        _require(root.is_dir() and not root.is_symlink(), "Missing or symlinked upstream directory")
        file_rows = manifest["files"]
        paths = [row["path"] for row in file_rows]
        _require(paths == sorted(set(paths)), "File inventory must be sorted and unique")
        for path in paths:
            _relative_path(path)
        actual_paths = []
        for path in root.rglob("*"):
            _require(not path.is_symlink(), f"Symlink in frozen bank: {path}")
            if path.is_dir():
                continue
            _require(path.is_file(), f"Non-regular file in frozen bank: {path}")
            actual_paths.append(path.relative_to(root).as_posix())
        _require(sorted(actual_paths) == paths, "Upstream file inventory mismatch (missing or extra file)")

        source_files: dict[str, FrozenSkillFile] = {}
        for row in file_rows:
            path = root.joinpath(*_relative_path(row["path"]).parts)
            content = path.read_bytes()
            _require(len(content) == row["bytes"], f"Source byte-length mismatch: {row['path']}")
            _require(_digest(content) == row["sha256"], f"Source SHA-256 mismatch: {row['path']}")
            content.decode("utf-8")
            source_files[row["path"]] = FrozenSkillFile(row["path"], row["sha256"], content)
        inventory_text = "".join(f"{row['sha256']}  {row['path']}\n" for row in file_rows)
        content_sha256 = _digest(inventory_text.encode("utf-8"))
        _require(content_sha256 == manifest["content_sha256"], "Content inventory SHA-256 mismatch")

        source = manifest["source"]
        skill_directory = _relative_path(source["skill_directory"]).as_posix()
        license_path = _relative_path(source["license_file"]).as_posix()
        _require(license_path in source_files, "Missing upstream license")
        expected_main_files = sorted(
            path for path in paths
            if path.startswith(skill_directory + "/")
            and PurePosixPath(path).name == "SKILL.md"
        )
        rows = manifest["skills"]
        _require([row["main_file"] for row in rows] == expected_main_files, "Skill inventory mismatch")
        _require(bool(rows), "Frozen skill bank is empty")
        skills = []
        used_paths: set[str] = set()
        ids: set[str] = set()
        names: list[str] = []
        for row in rows:
            directory = _relative_path(row["directory"]).as_posix()
            _require(PurePosixPath(directory).parent.as_posix() == skill_directory, "Unexpected skill directory depth")
            main_file = f"{directory}/SKILL.md"
            _require(row["main_file"] == main_file, "Invalid main skill file")
            package_paths = [main_file] + sorted(
                path for path in paths if path.startswith(directory + "/") and path != main_file
            )
            _require(row["files"] == package_paths, "Skill must include every support file in stable order")
            header = _front_matter(source_files[main_file].text)
            _require(header.get("name") == row["name"] == PurePosixPath(directory).name, "Skill name mismatch")
            _require(header.get("description") == row["description"], "Skill description mismatch")
            _require(isinstance(row["description"], str) and bool(row["description"]), "Empty skill description")
            _require(row["skill_id"] == f"skillnet:{row['name']}", "Invalid SkillNet skill ID")
            _require(row["skill_id"] not in ids, "Duplicate skill ID")
            files = tuple(source_files[path] for path in package_paths)
            payload = render_raw_package(row["skill_id"], directory, files)
            payload_bytes = payload.encode("utf-8")
            _require(len(payload_bytes) == row["payload_utf8_bytes"], "Payload byte-length mismatch")
            _require(_digest(payload_bytes) == row["payload_sha256"], "Payload SHA-256 mismatch")
            skills.append(FrozenSkill(row["skill_id"], row["name"], row["description"], directory, files, payload, row["payload_sha256"]))
            used_paths.update(package_paths)
            ids.add(row["skill_id"])
            names.append(row["name"])
        _require(names == sorted(set(names)), "Skill names must be sorted and unique")
        _require(used_paths | {license_path} == set(paths), "Unassigned file in frozen bank")
        counts = {
            "skills": len(skills),
            "main_files": len(skills),
            "support_files": len(used_paths) - len(skills),
            "upstream_files": len(paths),
            "upstream_utf8_bytes": sum(len(file.content) for file in source_files.values()),
        }
        _require(counts == manifest["counts"], "Manifest counts mismatch")
        self._manifest = manifest
        self._manifest_path = manifest_path.resolve()
        self._manifest_sha256 = manifest_sha256
        self._content_sha256 = content_sha256
        self._skills = tuple(skills)
        self._by_id = MappingProxyType({skill.skill_id: skill for skill in skills})

    @property
    def bank_id(self) -> str:
        return self._manifest["bank_id"]

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @property
    def content_sha256(self) -> str:
        return self._content_sha256

    @property
    def skills(self) -> tuple[FrozenSkill, ...]:
        return self._skills

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(skill.skill_id for skill in self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    def get(self, skill_id: str) -> FrozenSkill:
        try:
            return self._by_id[skill_id]
        except (KeyError, TypeError) as error:
            raise FrozenBankError(f"Unknown skill ID: {skill_id!r}") from error

    def router_catalog(self) -> list[dict[str, str]]:
        """All original names/descriptions; fresh copies, no task filtering."""
        return [
            {"skill_id": skill.skill_id, "name": skill.name, "description": skill.description}
            for skill in self.skills
        ]

    def verification_summary(self) -> dict[str, Any]:
        return {
            "status": "verified_local_snapshot",
            "bank_id": self.bank_id,
            "manifest_sha256": self.manifest_sha256,
            "content_sha256": self.content_sha256,
            "source_commit": self._manifest["source"]["commit"],
            "counts": dict(self._manifest["counts"]),
            "candidate_scope": "all_tasks",
            "payload_renderer": RENDERER_VERSION,
            "maximum_payload_utf8_bytes": max(len(skill.payload.encode("utf-8")) for skill in self.skills),
            "executed_source_scripts": False,
            "router_implemented": False,
        }


def load_skillnet37(manifest_path: str | Path = SKILLNET37_MANIFEST) -> FrozenSkillBank:
    bank = FrozenSkillBank(manifest_path, expected_manifest_sha256=SKILLNET37_MANIFEST_SHA256)
    _require(len(bank) == 37, "SkillNet-37 must contain exactly 37 skills")
    _require(bank._manifest["source"]["commit"] == SKILLNET37_COMMIT, "Unexpected SkillNet commit")
    return bank


class FrozenSkillBankMemory(BaseMemory):
    """Stateless candidate/payload adapter, compatible with single-skill arms.

    ``general_skills`` is only the legacy bundle's transport field; it does not
    reclassify the upstream skills.  Passing the whole candidate bundle to the
    renderer is an error.  No router is implicitly substituted for the planned
    external selector, and historical environment defaults remain unchanged.
    """

    def __init__(self, bank: FrozenSkillBank):
        self._bank = bank

    @property
    def bank(self) -> FrozenSkillBank:
        return self._bank

    @property
    def disabled_skill_ids(self) -> frozenset[str]:
        """Read-only legacy evaluator contract: this full bank is never masked.

        Payload interventions happen after selection; exposing an immutable
        empty mask does not enable candidate filtering or mutate the bank.
        """
        return frozenset()

    def __len__(self) -> int:
        return len(self.bank)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.retrieve("")

    def reset(self, batch_size: int) -> None:
        """No episode state or mutable skill history is stored."""

    def fetch(self, step: int) -> None:
        return None

    def store(self, record: Any) -> None:
        raise FrozenBankError("Frozen bank cannot store rollout-derived updates")

    def retrieve(self, task_description: str, top_k: int | None = None, **kwargs: Any) -> dict[str, Any]:
        _require(not kwargs, "Frozen bank does not support candidate filters")
        _require(top_k is None or (type(top_k) is int and top_k == len(self)), "Frozen bank requires the complete candidate pool; top_k truncation is forbidden")
        candidates = [dict(item, title=item["name"]) for item in self.bank.router_catalog()]
        return {
            "bank_id": self.bank.bank_id,
            "bank_manifest_sha256": self.bank.manifest_sha256,
            "bank_content_sha256": self.bank.content_sha256,
            "candidate_scope": "all_tasks",
            "task_type": "all_tasks",
            "retrieval_mode": "frozen_bank_all",
            "general_skills": candidates,
            "task_specific_skills": [],
            "mistakes_to_avoid": [],
            "task_specific_examples": [],
            "retrieved_skill_ids": list(self.bank.skill_ids),
            "candidate_skill_ids": list(self.bank.skill_ids),
            "injected_skill_ids": [],
            "selected_skill_id": None,
        }

    def selected_bundle(self, skill_id: str | None) -> dict[str, Any]:
        """Package an ID already chosen externally; this performs no routing."""
        if skill_id is not None:
            self.bank.get(skill_id)
        bundle = self.retrieve("")
        bundle["general_skills"] = [item for item in bundle["general_skills"] if item["skill_id"] == skill_id]
        bundle["selected_skill_id"] = skill_id
        bundle["injected_skill_ids"] = [skill_id] if skill_id is not None else []
        bundle["payload_renderer"] = RENDERER_VERSION
        bundle["payload_sha256"] = self.bank.get(skill_id).payload_sha256 if skill_id is not None else _digest(b"")
        return bundle

    def format_for_prompt(self, retrieved_memories: Mapping[str, Any]) -> str:
        """Read canonical raw text only after one valid ID has been selected."""
        for key, value in (
            ("bank_id", self.bank.bank_id),
            ("bank_manifest_sha256", self.bank.manifest_sha256),
            ("bank_content_sha256", self.bank.content_sha256),
        ):
            _require(key not in retrieved_memories or retrieved_memories[key] == value, f"Wrong {key} in selected bundle")
        items = [
            item
            for field in ("general_skills", "task_specific_skills", "mistakes_to_avoid")
            for item in retrieved_memories.get(field, [])
        ]
        _require(len(items) <= 1, "Select at most one skill before rendering; full-bank injection is forbidden")
        skill_id = items[0].get("skill_id") if items else None
        _require(not items or isinstance(skill_id, str), "Selected item lacks a skill ID")
        if "selected_skill_id" in retrieved_memories:
            _require(retrieved_memories["selected_skill_id"] == skill_id, "Selected skill ID disagrees with payload item")
        if "injected_skill_ids" in retrieved_memories:
            _require(retrieved_memories["injected_skill_ids"] == ([skill_id] if skill_id is not None else []), "Injected skill IDs disagree with payload item")
        payload = self.bank.get(skill_id).payload if skill_id is not None else ""
        if "payload_renderer" in retrieved_memories:
            _require(retrieved_memories["payload_renderer"] == RENDERER_VERSION, "Wrong payload renderer in selected bundle")
        if "payload_sha256" in retrieved_memories:
            _require(retrieved_memories["payload_sha256"] == _digest(payload.encode("utf-8")), "Wrong payload SHA-256 in selected bundle")
        return payload

    def add_skills(self, *args: Any, **kwargs: Any) -> int:
        raise FrozenBankError("Frozen bank does not allow adding skills")

    def remove_skill(self, *args: Any, **kwargs: Any) -> bool:
        raise FrozenBankError("Frozen bank does not allow removing skills")

    def save_skills(self, *args: Any, **kwargs: Any) -> None:
        raise FrozenBankError("Frozen bank cannot be overwritten through the memory API")

    def set_disabled_skill_ids(self, *args: Any, **kwargs: Any) -> None:
        raise FrozenBankError("Do not mask routing candidates; apply ORIGINAL/PLACEBO/NULL after selection")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the pinned SkillNet-37 bank without network, models, or environment execution.")
    parser.add_argument("--manifest", type=Path, default=SKILLNET37_MANIFEST)
    args = parser.parse_args(argv)
    try:
        bank = load_skillnet37(args.manifest)
    except FrozenBankError as error:
        parser.exit(1, f"SkillNet-37 verification failed: {error}\n")
    print(json.dumps(bank.verification_summary(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
