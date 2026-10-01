"""Rebuild the frozen SRA-Bench LogicBench skill subset from pinned upstream bytes.

This script intentionally never reads the benchmark's question/answer file.
It copies the 19 upstream skill records without editing their text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

COMMIT = "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f"
ARCHIVE_SHA256 = "2e8abf91ad992bebe2c1bf87cb7b77996edf1828bf55ddeba44c456956673849"
CORPUS_SHA256 = "16ee509ae5bea8c2e17167dffecd89100a7d8dfa31256c3742426758c7169b5e"
LICENSE_SHA256 = "47bef547eb656e100323e67b6808dc59bf14846e0c1b4cef0bb9020e9c237b2a"
SKILL_IDS = tuple(f"logicbench_{index:03d}" for index in range(19))
RENDERER = "skillscope.sra_logicbench_markdown.v1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def payload(skill: dict[str, str]) -> str:
    return f"### Frozen Skill: {skill['skill_id']}\n\n{skill['content']}"


def build(archive: Path, license_file: Path, output: Path) -> None:
    if output.exists():
        raise ValueError(f"Refusing to overwrite frozen bank: {output}")
    if digest(archive.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError("Upstream corpus archive SHA-256 mismatch")
    license_bytes = license_file.read_bytes()
    if digest(license_bytes) != LICENSE_SHA256:
        raise ValueError("Upstream LICENSE SHA-256 mismatch")
    with zipfile.ZipFile(archive) as source:
        if source.namelist() != ["corpus.json"]:
            raise ValueError("Unexpected archive members")
        corpus_bytes = source.read("corpus.json")
    if digest(corpus_bytes) != CORPUS_SHA256:
        raise ValueError("Upstream corpus JSON SHA-256 mismatch")
    corpus = json.loads(corpus_bytes)
    selected = [row for row in corpus if row.get("skill_id") in SKILL_IDS]
    selected.sort(key=lambda row: row["skill_id"])
    if [row["skill_id"] for row in selected] != list(SKILL_IDS):
        raise ValueError("Expected exactly one record for each of the 19 LogicBench skill IDs")
    if any(set(row) != {"skill_id", "name", "description", "content"} for row in selected):
        raise ValueError("Upstream skill record schema changed")

    output.mkdir(parents=True)
    skill_bytes = (json.dumps(selected, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (output / "skills.json").write_bytes(skill_bytes)
    (output / "LICENSE").write_bytes(license_bytes)
    entries = []
    for row in selected:
        content_bytes = row["content"].encode("utf-8")
        rendered = payload(row).encode("utf-8")
        entries.append({
            "skill_id": row["skill_id"],
            "name": row["name"],
            "description": row["description"],
            "content_sha256": digest(content_bytes),
            "payload_sha256": digest(rendered),
            "payload_utf8_bytes": len(rendered),
        })
    manifest = {
        "schema_version": "skillscope.sra_logicbench_bank.v1",
        "bank_id": "sra-logicbench-19-277fd8d2bbd7",
        "source": {
            "repository": "https://github.com/oneal2000/SR-Agents",
            "commit": COMMIT,
            "archive_path": "data/bench/corpus/corpus.json.zip",
            "archive_sha256": ARCHIVE_SHA256,
            "corpus_json_sha256": CORPUS_SHA256,
            "license_sha256": LICENSE_SHA256,
            "license": "MIT",
        },
        "content_policy": {
            "source_skill_text_unchanged": True,
            "router_catalog_fields": ["skill_id", "name", "description"],
            "payload_renderer": RENDERER,
            "payload_prefix": "### Frozen Skill: {skill_id}\\n\\n",
            "candidate_scope": "19 LogicBench gold skills",
            "candidate_order": "upstream skill_id ascending",
            "read_only": True,
            "task_answers_excluded": True,
        },
        "counts": {"skills": len(selected), "source_records": len(selected)},
        "skills_json_sha256": digest(skill_bytes),
        "skills": entries,
    }
    (output / "manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--license-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.archive, args.license_file, args.output)


if __name__ == "__main__":
    main()
