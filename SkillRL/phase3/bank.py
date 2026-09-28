"""Immutable branch banks and bounded, transactional ADD/MODIFY/DELETE/MERGE."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .common import content_hash, digest, positive_int, require, sha256, strict_json, write_new


@dataclass(frozen=True)
class Skill:
    skill_id: str
    name: str
    description: str
    payload: str
    revision: int = 0
    parents: tuple[str, ...] = ()

    def __post_init__(self):
        for text in (self.skill_id, self.name, self.description, self.payload):
            require(isinstance(text, str) and bool(text.strip()), "Empty skill field")
        require(self.payload.startswith("### "), "Skill payload requires a canonical Markdown wrapper")
        positive_int(self.revision, "revision", zero=True)
        for parent in self.parents:
            sha256(parent)

    @property
    def payload_sha256(self):
        return content_hash(self.payload)

    @property
    def version_sha256(self):
        return digest(asdict(self))


class Bank:
    def __init__(self, branch_id, skills, *, inactive=(), retired_ids=(), parent=None, event="initial", source=None):
        require(isinstance(branch_id, str) and branch_id and "/" not in branch_id and ".." not in branch_id,
                "Invalid branch identity")
        self.branch_id = branch_id
        self.skills = tuple(sorted(skills, key=lambda item: item.skill_id))
        self.inactive = tuple(sorted(inactive, key=lambda item: (item.skill_id, item.revision)))
        require(bool(self.skills) and len(self.skill_ids) == len(set(self.skill_ids)), "Empty or duplicate active skills")
        self.retired_ids = tuple(sorted(set(retired_ids)))
        self.parent, self.event, self.source = parent, event, source
        if parent is not None:
            sha256(parent)

    @classmethod
    def initial(cls, branch_id):
        from agent_system.memory.frozen_skill_bank import load_skillnet37
        original = load_skillnet37()
        return cls(branch_id, [Skill(item.skill_id, item.name, item.description, item.payload) for item in original.skills],
                   source=original.manifest_sha256)

    @property
    def skill_ids(self):
        return tuple(item.skill_id for item in self.skills)

    @property
    def active_versions(self):
        return {item.skill_id: item.version_sha256 for item in self.skills}

    @property
    def manifest_sha256(self):
        return digest(self.record())

    @property
    def content_sha256(self):
        return digest(self.active_versions)

    @property
    def bank_id(self):
        return f"phase3:{self.branch_id}:{self.manifest_sha256}"

    def __len__(self):
        return len(self.skills)

    def get(self, skill_id):
        for skill in self.skills:
            if skill.skill_id == skill_id:
                return skill
        raise KeyError("Inactive or unknown skill ID")

    def router_catalog(self):
        return [{"skill_id": item.skill_id, "name": item.name, "description": item.description} for item in self.skills]

    def record(self):
        return {"schema_version": "skillrl.phase3.bank.v1", "branch_id": self.branch_id,
                "skills": [asdict(item) for item in self.skills], "inactive": [asdict(item) for item in self.inactive],
                "retired_ids": list(self.retired_ids), "parent": self.parent, "event": self.event, "source": self.source}

    def save(self, directory):
        return write_new(Path(directory) / f"{self.manifest_sha256}.json", self.record())

    @classmethod
    def load(cls, path, expected_sha256):
        value = strict_json(Path(path).read_text())
        require(digest(value) == sha256(expected_sha256), "Changed bank snapshot")
        require(set(value) == {"schema_version", "branch_id", "skills", "inactive", "retired_ids", "parent", "event", "source"}
                and value["schema_version"] == "skillrl.phase3.bank.v1", "Invalid bank schema")
        make = lambda item: Skill(**{**item, "parents": tuple(item["parents"])})
        result = cls(value["branch_id"], map(make, value["skills"]), inactive=map(make, value["inactive"]),
                     retired_ids=value["retired_ids"], parent=value["parent"], event=value["event"], source=value["source"])
        require(result.manifest_sha256 == expected_sha256, "Noncanonical bank snapshot")
        return result

    def apply(self, proposal, *, event_id, evidence_ids, max_units=3, payload_validator=None):
        """Build a candidate bank; never mutate the current bank or publish acceptance."""
        require(isinstance(event_id, str) and bool(event_id), "Missing event identity")
        positive_int(max_units, "mutation budget")
        require(max_units <= 3 and set(proposal) == {"operations"}, "Invalid operation budget/schema")
        operations = proposal["operations"]
        require(isinstance(operations, list) and 1 <= len(operations) <= max_units, "Invalid operation count")
        active = {item.skill_id: item for item in self.skills}
        inactive, retired, touched, created = list(self.inactive), set(self.retired_ids), set(), []
        units, counts = 0, {name: 0 for name in ("ADD", "MODIFY", "DELETE", "MERGE", "NOOP")}
        for index, operation in enumerate(operations):
            require(set(operation) == {"op", "targets", "skill", "rationale", "evidence_ids"}, "Unexpected operation fields")
            kind = operation["op"]
            require(kind in counts and isinstance(operation["rationale"], str) and operation["rationale"].strip(),
                    "Unknown operation or missing rationale")
            refs = operation["evidence_ids"]
            require(isinstance(refs, list) and all(isinstance(ref, str) and ref in evidence_ids for ref in refs),
                    "Unregistered edit evidence")
            require(kind == "NOOP" or bool(refs), "Mutations require evidence references")
            targets = operation["targets"]
            require(isinstance(targets, list), "Targets must be versioned IDs")
            ids, parents = [], []
            for target in targets:
                require(set(target) == {"skill_id", "version_sha256"}, "Invalid edit target")
                sid = target["skill_id"]
                require(sid in active and sid not in touched and sid not in ids, "Unknown/repeated edit target")
                require(active[sid].version_sha256 == target["version_sha256"], "Stale edit target")
                ids.append(sid)
                parents.append(active[sid].version_sha256)
            needed = {"ADD": 0, "MODIFY": 1, "DELETE": 1, "NOOP": 0}
            require(len(ids) >= 2 if kind == "MERGE" else len(ids) == needed[kind], "Wrong target cardinality")
            units += len(ids) + 1 if kind == "MERGE" else int(kind != "NOOP")
            require(units <= max_units, "Normalized mutation budget exceeded")
            if kind in ("DELETE", "NOOP"):
                require(operation["skill"] is None, "DELETE/NOOP cannot create content")
            else:
                fields = operation["skill"]
                require(isinstance(fields, dict) and set(fields) == {"name", "description", "body"}, "Invalid skill content")
                for value in fields.values():
                    require(isinstance(value, str) and value.strip(), "Empty editor content")
                sid = ids[0] if kind == "MODIFY" else f"evolved:{self.branch_id}:{digest([event_id, index])[:20]}"
                require(kind == "MODIFY" or sid not in set(active) | retired, "New IDs must not be reused")
                revision = active[sid].revision + 1 if kind == "MODIFY" else 0
                item = Skill(sid, fields["name"], fields["description"], f"### Phase3 Skill: {sid}\n\n{fields['body']}",
                             revision, tuple(parents))
                if payload_validator is not None:
                    payload_validator(item)
                if kind == "MODIFY":
                    inactive.append(active[sid])
                else:
                    created.append(sid)
                active[sid] = item
            if kind in ("DELETE", "MERGE"):
                for sid in ids:
                    inactive.append(active.pop(sid))
                    retired.add(sid)
            if kind == "NOOP":
                require(len(operations) == 1, "NOOP must be the only operation")
            touched.update(ids)
            counts[kind] += 1
        candidate = self if units == 0 else Bank(self.branch_id, active.values(), inactive=inactive, retired_ids=retired,
                                                parent=self.manifest_sha256, event=event_id, source=self.source)
        return candidate, {"counts": counts, "mutation_units": units, "touched_existing_ids": sorted(touched),
                           "created_ids": created, "active_before": len(self), "active_after": len(candidate),
                           "noop": units == 0}

    def rollback_to(self, prior, *, event_id):
        require(self.branch_id == prior.branch_id and self.source == prior.source, "Foreign rollback bank")
        history = {item.version_sha256: item for item in (*self.inactive, *self.skills, *prior.inactive)}
        for item in prior.skills:
            history.pop(item.version_sha256, None)
        return Bank(self.branch_id, prior.skills, inactive=history.values(),
                    retired_ids=set(self.retired_ids) | (set(self.skill_ids) - set(prior.skill_ids)),
                    parent=self.manifest_sha256, event=event_id, source=self.source)
