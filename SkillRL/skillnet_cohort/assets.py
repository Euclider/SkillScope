"""Offline ALFWorld inventory and exact-token neutral controls. No env reset."""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path

from .common import file_hash, read_json

TASKS = {
    "pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
}
EXPECTED = {"train": 3553, "valid_seen": 140, "valid_unseen": 134}
PROMPT_BOUNDARIES = ("", "\n\nYour admissible actions", "\n\n## Current Progress")


def game_inventory(data_root, expected=EXPECTED):
    root = Path(data_root).resolve()
    result = {}
    for split, count in expected.items():
        rows = []
        for parent, _, files in os.walk(root / "json_2.1.1" / split):
            if "traj_data.json" not in files or "movable" in parent or "Sliced" in parent:
                continue
            folder = Path(parent)
            task = read_json(folder / "traj_data.json")["task_type"]
            game = folder / "game.tw-pddl"
            if task not in TASKS or not game.is_file() or not read_json(game).get("solvable"):
                continue
            rows.append({"game_id": game.relative_to(root).as_posix(), "task_type": task,
                         "game_sha256": file_hash(game), "trajectory_sha256": file_hash(folder / "traj_data.json")})
        rows.sort(key=lambda row: row["game_id"])
        if len(rows) != count or len({row["game_id"] for row in rows}) != count:
            raise ValueError(f"{split}: expected {count} distinct solvable games, found {len(rows)}")
        result[split] = {"count": count, "task_counts": dict(sorted(Counter(row["task_type"] for row in rows).items())),
                         "games": rows, "runtime_coverage_verified": False}
    return {"schema_version": "skillnet.games.v1", "data_root": str(root), "splits": result}


def model_inventory(model_path):
    root = Path(model_path).resolve()
    required = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    files = [root / name for name in required] + sorted(root.glob("*.safetensors"))
    index = root / "model.safetensors.index.json"
    if index.exists():
        files.append(index)
        named = set(read_json(index)["weight_map"].values())
        if named != {path.name for path in root.glob("*.safetensors")}:
            raise ValueError("Model shard inventory disagrees with its index")
    if not list(root.glob("*.safetensors")):
        raise ValueError("No local model weights")
    return {"model_path": str(root), "files": [
        {"path": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size} for path in files
    ], "weights_loaded": False}


class LocalTokenizer:
    """No transformers imports, model load, downloads, or tokenizer cache writes."""
    def __init__(self, model_path):
        from tokenizers import Tokenizer
        self.backend = Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.backend.decode(ids, skip_special_tokens=True)


def neutral_control(tokenizer, payload):
    target = len(tokenizer.encode(payload))
    header = payload.splitlines()[0]
    if not header.startswith("### "):
        raise ValueError("Unexpected canonical payload wrapper")
    prefix = header + "\n\n"
    # BPE merges across payload/template boundaries. Match both the isolated
    # payload and its actual two ALFWorld prompt suffixes, not just the former.
    boundaries = PROMPT_BOUNDARIES
    targets = [len(tokenizer.encode(payload + boundary)) for boundary in boundaries]
    trailing = re.search(r"\s*$", payload).group()
    sentence = "Typography studies printed letters, spacing, page margins, fonts, and paper textures. "
    stream = tokenizer.encode(sentence * (target + 1))
    # Matching is audited after decode/re-encode, never inferred from a sliced ID length.
    guess = max(0, target - len(tokenizer.encode(prefix)))
    candidates = list(range(max(0, guess - 16), min(len(stream), guess + 16) + 1))
    for suffix in dict.fromkeys((trailing, "." + trailing, ":" + trailing, "`" + trailing)):
        for length in candidates:
            text = prefix + tokenizer.decode(stream[:length]).rstrip() + suffix
            if [len(tokenizer.encode(text + boundary)) for boundary in boundaries] == targets:
                return {"text": text, "target_token_count": target, "actual_token_count": target,
                        "boundary_token_counts": targets,
                        "construction": "canonical first-line wrapper and trailing whitespace plus fixed task-irrelevant typography stream; exact isolated and prompt-boundary token counts, not identical internal Markdown layout"}
    raise ValueError(f"Cannot produce exact {target}-token control")


def build_controls(bank, tokenizer):
    return {"schema_version": "skillnet.placebos.v1", "bank_manifest_sha256": bank.manifest_sha256,
            "controls": {skill.skill_id: {**neutral_control(tokenizer, skill.payload),
                                        "original_payload_sha256": skill.payload_sha256}
                         for skill in bank.skills}}


def validate_controls(preparation):
    from agent_system.memory.frozen_skill_bank import load_skillnet37
    assets = Path(preparation).resolve().parent
    tokenizer = LocalTokenizer(read_json(assets / "spec.json")["model_path"])
    controls = read_json(assets / "placebos.json")["controls"]
    bank = load_skillnet37()
    if set(controls) != set(bank.skill_ids):
        raise ValueError("Controls must cover all 37 frozen skills")
    for skill in bank.skills:
        control = controls[skill.skill_id]
        if (control["original_payload_sha256"] != skill.payload_sha256
                or any(len(tokenizer.encode(skill.payload + boundary)) != len(tokenizer.encode(control["text"] + boundary))
                       for boundary in PROMPT_BOUNDARIES)):
            raise ValueError(f"Payload or prompt-boundary token control mismatch: {skill.skill_id}")
    return True


def write_placeholders(path, count, split):
    import pyarrow as pa
    import pyarrow.parquet as pq
    # Public recipe uses only row count/modality; no unrelated dataset download.
    rows = [{"data_source": "text", "prompt": [{"role": "user", "content": ""}],
             "ability": "agent", "extra_info": {"split": split, "index": i}} for i in range(count)]
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows), sink)
    content = sink.getvalue().to_pybytes()
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
