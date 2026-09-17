import json

from phase1.archive import append_jsonl_idempotent


def test_repeated_append_does_not_duplicate(tmp_path):
    output = tmp_path / "records.jsonl"
    row = {"checkpoint": 0, "game": "g1", "condition": "full_bank"}
    fields = ("checkpoint", "game", "condition")
    assert append_jsonl_idempotent(output, [row], fields) == 1
    assert append_jsonl_idempotent(output, [row], fields) == 0
    assert len([json.loads(line) for line in output.read_text().splitlines()]) == 1

