"""CPU audit of actual update evidence, without reading utility labels."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from phase1.archive import atomic_write_json, utc_now
from phase2.measure import counter_input
from phase2.protocol import controls_by_skill


def audit(root, update):
    root = Path(root).resolve()
    directory = root/"batches"/f"u{update:04d}"
    manifest = json.loads((directory/"manifest.json").read_text())
    output = directory/"alignment_audit.json"
    if output.exists():
        record = json.loads(output.read_text())
        if record["batch_sha256"] != manifest["batch_sha256"]:
            raise ValueError("Training batch changed after alignment audit")
        return record
    config = json.loads((root/"protocol.json").read_text())
    batch = torch.load(directory/"training_batch.pt", map_location="cpu", weights_only=False)
    tensors = batch["tensors"]
    tokenizer = AutoTokenizer.from_pretrained(root/"models"/f"u{update-1:04d}")
    repo = Path(__file__).resolve().parents[1]
    controls = controls_by_skill(config,repo)
    seen, stats = set(), defaultdict(lambda: {"decisions":0, "nonzero_decisions":0,
                                             "games":set(), "nonzero_games":set(), "steps":[]})
    counterprompts = 0
    for row, metadata in enumerate(batch["metadata"]):
        assert tensors["phase2_row_index"][row].item() == row
        mask = tensors["phase2_actual_loss_mask"][row].bool()
        assert torch.isfinite(tensors["advantages"][row,mask]).all()
        if metadata["decision_id"] in seen:
            continue
        seen.add(metadata["decision_id"])
        info = metadata["info"]
        skill = info.get("selected_skill_id")
        group = stats[skill]
        group["decisions"] += 1
        group["games"].add(info["extra.gamefile"])
        group["steps"].append(metadata["environment_step"])
        if tensors["advantages"][row,mask].abs().max().item() > 1e-12:
            group["nonzero_decisions"] += 1
            group["nonzero_games"].add(info["extra.gamefile"])
        if skill not in controls:
            continue
        if info["phase2_payload_text"] != controls[skill]["original_text"]:
            raise ValueError(f"Frozen evaluation/training Skill payload mismatch: {skill}")
        for arm in ("placebo", "null"):
            counter_input(tokenizer, metadata, tensors, row, arm, controls[skill]["text"])
            counterprompts += 1
    if len(list((root/"old_logprobs"/f"u{update:04d}").glob("row-*.pt"))) != manifest["row_count"]:
        raise ValueError("Missing old actor full-vocabulary rows")
    records = []
    for skill, values in sorted(stats.items(), key=lambda pair:str(pair[0])):
        records.append({"skill_id":skill, "decisions":values["decisions"],
                        "nonzero_decisions":values["nonzero_decisions"],
                        "games":len(values["games"]), "nonzero_games":len(values["nonzero_games"]),
                        "first_step":min(values["steps"]), "last_step":max(values["steps"])})
    result = {"created_at":utc_now(), "global_update":update, "gold_read":False,
              "batch_sha256":manifest["batch_sha256"], "unique_decisions":len(seen),
              "counterprompts_verified":counterprompts, "temperature":batch["meta_info"]["temperature"],
              "tensor_shapes":{k:list(v.shape) for k,v in tensors.items()}, "skills":records}
    atomic_write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.update), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
