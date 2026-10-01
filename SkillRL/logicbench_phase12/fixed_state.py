"""Reward-directed readouts on the actual U1 LogicBench GRPO response tokens.

This module reads the pre-update training batch and frozen model checkpoints.
It never reads Eval labels, generates a new action, or changes the skill bank.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from phase1.logicbench_single_step import LogicBenchQuestion, build_prompt
from phase2.stable_direction import token_signals


@torch.inference_mode()
def score_recorded_tokens(model, prompt_ids, response_ids, *, device="cuda"):
    """Return full-vocabulary log preferences for each recorded next token."""
    if not prompt_ids or not response_ids:
        raise ValueError("A fixed decision needs prompt and response tokens")
    ids = torch.as_tensor([list(prompt_ids) + list(response_ids[:-1])],
                          dtype=torch.long, device=device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   use_cache=False, logits_to_keep=len(response_ids))
    logits = output.logits[0].float()
    if logits.shape[0] != len(response_ids):
        raise ValueError("Wrong fixed-state score positions")
    return torch.log_softmax(logits, dim=-1)


def summarize_token_signals(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["skill_id"]].append(row)
    result = []
    for skill_id, group in sorted(groups.items()):
        def mean(key, members=group):
            return math.fsum(float(r[key]) for r in members) / len(members)
        signed_gate = -math.fsum(float(r["P_int"]) * bool(r["gate"]) for r in group) / len(group)
        b = mean("delta_centered_norm")
        directed = [float(r["advantage"]) * float(r["chosen_delta"]) for r in group]
        orientation = math.fsum(directed) / (math.fsum(map(abs, directed)) + 1e-12)
        result.append({"skill_id": skill_id,
                       "n_questions": len({r["question_id"] for r in group}),
                       "n_trajectories": len({r["trajectory_id"] for r in group}),
                       "n_loss_tokens": len(group),
                       "D_signed_gate": signed_gate,
                       "D_original": mean("D_contribution"),
                       "D_signed": -mean("P_int"),
                       "D_real": mean("D_real"),
                       "D_factor": -b * orientation,
                       "D_orientation": -orientation,
                       "M_delta_centered": b,
                       "M_delta_raw": mean("delta_norm"),
                       "P_int": mean("P_int")})
    return result


def _load_model(path, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True).to(device).eval()


def _control_ids(tokenizer, question):
    row = LogicBenchQuestion(question["question_id"], question["question"],
                             question["task_type"], question["answer"], "")
    text = build_prompt(row, None)
    chat = tokenizer.apply_chat_template([{"role": "user", "content": text}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tokenizer(chat, add_special_tokens=False)["input_ids"]


def run(seed_dir: Path, new_model: Path, output: Path, *, max_rows=None):
    from transformers import AutoTokenizer
    old_model = Path("/home/wangyifan/model/Qwen3.5-4B")
    batch_path = seed_dir / "phase2/batches/u0001/training_batch.pt"
    archived = torch.load(batch_path, map_location="cpu", weights_only=False)
    if archived["schema_version"] != "phase2.exact_training_batch.v1":
        raise ValueError("Wrong training evidence schema")
    tensors, meta, nontensor = (archived[key] for key in
                               ("tensors", "metadata", "non_tensor_batch"))
    if len(meta) != len(tensors["responses"]):
        raise ValueError("Decision metadata and training batch differ")
    questions = {row["question_id"]: row for row in json.loads(
        (Path(__file__).resolve().parents[1] /
         "data/logicbench/sra19/aug_split_v2/train.json").read_text())}
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(old_model, local_files_only=True)
    old = _load_model(old_model, device)
    new = _load_model(new_model, device)
    output.mkdir(parents=True, exist_ok=False)
    token_rows = []
    native_hf_old_chosen_max_abs_error = 0.
    total = len(meta) if max_rows is None else min(len(meta), max_rows)
    for i in range(total):
        mask = tensors["phase2_actual_loss_mask"][i].bool()
        if not bool(mask.any()):
            continue
        response_ids = tensors["responses"][i, mask].tolist()
        prompt_mask = tensors["attention_mask"][i, :tensors["prompts"].shape[-1]].bool()
        prompt_ids = tensors["prompts"][i, prompt_mask].tolist()
        question_id = str(nontensor["question_id"][i])
        question = questions[question_id]
        control_ids = _control_ids(tokenizer, question)
        old_capture = torch.load(seed_dir / "phase2/old_logprobs/u0001" /
                                 f"row-{i:06d}.pt", map_location="cpu", weights_only=False)
        if old_capture["token_ids"].tolist() != response_ids:
            raise ValueError("Old actor capture and actual loss tokens disagree")
        # Score every condition with the same HF forward backend. The native
        # FSDP capture verifies recorded actions, but mixing its rounded old
        # distribution with HF new/control distributions creates an avoidable
        # backend interaction in the readout itself.
        old_skill = score_recorded_tokens(old, prompt_ids, response_ids, device=device)
        new_skill = score_recorded_tokens(new, prompt_ids, response_ids, device=device)
        old_control = score_recorded_tokens(old, control_ids, response_ids, device=device)
        new_control = score_recorded_tokens(new, control_ids, response_ids, device=device)
        advantages = tensors["advantages"][i, mask].to(device)
        actions = torch.as_tensor(response_ids, device=device)
        native_chosen = old_capture["trainer_chosen_log_probs"].to(device)
        hf_chosen = old_skill.gather(1, actions[:, None]).flatten()
        native_hf_old_chosen_max_abs_error = max(
            native_hf_old_chosen_max_abs_error,
            float((native_chosen - hf_chosen).abs().max()))
        signals = token_signals(old_skill, new_skill, old_control, new_control,
                                actions, advantages, tau_delta=1e-8,
                                tau_c=0., epsilon=1e-12)
        u = (new_skill.double() - old_skill.double())
        delta = u - (new_control.double() - old_control.double())
        uc = u - u.mean(-1, keepdim=True)
        dc = delta - delta.mean(-1, keepdim=True)
        chosen_u = u.gather(1, actions[:, None]).flatten()
        real = -advantages.double() * chosen_u * (uc * dc).sum(-1) / (
            uc.square().sum(-1) + 1e-12)
        skill_id = meta[i]["info"]["selected_skill_id"]
        for j in range(len(response_ids)):
            token_rows.append({"skill_id": skill_id, "question_id": question_id,
                "trajectory_id": meta[i]["trajectory_id"], "decision_id": meta[i]["decision_id"],
                "response_token_offset": j, "action_token_id": response_ids[j],
                **{key: value[j].item() for key, value in signals.items()
                   if value.ndim == 1}, "D_real": real[j].item()})
        if (i + 1) % 16 == 0:
            print(json.dumps({"rows_scored": i + 1, "loss_tokens": len(token_rows)}), flush=True)
    if not token_rows:
        raise ValueError("No actual loss tokens were scored")
    pd.DataFrame(token_rows).to_parquet(output / "token_signals.parquet", index=False)
    summary = summarize_token_signals(token_rows)
    pd.DataFrame(summary).to_csv(output / "skill_scores.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": "skillscope.logicbench_fixed_state.v1", "rows_scored": total,
        "loss_tokens": len(token_rows), "skills_scored": len(summary),
        "native_actor_vs_hf_old_chosen_max_abs_error": native_hf_old_chosen_max_abs_error,
        "four_condition_forward_backend": "same_hf_bfloat16_sdpa",
        "old_model": str(old_model), "new_model": str(new_model),
        "training_batch": str(batch_path), "eval_labels_read": False,
        "readout_only_forward_passes": True}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    run(args.seed_dir, args.new_model, args.output, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
