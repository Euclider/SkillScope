"""Opt-in, lossless training evidence. Does not alter training tensors or RNG."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from phase1.archive import append_jsonl_idempotent, atomic_write_json, jsonable, sha256_file, utc_now


def save_tensor_file(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def attach_decisions(batch, infos, *, run_id: str, update: int, step: int) -> None:
    ids, metadata = [], []
    for i, info in enumerate(infos):
        trajectory = str(batch.non_tensor_batch["traj_uid"][i])
        decision = f"{run_id}:u{update}:{trajectory}:s{step}"
        item = {"decision_id": decision, "global_update": update,
                "environment_step": step, "trajectory_id": trajectory,
                "group_id": str(batch.non_tensor_batch["uid"][i]),
                "info": jsonable(info)}
        ids.append(decision)
        metadata.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
    batch.non_tensor_batch["phase2_decision_id"] = np.asarray(ids, dtype=object)
    batch.non_tensor_batch["phase2_metadata"] = np.asarray(metadata, dtype=object)


def mark_batch(batch, *, root: str, update: int) -> None:
    if "phase2_decision_id" not in batch.non_tensor_batch:
        raise ValueError("Phase2 decision IDs were lost before actor update")
    from phase2.staged_storage import AMENDMENT, admit
    if (Path(root)/AMENDMENT).exists():
        config = json.loads((Path(root)/"protocol.json").read_text())
        length = batch.batch["responses"].shape[-1]
        tokens = int(batch.batch["attention_mask"][:, -length:].bool().sum())
        admit(root, config, update, tokens=tokens)
    batch.batch["phase2_row_index"] = torch.arange(len(batch), dtype=torch.int64)
    batch.meta_info["phase2_capture"] = {"root": root, "update": int(update)}


def archive_batch(batch, *, root: str, update: int, config) -> None:
    from omegaconf import OmegaConf

    target = Path(root) / "batches" / f"u{update:04d}"
    tensors = {k: v.detach().cpu().clone() for k, v in batch.batch.items()}
    metadata = [json.loads(str(x)) for x in batch.non_tensor_batch["phase2_metadata"]]
    response_length = tensors["responses"].shape[1]
    multi_turn = bool(config.actor_rollout_ref.rollout.multi_turn.enable)
    mask = tensors["loss_mask"][:, -response_length:] if multi_turn else tensors["attention_mask"][:, -response_length:]
    if mask.shape != tensors["advantages"].shape:
        raise ValueError("Advantage and actual actor mask shape differ")
    if not torch.isfinite(tensors["advantages"][mask.bool()]).all():
        raise ValueError("Non-finite training advantage")
    tensors["phase2_actual_loss_mask"] = mask.clone()
    save_tensor_file(target / "training_batch.pt", {
        "schema_version": "phase2.exact_training_batch.v1",
        "tensors": tensors, "metadata": metadata,
        "non_tensor_batch": jsonable(batch.non_tensor_batch),
        "meta_info": jsonable(batch.meta_info),
    })
    atomic_write_json(target / "manifest.json", {
        "created_at": utc_now(), "global_update": update,
        "row_count": len(metadata),
        "unique_decisions": len({x["decision_id"] for x in metadata}),
        "loss_tokens": int(mask.sum()),
        "batch_sha256": sha256_file(target / "training_batch.pt"),
        "config": OmegaConf.to_container(config, resolve=True),
        "capture_point": "after actual GRPO advantages, before any optimizer step",
    })


def capture_old_logits(logits, returned_log_probs, micro_batch, capture) -> None:
    """Capture the exact live OLD actor full-vocabulary distribution."""
    stage = capture.get("stage", "old")
    if stage not in {"old", "new"}:
        raise ValueError(f"Unknown live-logit capture stage: {stage}")
    root = Path(capture["root"]) / f"{stage}_logprobs" / f"u{capture['update']:04d}"
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    response_length = micro_batch["responses"].shape[-1]
    mask = micro_batch["attention_mask"][:, -response_length:].bool()
    for i, row in enumerate(micro_batch["phase2_row_index"].tolist()):
        valid = mask[i]
        lp = torch.log_softmax(logits[i, valid].float(), dim=-1)
        token_ids = micro_batch["responses"][i, valid]
        chosen = lp.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        trainer = returned_log_probs[i, valid].float()
        payload = {"row_index": int(row), "rank": rank,
                   "token_positions": valid.nonzero().flatten().cpu(),
                   "token_ids": token_ids.cpu(), "log_probs": lp.cpu(),
                   "trainer_chosen_log_probs": trainer.cpu(),
                   "chosen_log_probs": chosen.cpu()}
        path = root / f"row-{row:06d}.pt"
        save_tensor_file(path, payload)
        append_jsonl_idempotent(root / f"rank-{rank}.jsonl", [{
            "row_index": row, "rank": rank, "tokens": len(token_ids),
            "vocab_size": lp.shape[-1], "path": str(path),
            "chosen_max_abs_error": float((chosen-trainer).abs().max()) if len(chosen) else 0.,
            "created_at": utc_now(),
        }], unique_fields=("row_index",))


def optimizer_counter(optimizer) -> int:
    return max((int(v["step"].item()) for v in optimizer.state.values() if "step" in v), default=0)


def log_forward_progress(capture, *, role, index, total):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    atomic_write_json(Path(capture["root"])/"forward_progress"/f"rank-{rank}.json", {
        "updated_at": utc_now(), "global_update": capture["update"],
        "role": role, "completed_microbatches": index, "total_microbatches": total,
    })


def archive_optimizer_step(capture, *, epoch, minibatch, rows, before, after, grad_norm, lr):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    append_jsonl_idempotent(Path(capture["root"]) / "optimizer_steps" / f"u{capture['update']:04d}-rank{rank}.jsonl", [{
        "global_update": capture["update"], "rank": rank, "epoch": epoch,
        "minibatch": minibatch, "batch_row_indices": rows,
        "adam_step_before": before, "adam_step_after": after,
        "grad_norm": float(grad_norm), "learning_rate": lr, "created_at": utc_now(),
    }], unique_fields=("global_update", "rank", "epoch", "minibatch"))
