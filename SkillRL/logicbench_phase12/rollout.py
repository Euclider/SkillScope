"""One-question rollout with the ordinary VERL actor and exact-answer reward."""

from __future__ import annotations

import json

import numpy as np

from phase1.logicbench_single_step import extract_final_answer
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def single_step_rollout(
    batch: DataProto, actor, tokenizer, *, repeats: int, run_id: str, update: int
) -> DataProto:
    """Generate repeated answers; retain question-level GRPO groups and provenance.

    Routing and prompt construction happen before this function. The gold answer
    is only used after generation and is never forwarded to the actor.
    """
    if not isinstance(repeats, int) or repeats < 1 or not run_id or update < 1:
        raise ValueError("Invalid LogicBench rollout identity or repeats")
    required = {"question_id", "selected_skill_id", "answer", "task_type", "data_source"}
    if not required.issubset(batch.non_tensor_batch):
        raise ValueError(f"LogicBench metadata missing: {sorted(required - set(batch.non_tensor_batch))}")
    if len(batch) * repeats % actor.world_size:
        raise ValueError("Repeated LogicBench batch must be divisible by actor world size")
    if len(set(map(str, batch.non_tensor_batch["question_id"]))) != len(batch):
        raise ValueError("Question IDs must be distinct within an update")

    repeated = batch.repeat(repeat_times=repeats, interleave=True)
    actor_input = DataProto(batch=repeated.batch, meta_info=dict(repeated.meta_info))
    padded, padding = pad_dataproto_to_divisor(actor_input, actor.world_size)
    generated = unpad_dataproto(actor.generate_sequences(padded), padding)
    if len(generated) != len(repeated):
        raise ValueError("Actor returned the wrong number of LogicBench generations")
    texts = tokenizer.batch_decode(generated.batch["responses"], skip_special_tokens=True)
    parsed = [extract_final_answer(text, str(task_type)) for text, task_type in
              zip(texts, repeated.non_tensor_batch["task_type"])]
    rewards = np.asarray([float(answer == str(gold)) for answer, gold in
                          zip(parsed, repeated.non_tensor_batch["answer"])], dtype=np.float32)
    valid = np.asarray([answer is not None for answer in parsed], dtype=bool)
    group_ids, trajectory_ids, decisions, metadata = [], [], [], []
    for i in range(len(repeated)):
        question_id = str(repeated.non_tensor_batch["question_id"][i])
        skill_id = str(repeated.non_tensor_batch["selected_skill_id"][i])
        group_id = f"{run_id}:u{update}:q{question_id}"
        trajectory_id = f"{group_id}:r{i % repeats}"
        decision_id = f"{trajectory_id}:s0"
        item = {"decision_id": decision_id, "global_update": update,
                "environment_step": 0, "trajectory_id": trajectory_id,
                "group_id": group_id,
                "info": {"selected_skill_id": skill_id, "question_id": question_id,
                         "task_type": str(repeated.non_tensor_batch["task_type"][i]),
                         "reward": float(rewards[i]), "format_valid": bool(valid[i])}}
        group_ids.append(group_id)
        trajectory_ids.append(trajectory_id)
        decisions.append(decision_id)
        metadata.append(json.dumps(item, ensure_ascii=False, sort_keys=True))

    generated.non_tensor_batch = dict(repeated.non_tensor_batch)
    generated.non_tensor_batch.update({
        "uid": np.asarray(group_ids, dtype=object),
        "traj_uid": np.asarray(trajectory_ids, dtype=object),
        "is_action_valid": valid,
        "episode_rewards": rewards,
        "episode_lengths": np.ones(len(repeated), dtype=np.float32),
        "tool_callings": np.zeros(len(repeated), dtype=np.float32),
        "rewards": rewards.copy(),
        "active_masks": np.ones(len(repeated), dtype=bool),
        "parsed_answer": np.asarray(parsed, dtype=object),
        "phase2_decision_id": np.asarray(decisions, dtype=object),
        "phase2_metadata": np.asarray(metadata, dtype=object),
    })
    return generated
