"""LogicBench Phase3 preflight (default) or explicitly approved one-boundary run.

Example (no model generation/API):
python -m phase3.logicbench_run --setting configs/phase3_logicbench_draft_v1.json \
    --seed 404 --method D_signed_gate --output artifacts/logicbench/phase3-preflight/seed404
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from phase1.logicbench_single_step import EVAL_PATH, EVAL_SHA256, GenerationOutput, LogicBenchQuestion, build_prompt, extract_final_answer, load_logicbench_eval
from skillnet_cohort.common import file_hash
from .api import APIConfig, JSONClient
from .common import digest, finite, positive_int, require, strict_json, write_new
from .evolution import paired_effect
from .logicbench import READOUTS, SkillPromptBudgetError, editor_input, evaluate_bank, initial_bank, rank_skills, revise_once, validate_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/logicbench/sra19/aug_split_v2"
OLD_MODEL = Path("/home/wangyifan/model/Qwen3.5-4B")


def validate_setting(config):
    require(config["schema_version"] == "skillrl.phase3.logicbench.setting.v1"
            and config["mode"] == "fixed_u5_single_revision", "Unsupported LogicBench setting")
    require(type(config["execution_approved"]) is bool, "Invalid approval state")
    seeds = config["eval_seeds"]
    require(seeds and len(set(seeds)) == len(seeds)
            and all(type(s) is int and s >= 0 for s in seeds), "Invalid evaluation seeds")
    require(config["rng_mode"] in ("legacy_seed", "question_seed_v1"), "Invalid RNG mode")
    require(config["methods"] and len(set(config["methods"])) == len(config["methods"])
            and set(config["methods"]) <= {*READOUTS, "no_edit", "failure_rate", "random"}, "Invalid selectors")
    require(config["train_seeds"] and len(set(config["train_seeds"])) == len(config["train_seeds"])
            and set(config["train_seeds"]) <= {404, 505}, "This importer supports existing 404/505 U0-U5 windows")
    for field in ("candidate_k", "evidence_per_skill", "mutation_units", "gate_questions", "max_new_tokens", "max_prompt_tokens", "max_edited_skill_tokens"):
        positive_int(config[field], field)
    require(config["mutation_units"] <= 3, "Bank patch budget is at most 3")
    require(0 <= finite(config["gate_tolerance_pp"], "tolerance") <= 100, "Invalid gate tolerance")
    require(finite(config["temperature"], "temperature") > 0 and 0 < finite(config["top_p"], "top_p") <= 1,
            "Invalid decoding parameters")
    APIConfig(**config["editor"])


def load_training_evidence(seed_dir, bank):
    """Decode actual archived actions; verify reward and prompt without Eval data."""
    import torch
    from transformers import AutoTokenizer

    path = seed_dir / "phase2/batches/u0001/training_batch.pt"
    archived = torch.load(path, map_location="cpu", weights_only=False)
    require(archived["schema_version"] == "phase2.exact_training_batch.v1", "Wrong batch schema")
    tensors, metadata, nt = archived["tensors"], archived["metadata"], archived["non_tensor_batch"]
    tokenizer = AutoTokenizer.from_pretrained(OLD_MODEL, local_files_only=True)
    train = strict_json((DATA / "train.json").read_text())
    questions = {r["question_id"]: r for r in train}
    require(len(metadata) == len(tensors["responses"]) == 1024, "Expected original 128 x 8 U1 batch")
    records, expected_prompts = [], {}
    for i, meta in enumerate(metadata):
        qid, sid = str(nt["question_id"][i]), str(nt["selected_skill_id"][i])
        q = questions[qid]
        require(meta["global_update"] == 1 and meta["environment_step"] == 0
                and meta["info"]["question_id"] == qid and meta["info"]["selected_skill_id"] == sid,
                "Wrong training decision identity")
        require(str(nt["context_id"][i]) == q["context_id"] and str(nt["answer"][i]) == q["answer"]
                and str(nt["task_type"][i]) == q["task_type"], "Training question/context metadata differs")
        if (qid, sid) not in expected_prompts:
            row = LogicBenchQuestion(qid, q["question"], q["task_type"], q["answer"], "")
            chat = tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(row, bank.get(sid).payload)}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False)
            expected_prompts[qid, sid] = tokenizer(chat, add_special_tokens=False)["input_ids"]
        prompt_mask = tensors["attention_mask"][i, :tensors["prompts"].shape[1]].bool()
        require(tensors["prompts"][i, prompt_mask].tolist() == expected_prompts[qid, sid],
                "Archived prompt differs from frozen bank/direct-label template")
        response = tokenizer.decode(tensors["responses"][i, tensors["response_mask"][i].bool()], skip_special_tokens=True)
        success = extract_final_answer(response, q["task_type"]) == q["answer"]
        require(float(nt["episode_rewards"][i]) == float(success) == float(meta["info"]["reward"]),
                "Archived action/reward mismatch")
        records.append({"evidence_id": meta["trajectory_id"], "question_id": qid,
            "context_id": q["context_id"], "split": "train", "sampling_policy_update": 0,
            "selected_skill_id": sid, "skill_version_sha256": bank.get(sid).version_sha256,
            "question": q["question"], "task_type": q["task_type"], "response": response, "success": success})
    counts = Counter(r["question_id"] for r in records)
    require(len(counts) == 128 and set(counts.values()) == {8}, "Unbalanced training response groups")
    return records, train


def prepare(config, seed, method):
    from scripts.prepare_sra_logicbench_phase12 import select_questions

    validate_setting(config)
    require(seed in config["train_seeds"] and method in config["methods"], "Unregistered branch")
    seed_dir = (ROOT / config["phase12_root"]).resolve() / f"seed{seed}"
    bank = initial_bank(f"logicbench-s{seed}-{method}")
    manifest = strict_json((DATA / "manifest.json").read_text())
    for split in ("train", "dev"):
        require(file_hash(DATA / f"{split}.json") == manifest[f"{split}_sha256"], "Changed audited Aug split")
    evidence, train = load_training_evidence(seed_dir, bank)
    readout_path = seed_dir / "readout/skill_scores.csv"
    readout_manifest = strict_json((seed_dir / "readout/manifest.json").read_text())
    require(Path(readout_manifest["training_batch"]).resolve() == seed_dir / "phase2/batches/u0001/training_batch.pt"
            and Path(readout_manifest["new_model"]).resolve() == seed_dir / "merged-u5"
            and Path(readout_manifest["old_model"]).resolve() == OLD_MODEL.resolve()
            and readout_manifest["eval_labels_read"] is False and readout_manifest["rows_scored"] == len(evidence),
            "Wrong readout window/evidence")
    with readout_path.open() as f:
        scores = [{k: v if k == "skill_id" else float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    rankings = {m: rank_skills(bank, evidence, scores, m, k=len(scores), selection_seed=config["selection_seed"])
                for m in (*READOUTS, "failure_rate", "random")}
    priorities = [] if method == "no_edit" else rankings[method][:config["candidate_k"]]
    payload = None if method == "no_edit" else editor_input(bank, evidence, priorities,
        evidence_per_skill=config["evidence_per_skill"], mutation_units=config["mutation_units"])
    if payload is not None:
        payload["max_edited_skill_policy_tokens"] = config["max_edited_skill_tokens"]
        payload["max_complete_policy_prompt_tokens"] = config["max_prompt_tokens"]
    dev = strict_json((DATA / "dev.json").read_text())
    # Phase12 prepared dev[:32] for the trainer dataset contract. Exclude their
    # entire contexts regardless of whether validation actually ran.
    used_dev_contexts = {r["context_id"] for r in dev[:32]}
    available = [r for r in dev if r["context_id"] not in used_dev_contexts]
    gate = select_questions(available, seed=config["split_seed"], count=config["gate_questions"])
    validate_split(train, gate)
    source_files = [DATA / "train.json", DATA / "dev.json", DATA / "manifest.json", readout_path,
                    seed_dir / "readout/manifest.json", seed_dir / "phase2/batches/u0001/training_batch.pt"]
    audit = {"schema_version": "skillrl.phase3.logicbench.preflight.v1", "training_seed": seed,
        "method": method, "config_sha256": digest(config), "bank_sha256": bank.manifest_sha256,
        "source_sha256": {str(p): file_hash(p) for p in source_files},
        "checkpoint": str(seed_dir / "merged-u5"), "rollouts": len(evidence),
        "training_questions": len({r["question_id"] for r in evidence}), "eligible_skills": len(scores),
        "all_rankings": rankings, "priority_ids": priorities, "editor_payload": payload,
        "gate_questions": len(gate), "gate_contexts": len({r["context_id"] for r in gate}),
        "gate_question_ids": [r["question_id"] for r in gate], "gate_sha256": digest(gate),
        "excluded_phase12_dev_contexts": len(used_dev_contexts),
        "eval_labels_used_for_selection_or_editing": False, "external_api_calls": 0,
        "generation_calls": 0}
    return bank, gate, audit


def prompt_validator(tokenizer, questions, config):
    """Check input feasibility only; no gold answers or performance outcomes."""
    def validate(skill):
        if len(tokenizer(skill.payload, add_special_tokens=False)["input_ids"]) > config["max_edited_skill_tokens"]:
            raise SkillPromptBudgetError("Edited skill exceeds declared policy-token cap")
        for question in questions:
            chat = tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(question, skill.payload)}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False)
            if len(tokenizer(chat, add_special_tokens=False)["input_ids"]) > config["max_prompt_tokens"]:
                raise SkillPromptBudgetError("Edited skill exceeds complete policy prompt cap")
    return validate


def evaluation_prompt_contexts():
    # Project benchmark INPUTS to a label-free type for length checks. Never
    # send these questions to editor/selector or compute their reward here.
    require(file_hash(EVAL_PATH) == EVAL_SHA256, "Changed Eval input snapshot")
    return [LogicBenchQuestion(r["instance_id"], r["question"], r["eval_data"]["task_type"], "", "")
            for r in strict_json(EVAL_PATH.read_text())]


class HFGenerator:
    def __init__(self, checkpoint, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        require(torch.cuda.is_available(), "GPU unavailable; formal evaluation has no CPU fallback")
        self.torch, self.config = torch, config
        self.tokenizer = AutoTokenizer.from_pretrained(OLD_MODEL, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True).to("cuda:0").eval()

    def __call__(self, prompt, seed):
        torch, config = self.torch, self.config
        chat = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False)
        inputs = self.tokenizer(chat, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        length = inputs["input_ids"].shape[-1]
        require(length <= config["max_prompt_tokens"], "Edited skill exceeds prompt budget; no truncation")
        with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed)
            result = self.model.generate(**inputs, do_sample=True, temperature=config["temperature"],
                top_p=config["top_p"], max_new_tokens=config["max_new_tokens"],
                pad_token_id=self.tokenizer.eos_token_id)[0, length:]
        return GenerationOutput(self.tokenizer.decode(result, skip_special_tokens=True),
            int(result.numel()), int(result.numel()) == config["max_new_tokens"], int(length))


def summarize_test(before, after):
    report = {"overall": paired_effect(before, after)}
    for kind in sorted({r["task_type"] for r in before}):
        report[kind] = paired_effect([r for r in before if r["task_type"] == kind],
                                    [r for r in after if r["task_type"] == kind])
    for label, rows in (("before", before), ("after", after)):
        if all("format_valid" in r for r in rows):
            report[label + "_format_valid_rate"] = sum(r["format_valid"] for r in rows) / len(rows)
            report[label + "_length_cap_rate"] = sum(r["hit_length_cap"] for r in rows) / len(rows)
    return report


def execute(config, bank, gate, audit, output):
    """Only called after explicit confirmed setting; policy remains fixed at U5."""
    from .logicbench_routing import LogicBenchRouterPool
    require(config["execution_approved"] is True, "Phase3 setting must be confirmed before execution")
    checkpoint = Path(audit["checkpoint"])
    # Bind actual weights, not merely a checkpoint directory, before any calls.
    weights = sorted(checkpoint.glob("*.safetensors"))
    require(weights, "Missing U5 policy weights")
    policy_files = weights + [checkpoint / "config.json"]
    if (checkpoint / "generation_config.json").exists():
        policy_files.append(checkpoint / "generation_config.json")
    tokenizer_names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                       "tokenizer.model", "vocab.json", "merges.txt", "added_tokens.json")
    tokenizer_files = [OLD_MODEL / name for name in tokenizer_names if (OLD_MODEL / name).exists()]
    write_new(output / "model_identity.json", {str(p): file_hash(p) for p in policy_files + tokenizer_files})
    generate = HFGenerator(checkpoint, config)
    import transformers
    write_new(output / "decoding_identity.json", {
        "inherited_generation_config": generate.model.generation_config.to_dict(),
        "explicit_overrides": {k: config[k] for k in ("temperature", "top_p", "max_new_tokens", "max_prompt_tokens", "rng_mode")},
        "pad_token_id": generate.tokenizer.eos_token_id, "do_sample": True, "enable_thinking": False,
        "torch_version": str(generate.torch.__version__), "transformers_version": transformers.__version__})
    provider = LogicBenchRouterPool(config["router"], output / "router-local.sqlite3", bank.branch_id)
    questions = [LogicBenchQuestion(r["question_id"], r["question"], r["task_type"], r["answer"], "") for r in gate]
    budget_check = prompt_validator(generate.tokenizer, questions + evaluation_prompt_contexts(), config)

    def evaluate(current, rows=questions):
        return evaluate_bank(current, rows, provider, generate,
                             seeds=config["eval_seeds"], rng_mode=config["rng_mode"])

    try:
        selected = bank
        if audit["editor_payload"] is not None:
            # Warm router before spending any editor call.
            provider.for_bank(bank).route_question(questions[0].question)
            api = JSONClient(APIConfig(**config["editor"]), output / "editor.sqlite3", allow_live=True)
            result = revise_once(bank, api=api, payload=audit["editor_payload"], evaluate=evaluate,
                output=output / "revision", event_id="u5", tolerance_pp=config["gate_tolerance_pp"],
                gate_ids={q.instance_id for q in questions}, identity={"preflight_sha256": digest(audit),
                    "model_identity_sha256": file_hash(output / "model_identity.json"),
                    "decoding_identity_sha256": file_hash(output / "decoding_identity.json")},
                payload_validator=budget_check)
            selected = result["bank"]
        bank.save(output / "banks")
        selected.save(output / "banks")
        # Only now expose final labels to the scorer, after acceptance. The
        # earlier feasibility check used question/task-type fields only.
        test = load_logicbench_eval()
        outputs = []
        for name, current in (("before", bank), ("after", selected)):
            path = output / f"test_{name}.json"
            rows = strict_json(path.read_text()) if path.exists() else (
                outputs[0] if name == "after" and selected.manifest_sha256 == bank.manifest_sha256 else evaluate(current, test))
            require(all(r["bank_sha256"] == current.manifest_sha256 for r in rows), "Changed test bank")
            require({(r["game_id"], r["eval_seed"]) for r in rows}
                    == {(q.instance_id, s) for q in test for s in config["eval_seeds"]}, "Incomplete test coverage")
            write_new(path, rows)
            outputs.append(rows)
        write_new(output / "test_report.json", summarize_test(*outputs))
        write_new(output / "complete.json", {"status": "complete", "policy_fixed": True,
            "selected_bank_sha256": selected.manifest_sha256, "preflight_sha256": digest(audit)})
    finally:
        provider.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = strict_json(args.setting.read_text())
    validate_setting(config)
    if args.execute:
        require(config["execution_approved"] is True, "Phase3 setting must be confirmed before execution")
    bank, gate, audit = prepare(config, args.seed, args.method)
    write_new(args.output / "setting.json", config)
    write_new(args.output / "preflight.json", audit)
    if args.execute:
        execute(config, bank, gate, audit, args.output)
    else:
        print(canonical_summary(audit))


def canonical_summary(audit):
    from .common import canonical
    return canonical({k: audit[k] for k in ("training_seed", "method", "rollouts", "training_questions",
        "eligible_skills", "priority_ids", "gate_questions", "external_api_calls", "generation_calls")})


if __name__ == "__main__":
    main()
