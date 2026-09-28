"""All-37 natural invocation support and a window-specific legacy-math adapter."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .common import REPO, digest, file_hash, load_preparation, read_json, write_new_bytes, write_new_json
from .evaluate import collect_results, job_plan, result_path
from .runtime import authorized_runtime, runtime_settings


def build_support(preparation, source, output):
    from agent_system.memory.frozen_skill_bank import load_skillnet37
    from phase1.first_invocation import build_anchor
    from phase1.build_all_first_invocation_anchors import round_robin_games
    preparation, source, output = Path(preparation).resolve(), Path(source).resolve(), Path(output).resolve()
    load_preparation(preparation)
    spec = read_json(preparation.parent / "spec.json")
    plan = read_json(source / "plan.json")
    if (plan["identity"]["preparation_sha256"] != file_hash(preparation)
            or plan["identity"]["purpose"] != "anchors"):
        raise ValueError("Natural anchors must come from this preparation's anchor-only source plan")
    canonical = job_plan(preparation, plan["identity"]["checkpoint"], plan["identity"]["update"],
                         plan["identity"]["split"], "anchors")
    if any(plan.get(key) != value for key, value in canonical.items()):
        raise ValueError("Source plan does not contain the full registered game/seed set")
    rows, missing = collect_results(plan, source)
    if missing or not (source / "completion.json").exists():
        raise ValueError("Full source traversal must finish before choosing natural anchors")
    if read_json(source / "completion.json").get("plan_sha256") != digest(plan):
        raise ValueError("Source completion belongs to a different plan")
    bank = load_skillnet37()
    by_skill = {skill.skill_id: [] for skill in bank.skills}
    for row in rows:
        trajectory = row["result"]
        if [step["step_index"] for step in trajectory["steps"]] != list(range(len(trajectory["steps"]))):
            raise ValueError("Incomplete or reordered source trajectory")
        index = {**row["job"], "checkpoint_id": f"u{row['job']['update']:04d}",
                 "max_steps": spec["evaluation"]["max_steps"], "context_id": "all_alfworld"}
        for skill_id in {step.get("selected_skill_id") for step in trajectory["steps"]} - {None}:
            if skill_id not in by_skill:
                raise ValueError("Foreign skill selected in source trajectory")
            anchor = build_anchor(trajectory=trajectory, trajectory_path=str(result_path(source, row['job']['job_id'])),
                                  skill_id=skill_id, source_index=index)
            anchor["context_id"] = "all_alfworld"
            by_skill[skill_id].append(anchor)
    controls = read_json(preparation.parent / "placebos.json")["controls"]
    settings = spec["evaluation"]
    from .day_budget import select_gold_skills
    eligible = [skill for skill, anchors in by_skill.items()
                if len(anchors) >= settings['minimum_anchor_occurrences']
                and len({a['game_id'] for a in anchors}) >= settings['minimum_anchor_games']]
    selected_skills = set(select_gold_skills(eligible, settings))
    coverage, sets = [], []
    for skill_id, anchors in sorted(by_skill.items()):
        games = len({anchor["game_id"] for anchor in anchors})
        naturally_supported = skill_id in eligible
        supported = skill_id in selected_skills
        selected = round_robin_games(anchors, settings["maximum_anchors_per_skill"]) if supported else []
        stem = skill_id.replace(":", "--")
        path, control_path = output / f"{stem}.jsonl", output / f"{stem}.placebo.json"
        if selected:
            content = "".join(json.dumps(anchor, ensure_ascii=False, sort_keys=True) + "\n" for anchor in selected)
            write_new_bytes(path, content.encode())
            write_new_json(control_path, {**controls[skill_id], "original_text": bank.get(skill_id).payload})
            sets.append({"skill_id": skill_id, "context_id": "all_alfworld",
                         "anchors_path": str(path), "anchors_sha256": file_hash(path),
                         "placebo_path": str(control_path), "placebo_sha256": file_hash(control_path)})
        coverage.append({"skill_id": skill_id, "occurrences": len(anchors), "games": games,
                         "supported": supported, "selected": len(selected),
                         "unsupported_reason": None if supported else (
                             'budget_not_selected' if naturally_supported else "insufficient_natural_support")})
    manifest = {"preparation": str(preparation), "preparation_sha256": file_hash(preparation),
                "source_plan_sha256": file_hash(source / "plan.json"),
                "source_update": plan["identity"]["update"], "split": plan["identity"]["split"],
                "source_checkpoint_identity": plan.get("checkpoint_identity"),
                "all_candidate_count": 37, "coverage": coverage, "anchor_sets": sets,
                "anchor_count": sum(row["selected"] for row in coverage)}
    write_new_json(output / "manifest.json", manifest)
    return manifest


def register_window(preparation, support, training_root, output, start, authorization=None):
    preparation, support = Path(preparation).resolve(), Path(support).resolve()
    training_root, output = Path(training_root).resolve(), Path(output).resolve()
    load_preparation(preparation)
    spec, evidence = read_json(preparation.parent / "spec.json"), read_json(support)
    window = next((window for window in spec["readout"]["windows"] if window["start"] == start), None)
    if window is None or evidence["source_update"] != start or not evidence["anchor_sets"]:
        raise ValueError("Unregistered window or insufficient natural support; abstain, do not force calls")
    if evidence["preparation_sha256"] != file_hash(preparation):
        raise ValueError("Support belongs to another preparation")
    from .assets import model_inventory
    if evidence.get("source_checkpoint_identity") != model_inventory(training_root / "models" / f"u{start:04d}"):
        raise ValueError("Anchor source weights/tokenizer differ from the readout start endpoint")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Window output must be new; no old report regeneration")
    # Keep the validated math/ranking implementation, not the old library or launcher.
    template_path = REPO / "phase2/config/ranking_u35_to40_v1.json"
    template = read_json(template_path)
    config = {key: copy.deepcopy(template[key]) for key in ("signals", "prediction", "ranking", "primary_control", "primary_estimand")}
    config['ranking']['scores'].update(spec['readout'].get('additional_fixed_ranking_scores', {}))
    end = window["end"]
    eval_settings = spec["evaluation"]
    runtime = runtime_settings(spec, training_root / "router.sqlite3", 0)
    if authorization is not None:
        from .common import require_authorization
        permit = require_authorization(authorization, preparation, "evaluation")
        runtime = authorized_runtime(spec, permit["router_cache_path"], permit)
    runtime.update(preparation=str(preparation), authorization_path=str(Path(authorization).resolve()) if authorization else None,
                   data_root=spec["data_root"])
    config.update(schema_version="phase2.extended.skillnet37.v1", status="frozen", root=str(output),
                  run_id=f"skillnet37-s{spec['seed']}-u{start}-u{end}",
                  rl_path_id=f"skillnet37-s{spec['seed']}", parent_update=start,
                  post_updates=list(range(start + 1, end + 1)), windows=[window],
                  window_direction="start_batch_endpoint_projection", runtime=runtime,
                  training={"run_id_prefix": f"skillnet37-s{spec['seed']}"},
                  score_definition_source_sha256=file_hash(template_path),
                  evaluation={"split": evidence["split"], "anchor_sets": evidence["anchor_sets"], "anchor_count": evidence["anchor_count"],
                              "arms": ["original", "placebo", "null"],
                              "old_evidence_seeds": eval_settings["evidence_seeds"], "gold_seeds": eval_settings["gold_seeds"],
                              "shards": 8, "bootstrap_repetitions": 10000, "tau_M": 0.05,
                              **{key: eval_settings[key] for key in ("temperature", "top_p", "history_length", "max_steps", "max_new_tokens", "max_prompt_tokens")}})
    from phase2.protocol import validate_extended
    validate_extended(config, REPO)
    sources = ['models', 'batches', 'old_logprobs', 'optimizer_steps']
    if spec['readout'].get('capture_scope') == 'window_start_old_only_v1':
        config['capture_scope'] = spec['readout']['capture_scope']
        if not (training_root / 'batches' / f'u{start+1:04d}' / 'training_batch.pt').is_file():
            raise ValueError('Missing actual window-start training batch')
    else:
        sources.append('new_logprobs')
    for name in sources:
        source = training_root / name
        if not source.is_dir():
            raise ValueError(f"Required training evidence is missing: {source}")
    # Source links are new paths; this command never deletes source evidence.
    output.mkdir(parents=True, exist_ok=True)
    for name in sources:
        source = training_root / name
        (output / name).symlink_to(source, target_is_directory=True)
    write_new_json(output / "protocol.json", config)
    write_new_json(output / "manifest.json", {"registered_protocol_sha256": file_hash(output / "protocol.json"),
                                            "skill_bank_sha256": spec["bank_manifest_sha256"],
                                            "support_manifest_sha256": file_hash(support)})
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("anchors")
    build.add_argument("--preparation", type=Path, required=True)
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    window = commands.add_parser("window")
    window.add_argument("--preparation", type=Path, required=True)
    window.add_argument("--support", type=Path, required=True)
    window.add_argument("--training-root", type=Path, required=True)
    window.add_argument("--output", type=Path, required=True)
    window.add_argument("--start", type=int, required=True)
    window.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if args.command == "anchors":
        result = build_support(args.preparation, args.source, args.output)
    else:
        result = register_window(args.preparation, args.support, args.training_root, args.output, args.start, args.authorization)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
