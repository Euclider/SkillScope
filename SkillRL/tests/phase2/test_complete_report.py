import pandas as pd
import pytest

from phase2.complete_report import behavior, build, rank_summary


def test_full_report_never_labels_partial_evaluations_complete(tmp_path):
    with pytest.raises(ValueError, match="completed verified"): build(tmp_path)


def test_behavior_pairs_use_fixed_anchor_seed_not_unpaired_success_means():
    rows = []
    for seed in (11, 21):
        for arm in ("original", "placebo", "null"):
            rows.append({"update": 40, "purpose": "gold", "skill_id": "s", "anchor_id": "a",
                "continuation_seed": seed, "arm": arm, "first_action": "x" if arm == "original" else "y",
                "action_sequence": ["x", "z"] if arm == "original" else ["y"],
                "success": arm == "original", "selected_skill_ids": ["s", "g", "s"], "suffix_trajectory_length": 2})
    result = behavior(pd.DataFrame(rows))
    assert result.paired_anchor_repeats.eq(2).all()
    assert result.first_action_flip.eq(1).all() and result.reward_disagreement.eq(1).all()
    assert result.original_suffix_unique_skills.eq(2).all()
    with pytest.raises(ValueError, match="every matched"): behavior(pd.DataFrame(rows[:-1]))


def test_ranking_summary_reports_one_locked_pool_without_selecting_a_different_phase():
    shared = {"context_id": "clean", "phase": "all", "start_update": 35, "global_update": 40,
              "target": "decline", "threshold": 0, "k": 1, "score": "D_contribution",
              "precision_at_k": .5, "captured_change_mass": .25}
    metrics = pd.DataFrame([shared, {**shared, "phase": "late", "precision_at_k": 1}])
    pool = {"context_id": "clean", "start_update": 35, "global_update": 40,
            "shared_supported_units": 3, "shared_skill_ids": ["a", "b", "c"], "excluded_units": 1}
    text = rank_summary(metrics, pool)
    assert "0.500" in text and "1.000" not in text and "3 个 Skill" in text


def test_full_report_end_to_end_has_provenance_and_separate_batch_sections(tmp_path, monkeypatch):
    import json
    from phase1.archive import sha256_file
    from phase2 import complete_report as module
    project = tmp_path/"skill-RL"
    root = project/"SkillRL/artifacts/phase2/run"; root.mkdir(parents=True)
    legacy = project/"legacy"
    archive = legacy/"reports/2026-09-12-observation-audit-v1"; archive.mkdir(parents=True)
    (legacy/"metrics").mkdir()
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
    anchors = root/"anchors.jsonl"
    anchors.write_text(json.dumps({"game_id": "g", "trigger_step": 1})+"\n")
    config = {"post_updates": list(range(36, 41)), "evaluation": {"anchor_sets": [{"skill_id": "s", "anchors_path": str(anchors)}]}}
    write(root/"protocol.json", config)
    output = project/"full.md"
    write(root/module.PLAN, {"protocol_sha256": sha256_file(root/"protocol.json"), "frozen_sources": [],
                            "legacy_root": str(legacy), "output": str(output)})
    pd.DataFrame([{"signal": "D_contribution", "scope": "heldout", "risk_orientation": "+value", "units": 7,
                   "updates": 2, "rho_raw_magnitude": .4, "rho_risk_decline": .5, "ap_any_decline": .8}]).to_csv(archive/"comparisons.csv", index=False)
    pd.DataFrame([{"global_update": 34, "delta_utility": .1}, {"global_update": 35, "delta_utility": -.1}]).to_csv(archive/"heldout.csv", index=False)
    pd.DataFrame([{"unit": "old"}]).to_csv(archive/"all_supported.csv", index=False)
    write(legacy/"metrics/prospective_performance.json", {"results": [{"model": "zero", "test_units": 7,
          "test_updates": 2, "signed_MAE": .04, "conditional_sign_accuracy": None}]})
    for cohort, values in ((legacy, range(31, 36)), (root, range(36, 41))):
        for u in values:
            write(cohort/"signals"/f"u{u:04d}"/"committed.json", {"optimizer_steps": 14, "adam_step_before": 10, "adam_step_after": 24})
            write(cohort/"signals"/f"u{u:04d}"/"parameter_delta.json", {"delta_l2": .2, "relative_delta_l2": .001})
    wm = root/"window_metrics"; wm.mkdir()
    record = {"start_update": 35, "global_update": 40, "control": "placebo", "skill_id": "s", "context_id": "clean",
              "phase": "all", "supported": True, "gold_evaluation_available": True, "anchor_count": 1, "game_count": 1,
              "utility_old": .2, "utility_new": .1, "delta_utility": -.1, "ci_low": -.2, "ci_high": .05,
              "original_old": .8, "original_new": .9, "control_old": .6, "control_new": .8, "delta_original": .1, "delta_control": .2,
              "P_int": -.3, "D_contribution": .2, "D_ungated_contribution": .3, "delta_centered_norm": 2, "delta_norm": 3,
              "C_upd": .4, "gate_coverage": .5}
    pd.DataFrame([record]).to_csv(wm/"raw_features_and_semantic_utility.csv", index=False)
    pd.DataFrame([record, {**record, "control": "null"}]).to_parquet(wm/"utility_units.parquet")
    ranks = [{"start_update": 35, "global_update": 40, "context_id": "clean", "phase": "all", "target": t,
              "threshold": v, "score": s, "n_candidates": 1, "events": 1, "k": 1, "precision_at_k": 1,
              "recall_at_k": 1, "captured_change_mass": 1, "spearman": None, "kendall": None}
             for t in ("decline", "any_change") for v in (0, .05) for s in ("D_contribution", "P_int", "delta_centered_norm", "random_expected")]
    pd.DataFrame(ranks).to_csv(wm/"ranking_metrics.csv", index=False)
    write(wm/"ranking_support.json", {"candidate_pools": [{"context_id": "clean", "phase": "all", "start_update": 35,
          "global_update": 40, "shared_supported_units": 1, "shared_skill_ids": ["s"], "excluded_units": 0}]})
    write(root/"evaluation_completion.json", {"status": "complete_and_verified", "protocol_sha256": sha256_file(root/"protocol.json"),
          "raw_features_and_utility_sha256": sha256_file(wm/"raw_features_and_semantic_utility.csv"),
          "ranking_metrics_sha256": sha256_file(wm/"ranking_metrics.csv")})
    rows = [{"update": u, "purpose": "gold", "skill_id": "s", "anchor_id": "a", "continuation_seed": 1,
             "arm": a, "first_action": "x", "action_sequence": ["x"], "success": True,
             "selected_skill_ids": ["s"], "suffix_trajectory_length": 1} for u in (35, 40) for a in ("original", "placebo", "null")]
    monkeypatch.setattr(module, "read_evaluations", lambda r: pd.DataFrame(rows))
    assert module.build(root) == output
    contents = output.read_text()
    assert "旧单步结果" in contents and "新窗口的边际效用" in contents and "JVP" in contents
    assert json.loads((root/"full_report_completion.json").read_text())["report_sha256"] == sha256_file(output)
