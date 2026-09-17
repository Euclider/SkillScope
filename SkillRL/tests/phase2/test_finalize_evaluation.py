import pytest

from phase2.finalize_evaluation import assert_prospective


def prediction():
    return {"target_gold_read": False, "protocol_sha256": "protocol", "features_sha256": "features",
            "created_at": "2026-09-13T13:00:00+00:00"}


def test_prospective_lock_must_precede_every_target_trajectory():
    assert_prospective(prediction(), [{"created_at": "2026-09-13T13:00:01+00:00"}], "protocol", "features")
    with pytest.raises(ValueError, match="postdate"):
        assert_prospective(prediction(), [{"created_at": "2026-09-13T12:59:59+00:00"}], "protocol", "features")


def test_completion_requires_unchanged_features_and_unopened_gold():
    rows = [{"created_at": "2026-09-13T14:00:00+00:00"}]
    with pytest.raises(ValueError, match="changed"): assert_prospective(prediction(), rows, "protocol", "changed")
    with pytest.raises(ValueError, match="prospective"): assert_prospective({**prediction(), "target_gold_read": True}, rows, "protocol", "features")
    with pytest.raises(ValueError, match="postdate"): assert_prospective(prediction(), [], "protocol", "features")


def test_finalizer_preserves_raw_p_and_archives_complete_results(tmp_path, monkeypatch):
    import json
    import pandas as pd
    from phase1.archive import sha256_file
    from phase2 import finalize_evaluation as module
    root = tmp_path
    report = root/"report.md"; report.write_text("# Execution\n")
    config = {"parent_update": 35, "windows": [{"start": 35, "end": 40}], "report_path": str(report)}
    (root/"protocol.json").write_text(json.dumps(config))
    signal = root/"window_signals/u0035-u0040"; signal.mkdir(parents=True)
    (signal/"skill_context_features.parquet").write_bytes(b"frozen feature bytes")
    feature = {"D_contribution": .2, "P_int": -.3, "delta_centered_norm": 2.,
               "delta_norm": 3., "C_upd": .4, "gate_coverage": .5}
    locked = {**prediction(), "protocol_sha256": sha256_file(root/"protocol.json"),
              "features_sha256": sha256_file(signal/"skill_context_features.parquet"),
              "ranking_scores": [{"skill_id": "s", "context_id": "clean", "phase": "all", "supported": True,
                                  "raw_features": feature}]}
    (signal/"prediction.json").write_text(json.dumps(locked))
    locked["ranking_scores"].append({"skill_id": "s", "context_id": "clean", "phase": "early", "supported": False,
                                     "raw_features": feature})
    (signal/"prediction.json").write_text(json.dumps(locked))
    trajectory = root/"new.json"
    trajectory.write_text(json.dumps({"trajectory_id": "new", "success": False, "steps": []}))
    rows = pd.DataFrame([{"update": 35, "trajectory_id": "old", "created_at": "2026-09-12T00:00:00+00:00"},
                         {"update": 40, "trajectory_id": "new", "created_at": "2026-09-13T14:00:00+00:00",
                          "trajectory_path": str(trajectory), "success": False}])
    monkeypatch.setattr(module, "validate_extended", lambda *a: None)
    monkeypatch.setattr(module, "read_evaluations", lambda *a: rows)
    metrics = root/"window_metrics"; metrics.mkdir(); (root/"reports").mkdir()
    def report_command(*a, **k):
        pd.DataFrame([{"start_update": 35, "global_update": 40, "skill_id": "s", "context_id": "clean",
                       "phase": "all", "control": "placebo", "utility_old": .2, "utility_new": .1,
                       "delta_utility": -.1, "ci_low": -.2, "ci_high": .05}]).to_parquet(metrics/"utility_units.parquet")
        (metrics/"ranking_metrics.csv").write_text("score\nD_contribution\n")
    monkeypatch.setattr(module.subprocess, "run", report_command)
    module.finalize(root)
    raw = pd.read_csv(metrics/"raw_features_and_semantic_utility.csv")
    assert raw.P_int.iloc[0] == -.3  # not the +.3 oriented ranking score
    assert len(raw) == 2 and not raw[raw.phase == "early"].gold_evaluation_available.item()
    assert json.loads((root/"evaluation_completion.json").read_text())["status"] == "complete_and_verified"
    module.finalize(root)
    assert report.read_text().count("<!-- phase2-evaluation-completion-audit-v1 -->") == 1
