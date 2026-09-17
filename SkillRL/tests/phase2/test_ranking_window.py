import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase1.archive import sha256_file
from phase2.ranking import budget_metrics, evaluate_snapshot, score_snapshot, selection_weights


def settings():
    return {"scores": {"D_contribution": 1, "P_int": -1}, "budgets_k": [1, 2],
            "budgets_fraction": [.25, .5], "event_thresholds": [0., .05]}


def test_ties_use_expected_selection_without_reading_gold():
    assert np.allclose(selection_weights([2, 2, 1], 1), [.5, .5, 0])
    assert np.allclose(selection_weights([2, 2, 1], 2), [1, 1, 0])
    result = budget_metrics([2, 2, 1], [-.2, .1, -.1], 1)
    assert result["precision_at_k"] == .5
    assert result["recall_at_k"] == .25
    assert result["captured_change_mass"] == pytest.approx(1/3)


def test_random_expected_is_budget_fraction_not_lucky_tie_order():
    result = budget_metrics([0, 0, 0, 0], [-.2, -.1, .4, 0], 1)
    assert result["precision_at_k"] == .5
    assert result["recall_at_k"] == .25
    assert result["captured_change_mass"] == pytest.approx(.25)
    assert np.isnan(result["spearman"])


def test_no_decline_recall_is_undefined_and_any_change_is_separate():
    result = budget_metrics([3, 2], [.2, .1], 1)
    assert result["events"] == 0 and np.isnan(result["recall_at_k"])
    assert np.isnan(result["captured_change_mass"])
    assert budget_metrics([3, 2], [.2, .1], 1, target="any_change")["events"] == 2


def test_score_signs_fixed_and_common_pool_excludes_unsupported():
    frame = pd.DataFrame([{"skill_id": f"s{i}", "context_id": "clean", "phase": "all", "supported": i != 2,
                           "D_contribution": i, "P_int": -.1*i} for i in range(3)])
    snap = score_snapshot(frame, settings())
    assert snap[1]["scores"]["P_int"] == .1
    labels = frame.assign(delta_utility=[.1, -.2, -.9])
    result, support, _ = evaluate_snapshot(snap, labels, settings())
    assert result.n_candidates.eq(2).all()
    assert support[0]["shared_skill_ids"] == ["s0", "s1"]
    assert result[(result.score == "D_contribution") & (result.target == "decline") & (result.k == 1)].precision_at_k.eq(1).all()


def test_ranking_does_not_pool_contexts_or_phases():
    frame = pd.DataFrame([{"skill_id": "s", "context_id": c, "phase": p, "supported": True,
                           "D_contribution": 1, "P_int": -.1} for c in ("clean", "heat") for p in ("all", "middle")])
    result, support, _ = evaluate_snapshot(score_snapshot(frame, settings()), frame.assign(delta_utility=-.2), settings())
    assert len(support) == 4
    assert result.n_candidates.eq(1).all()


def test_storage_refuses_broad_target_and_symlinks(tmp_path):
    from phase2.storage import checked_tree
    with pytest.raises(ValueError): checked_tree(tmp_path, tmp_path)
    target = tmp_path/"a"; target.mkdir()
    (target/"foreign").symlink_to(tmp_path/"elsewhere")
    with pytest.raises(ValueError, match="symlinks"): checked_tree(target, tmp_path)


def test_audited_remove_keeps_metadata_and_native_source(tmp_path):
    from phase2.storage import archive_and_remove
    native = tmp_path/"native"; native.mkdir(); (native/"source.pt").write_text("untouched")
    parent = tmp_path/"copies"; target = parent/"copy"; target.mkdir(parents=True)
    (target/"manifest.json").write_text("{}")
    archive = tmp_path/"audit"
    result = archive_and_remove(target, parent, archive, "unit test", {"native": str(native)})
    assert result["state"] == "deleted" and not target.exists()
    assert (archive/"metadata/manifest.json").exists()
    assert (native/"source.pt").read_text() == "untouched"


def test_recovery_rotation_preserves_two_and_requires_old_policy_and_signal(tmp_path, monkeypatch):
    import phase2.storage as storage
    parent = tmp_path/"checkpoints"; parent.mkdir()
    (parent/"latest_checkpointed_iteration.txt").write_text("38")
    for update in (36, 37, 38):
        ck = parent/f"global_step_{update}"; ck.mkdir(); (ck/"data.pt").write_text("checkpoint")
    model = tmp_path/"models/u0036"; model.mkdir(parents=True)
    (model/"phase2_export.json").write_text(json.dumps({"dtype": "float32"}))
    called = []
    monkeypatch.setattr(storage, "validate_full_checkpoint", lambda p: called.append(p.name))
    monkeypatch.setattr(storage, "validate_model_only", lambda p: None)
    config = {"post_updates": [36, 37, 38], "storage": {"rolling_recovery_authorized": True, "keep_full_recovery": 2}}
    with pytest.raises(ValueError, match="signals commit"): storage.rotate_recovery(tmp_path, config, 38)
    assert (parent/"global_step_36").exists()
    signal = tmp_path/"signals/u0036"; signal.mkdir(parents=True); (signal/"committed.json").write_text("{}")
    storage.rotate_recovery(tmp_path, config, 38)
    assert not (parent/"global_step_36").exists()
    assert (parent/"global_step_37").exists() and (parent/"global_step_38").exists()
    assert (model/"phase2_export.json").exists()
    assert set(called) == {"global_step_37", "global_step_38"}


def test_import_baseline_requires_matching_generation_protocol(tmp_path, monkeypatch):
    import phase2.import_baseline as module
    source = tmp_path/"source"; root = tmp_path/"target"
    source.mkdir(); root.mkdir()
    (source/"protocol.json").write_text(json.dumps({"parent_update": 35, "evaluation": {"temperature": .4}}))
    config = {"baseline_source": str(source), "baseline_protocol_sha256": sha256_file(source/"protocol.json"),
              "parent_update": 35, "evaluation": {"temperature": .7}}
    (root/"protocol.json").write_text(json.dumps(config))
    monkeypatch.setattr(module, "validate_extended", lambda *args: None)
    with pytest.raises(ValueError, match="Cannot reuse baseline"): module.import_baseline(root)


def test_completed_rotated_updates_resume_without_requiring_old_optimizer(tmp_path, monkeypatch):
    import phase2.run_extended as runner
    p = runner.ExtendedPipeline.__new__(runner.ExtendedPipeline)
    p.extended = True; p.root = tmp_path
    p.config = {"parent_update": 35, "post_updates": [36], "evaluation": {}, "windows": []}
    (tmp_path/"models/u0035").mkdir(parents=True)
    (tmp_path/"models/u0036").mkdir()
    (tmp_path/"models/u0036/phase2_export.json").write_text("{}")
    (tmp_path/"signals/u0036").mkdir(parents=True)
    (tmp_path/"signals/u0036/committed.json").write_text("{}")
    (tmp_path/"completed_updates.jsonl").write_text(json.dumps({"global_update": 36})+"\n")
    p.parallel = lambda *a: None
    p.status = lambda *a, **k: None; p.refresh_report = lambda: None
    monkeypatch.setattr(runner, "validate_full_checkpoint", lambda _: pytest.fail("old full checkpoint was intentionally rotated"))
    p.run()
