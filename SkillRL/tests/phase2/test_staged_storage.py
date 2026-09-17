import json
from types import SimpleNamespace

import pytest

from phase2 import staged_storage as module


def test_peak_accounts_for_export_after_verified_rotation():
    assert module.staged_peak(100, 18, 50) == 82
    assert module.staged_peak(100, 18, 10) == 90
    assert module.staged_peak(100, 18, 0) == 100
    with pytest.raises(ValueError): module.staged_peak(10, 18, 50)


def fake_admission(tmp_path, monkeypatch, free):
    monkeypatch.setattr(module, "load_amendment", lambda *a: {"protocol_sha256": "frozen"})
    monkeypatch.setattr(module, "storage_inputs", lambda *a: {
        "deferred_export_bytes": 18*module.GIB, "reclaim_bytes": 50*module.GIB,
        "source_checkpoint_bytes": 50*module.GIB})
    monkeypatch.setattr(module.shutil, "disk_usage", lambda *a: SimpleNamespace(free=free*module.GIB))
    return {"storage": {"minimum_next_update_allowance_gib": 100,
                        "projected_transient_reserve_gib": 200}, "signals": {"vocab_size": 248320}}


def test_admission_keeps_original_total_budget_and_reserve(tmp_path, monkeypatch):
    config = fake_admission(tmp_path, monkeypatch, 293)
    result = module.admit(tmp_path, config, 40)
    assert result["peak_net_growth_bytes"] == 82*module.GIB
    assert result["total_write_allowance_bytes"] == 100*module.GIB
    assert result["reserve_bytes"] == 200*module.GIB
    assert (tmp_path/"storage_audit/admission-u0040-prelaunch.json").exists()


def test_admission_stops_when_external_disk_growth_consumes_reserve(tmp_path, monkeypatch):
    config = fake_admission(tmp_path, monkeypatch, 281)
    with pytest.raises(RuntimeError, match="reserve insufficient"):
        module.admit(tmp_path, config, 40)


def test_actual_token_capture_budget_can_reject_before_optimizer(tmp_path, monkeypatch):
    config = fake_admission(tmp_path, monkeypatch, 293)
    result = module.admit(tmp_path, config, 40, tokens=8000)
    assert result["live_old_new_bytes"] == 8000*248320*8
    with pytest.raises(RuntimeError, match="reserve insufficient"):
        module.admit(tmp_path, config, 40, tokens=60000)
    record = json.loads((tmp_path/"storage_audit/admission-u0040-actual_batch_before_old_forward.json").read_text())
    assert not record["admitted"] and record["response_tokens"] == 60000


def test_no_amendment_leaves_legacy_behavior_unchanged(tmp_path):
    assert module.load_amendment(tmp_path, 40) is None
    assert module.admit(tmp_path, {}, 40) is None


def test_amendment_refuses_changed_scientific_protocol(tmp_path):
    (tmp_path/"protocol.json").write_text("{}")
    (tmp_path/module.AMENDMENT).write_text(json.dumps({"root": str(tmp_path), "protocol_sha256": "wrong"}))
    with pytest.raises(ValueError, match="immutable protocol"):
        module.load_amendment(tmp_path, 40)


def test_native_source_never_removed_by_registered_scratch_cleanup(tmp_path, monkeypatch):
    from phase2 import storage
    monkeypatch.setattr(storage, "validate_full_checkpoint", lambda *a: None)
    scratch = tmp_path/"scratch"; scratch.mkdir()
    monkeypatch.setattr(module, "load_amendment", lambda *a: {"scratch_parent": str(scratch)})
    native = tmp_path/"checkpoints/global_step_39"; native.mkdir(parents=True)
    allocations = tmp_path/"allocations"; allocations.mkdir()
    (allocations/"u0040.json").write_text(json.dumps({"resume_checkpoint": str(native)}))
    storage.release_run_conversion(tmp_path, {"storage": {"release_completed_conversions": True}}, 40)
    assert native.exists()


def test_scratch_conversion_cleanup_keeps_native_and_archives_metadata(tmp_path, monkeypatch):
    from phase2 import storage
    monkeypatch.setattr(storage, "validate_full_checkpoint", lambda *a: None)
    scratch = tmp_path/"scratch"; scratch.mkdir()
    monkeypatch.setattr(module, "load_amendment", lambda *a: {"scratch_parent": str(scratch)})
    native = tmp_path/"checkpoints/global_step_39"; native.mkdir(parents=True)
    target = scratch/"u0039-w5/global_step_39"; (target/"actor").mkdir(parents=True)
    (target/"actor/elastic_resume.json").write_text(json.dumps({"source_checkpoint": str(native)}))
    allocations = tmp_path/"allocations"; allocations.mkdir()
    (allocations/"u0040.json").write_text(json.dumps({"resume_checkpoint": str(target)}))
    storage.release_run_conversion(tmp_path, {"storage": {"release_completed_conversions": True}}, 40)
    assert native.exists() and not target.parent.exists()
    assert (tmp_path/"storage_audit/conversion-used-u0040/receipt.json").exists()
