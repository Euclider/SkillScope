"""Recovery must accept documented retention, never arbitrary missing evidence."""
import json
from pathlib import Path

import pytest

from phase3.common import ProtocolError, digest
from skillnet_cohort.common import file_hash
from scripts import resume_two_arm_u50_editor, resume_two_arm_u50_retention


@pytest.fixture(params=[resume_two_arm_u50_editor, resume_two_arm_u50_retention])
def launcher(request):
    return request.param


@pytest.fixture
def retired(tmp_path, monkeypatch, launcher):
    root = tmp_path / 'experiment'
    run = root / 'runs/skillrl_failure'
    event = run / 'events/u0040'
    event.mkdir(parents=True)
    complete = {'event_id': 'u0040', 'outcome': 'accepted'}
    (event / 'complete.json').write_text(json.dumps(complete))
    identity = {'start': 35, 'end': 40, 'old_policy_sha256': 'old',
                'new_policy_sha256': 'new', 'branch_id': 'skillrl_failure'}
    (event / 'identity.json').write_text(json.dumps(identity))
    receipt = event / 'retention_complete.json'
    receipt.write_text(json.dumps({'event_complete_sha256': digest(complete),
        'old_policy_sha256': 'old', 'new_policy_sha256': 'new',
        'targets': ['checkpoints/global_step_35', 'models/u0035'],
        'targets_absent': True}))
    missing = run / 'models/u0035/phase2_export.json'
    numeric = tmp_path / 'numeric.json'
    numeric.write_text(json.dumps({'passed': True, 'source_hashes': {}}))
    authority = tmp_path / 'authorization.json'
    record = {'run_root': str(root), 'order': list(launcher.ORDER),
        'starts': launcher.STARTS, 'stop_update': 50, 'sources': {},
        'evidence': {str(missing): 'original-file-hash'},
        'retired_evidence': {str(missing): {'receipt': str(receipt),
                                         'receipt_sha256': file_hash(receipt)}},
        'numeric_receipt': str(numeric), 'candidate': str(tmp_path),
        'additional_dependencies': {}}
    authority.write_text(json.dumps(record))
    monkeypatch.setattr(launcher, 'ROOT', root)
    monkeypatch.setattr(launcher, 'AUTHORITY', authority)
    return authority, record, missing, receipt


def test_sealed_retired_export_does_not_block_next_window(retired, launcher):
    authority, record, _, _ = retired
    assert launcher.authority() == record


def test_undocumented_missing_export_is_rejected(retired, launcher):
    authority, record, _, _ = retired
    record['retired_evidence'] = {}
    authority.write_text(json.dumps(record))
    with pytest.raises((ProtocolError, FileNotFoundError)):
        launcher.authority()


def test_changed_retirement_receipt_is_rejected(retired, launcher):
    _, _, _, receipt = retired
    receipt.write_text('{}')
    with pytest.raises((ProtocolError, FileNotFoundError)):
        launcher.authority()


def test_unsealed_event_cannot_authorize_retirement(retired, launcher):
    _, _, _, receipt = retired
    (receipt.parent / 'complete.json').write_text('{}')
    with pytest.raises((ProtocolError, FileNotFoundError)):
        launcher.authority()


def test_present_but_changed_export_is_not_excused_by_receipt(retired, launcher):
    _, _, missing, _ = retired
    missing.parent.mkdir(parents=True)
    missing.write_text('{}')
    with pytest.raises(ProtocolError):
        launcher.authority()


def test_missing_non_export_file_is_never_excused(retired, launcher):
    authority, record, missing, _ = retired
    foreign = missing.parents[2] / 'metrics/u0035.json'
    record['evidence'] = {str(foreign): 'original-file-hash'}
    record['retired_evidence'][str(foreign)] = record['retired_evidence'].pop(str(missing))
    authority.write_text(json.dumps(record))
    with pytest.raises((ProtocolError, FileNotFoundError)):
        launcher.authority()
