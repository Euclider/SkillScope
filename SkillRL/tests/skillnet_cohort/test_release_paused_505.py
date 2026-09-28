import pytest

from skillnet_cohort import release_paused_505 as r


@pytest.mark.parametrize('state', ['T', 'S', 'R'])
def test_release_validation_rejects_live_original(monkeypatch, state):
    monkeypatch.setattr(r, 'process_identity', lambda pid: {'start_ticks': 4, 'state': state})
    with pytest.raises(ProcessLookupError):
        r.released({'processes': [{'pid': 10, 'start_ticks': 4}]})


@pytest.mark.parametrize('current', [None, {'start_ticks': 4, 'state': 'Z'}, {'start_ticks': 5, 'state': 'S'}])
def test_gone_or_recycled_identity_never_signalled(monkeypatch, current):
    monkeypatch.setattr(r, 'process_identity', lambda pid: current)
    monkeypatch.setattr(r, 'gpu_users', lambda: [])
    r.released({'processes': [{'pid': 10, 'start_ticks': 4}]})


def test_gpu_release_required(monkeypatch):
    monkeypatch.setattr(r, 'process_identity', lambda pid: None)
    monkeypatch.setattr(r, 'gpu_users', lambda: [10])
    with pytest.raises(RuntimeError):
        r.released({'processes': [{'pid': 10, 'start_ticks': 4}]})


def test_no_repeat_of_release(monkeypatch, tmp_path):
    monkeypatch.setattr(r, 'OUTPUT', tmp_path)
    (tmp_path/'release-intent.json').write_text('{}')
    with pytest.raises(FileExistsError):
        r.release()
