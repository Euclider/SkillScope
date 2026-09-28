from pathlib import Path

import pytest

from skillnet_cohort.common import write_new_bytes
from skillnet_cohort.runtime import storage_bytes, disk_gate


@pytest.mark.parametrize('folder,filename', [('forward_progress', 'rank-7.json'), ('seed-404', 'runtime-status.json')])
def test_progress_atomic_rename_does_not_stop_capture(tmp_path, monkeypatch, folder, filename):
    progress = tmp_path / folder
    progress.mkdir()
    stable = progress / filename
    write_new_bytes(stable, b'old')
    temporary = progress / f'.{filename}.4x5c1u6s'
    write_new_bytes(temporary, b'new-progress')
    original = Path.lstat
    def racing_lstat(path):
        if path == temporary and temporary.exists():
            temporary.replace(stable)
        return original(path)
    monkeypatch.setattr(Path, 'lstat', racing_lstat)
    assert storage_bytes(tmp_path) in (3, 12)
    assert stable.read_bytes() == b'new-progress'
    assert disk_gate(tmp_path, minimum_free_bytes=1, maximum_run_bytes=100)['used_bytes'] == 12


def test_missing_tensor_and_permission_errors_still_fail_closed(tmp_path, monkeypatch):
    row = tmp_path / 'row-000000.pt'
    write_new_bytes(row, b'evidence')
    original = Path.lstat
    def disappearing(path):
        if path == row:
            raise FileNotFoundError(path)
        return original(path)
    monkeypatch.setattr(Path, 'lstat', disappearing)
    with pytest.raises(FileNotFoundError):
        storage_bytes(tmp_path)
    def denied(path):
        raise PermissionError(path)
    monkeypatch.setattr(Path, 'lstat', denied)
    with pytest.raises(PermissionError):
        storage_bytes(tmp_path)


def test_storage_does_not_follow_symlinks_or_relax_caps(tmp_path):
    owned = tmp_path / 'owned'
    owned.mkdir()
    write_new_bytes(owned / 'row.pt', b'12345')
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    write_new_bytes(foreign / 'model.pt', b'large model')
    (owned / 'model').symlink_to(foreign, target_is_directory=True)
    (owned / 'weights').symlink_to(foreign / 'model.pt')
    assert storage_bytes(owned) == 5
    with pytest.raises(OSError, match='Storage budget'):
        disk_gate(owned, 2, minimum_free_bytes=1, maximum_run_bytes=6)
