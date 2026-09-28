import pytest
import torch

from skillnet_cohort.capture_storage import encoded_bound, save_row, settings
from skillnet_cohort.common import file_hash, write_new_json
from skillnet_cohort.lossless_tensor import bitwise_equal, load, shuffled_profile


def limits(tmp_path):
    value = {'full_vocab_compression': shuffled_profile(),
        'full_vocab_max_encoded_bytes_per_token': 5000, 'full_vocab_max_encoded_overhead_bytes': 10000,
        'minimum_free_bytes': 1, 'maximum_run_bytes': 1024**3, 'checkpoint_reserve_bytes': 1}
    write_new_json(tmp_path/'resource_limits.json', value)
    return value


def test_only_temporary_vocab_rows_compressed_and_no_overwrite(tmp_path):
    config = limits(tmp_path)
    value = {'log_probs': torch.arange(32*256, dtype=torch.float32).reshape(32, 256), 'row_index': 4}
    path = tmp_path/'old_logprobs/u0001/row-000004.pt'
    result = save_row(tmp_path, path, value, config)
    assert result['bitwise_roundtrip_verified'] and result['published_bytes_verified']
    assert bitwise_equal(value, load(path))
    before = file_hash(path)
    with pytest.raises(FileExistsError):
        save_row(tmp_path, path, value, config)
    assert file_hash(path) == before
    with pytest.raises(ValueError, match='restricted'):
        save_row(tmp_path, tmp_path/'batches/u0001/training_batch.pt', value, config)


def test_missing_opt_in_preserves_legacy_and_incomplete_settings_fail(tmp_path):
    assert settings(tmp_path) is None
    write_new_json(tmp_path/'resource_limits.json', {'full_vocab_compression': shuffled_profile()})
    with pytest.raises(ValueError, match='positive'):
        settings(tmp_path)


def test_size_bound_and_disk_guard_fail_without_publishing_or_deleting(tmp_path, monkeypatch):
    config = limits(tmp_path)
    value = {'log_probs': torch.ones(3, 40)}
    path = tmp_path/'new_logprobs/u0001/row-000001.pt'
    with pytest.raises(OSError, match='measured admission'):
        save_row(tmp_path, path, value, {**config, 'full_vocab_max_encoded_bytes_per_token': 1,
                                       'full_vocab_max_encoded_overhead_bytes': 1})
    assert not path.exists()
    def deny(*a, **k):
        raise OSError('disk protected')
    monkeypatch.setattr('skillnet_cohort.runtime.disk_gate', deny)
    with pytest.raises(OSError, match='disk protected'):
        save_row(tmp_path, path, value, config)
    assert not path.exists() and (tmp_path/'resource_limits.json').exists()


def test_encoded_bound_requires_exact_rows_and_tokens(tmp_path):
    config = limits(tmp_path)
    assert encoded_bound(config, 7, 2) == 55000
    with pytest.raises(ValueError):
        encoded_bound(config, 7, None)
