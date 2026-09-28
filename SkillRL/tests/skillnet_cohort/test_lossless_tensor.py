import io

import pytest
import torch

from skillnet_cohort.lossless_tensor import bitwise_equal, decode, encode, load, profile, shuffled_profile


def test_compressed_archive_is_native_torch_load_compatible_and_exact():
    values = torch.tensor([0., -0., float('nan'), float('inf'), -float('inf'), 1.], dtype=torch.float32)
    payload = {'row_index': 4, 'log_probs': values.repeat(1000, 1),
               'positions': torch.arange(1000), 'token_ids': torch.arange(1000),
               'nested': [True, None, 'exact', (4, 2.)]}
    encoded, audit = encode(payload, profile())
    restored = torch.load(io.BytesIO(encoded), map_location='cpu', weights_only=False)
    assert bitwise_equal(payload, restored)
    assert audit['encoded_bytes'] < audit['plain_torch_save_bytes']
    assert audit['bitwise_roundtrip_verified']


def test_bitwise_check_distinguishes_signed_zero_and_dtype_changes():
    assert not bitwise_equal(torch.tensor([0.]), torch.tensor([-0.]))
    assert not bitwise_equal(torch.tensor([1.]), torch.tensor([1.], dtype=torch.float64))


def test_unknown_or_nonverifying_profiles_are_rejected():
    for change in ({'level': 9}, {'roundtrip_bitwise_verify': False}, {'codec': 'lossy'}):
        with pytest.raises(ValueError, match='Unknown'):
            encode({'log_probs': torch.ones(3)}, {**profile(), **change})


def test_profile_returns_independent_copy():
    changed = profile()
    changed['level'] = 3
    assert profile()['level'] == 1


def test_shuffled_fp32_bits_metadata_and_legacy_file_loading(tmp_path):
    # Preserve NaN payload bits, signed zeros, infinities, subnormals, and strided tensors.
    bits = torch.tensor([0, -2147483648, 2143289345, 2143289346, 2139095040,
                         -8388608, 1, 8388607], dtype=torch.int32)
    value = {'log_probs': bits.view(torch.float32).repeat(32, 1)[:, ::2],
             'ids': torch.arange(20), 'empty': torch.empty(0, 7), 'metadata': ['exact', 5, None]}
    encoded, audit = encode(value, shuffled_profile())
    assert bitwise_equal(value, decode(encoded))
    assert audit['bitwise_roundtrip_verified']
    from skillnet_cohort.common import write_new_bytes
    new, old = tmp_path/'new.pt', tmp_path/'old.pt'
    write_new_bytes(new, encoded)
    torch.save(value, old)
    assert bitwise_equal(value, load(new)) and bitwise_equal(value, load(old))


@pytest.mark.parametrize('mutation', ['truncate', 'flip', 'append', 'sha', 'oversize'])
def test_corrupt_or_unbounded_stream_is_rejected(mutation):
    from skillnet_cohort.lossless_tensor import HEADER, MAGIC, MAX_PLAIN_BYTES
    encoded, _ = encode({'log_probs': torch.ones(4, 7)}, shuffled_profile())
    if mutation == 'truncate':
        encoded = encoded[:-5]
    elif mutation == 'flip':
        encoded = encoded[:60] + bytes([encoded[60] ^ 4]) + encoded[61:]
    elif mutation == 'append':
        encoded += b'extra'
    elif mutation == 'sha':
        encoded = encoded[:20] + bytes([encoded[20] ^ 4]) + encoded[21:]
    else:
        encoded = HEADER.pack(MAGIC, MAX_PLAIN_BYTES+1, bytes(32)) + encoded[HEADER.size:]
    with pytest.raises(ValueError):
        decode(encoded)
