"""Lossless serialization for opt-in new full-vocabulary capture only.

No quantization, reduced vocabulary, token sampling or changed tensor values.
The ZIP candidate stays torch.load-compatible. The stronger shuffle/LZMA wrapper
uses load() below, with legacy Torch fallback. Both reconstruct every original
FP32 bit and metadata field. No dtype conversion or value approximation occurs.
"""
from __future__ import annotations

from copy import deepcopy
import io
import hashlib
import lzma
from pathlib import Path
import struct
import zipfile


PROFILE = {'codec': 'torch_zip_deflate_v1', 'level': 1, 'roundtrip_bitwise_verify': True}
SHUFFLE_PROFILE = {'codec': 'torch_shuffle4_lzma_v1', 'preset': 0,
                   'shuffle_width_bytes': 4, 'roundtrip_bitwise_verify': True}
MAGIC = b'SSFP4XZ1'
HEADER = struct.Struct('<8sQ32s')
MAX_PLAIN_BYTES = 1024**3  # More than one full 512 x 248320 FP32 row plus metadata.


def profile():
    return deepcopy(PROFILE)


def shuffled_profile():
    return deepcopy(SHUFFLE_PROFILE)


def validate_profile(value):
    if value not in (PROFILE, SHUFFLE_PROFILE):
        raise ValueError('Unknown lossless full-vocabulary serialization profile')


def bitwise_equal(before, after):
    import torch
    if isinstance(before, torch.Tensor):
        return (isinstance(after, torch.Tensor) and before.dtype == after.dtype
                and before.shape == after.shape and before.layout == after.layout
                and before.device.type == after.device.type == 'cpu'
                and torch.equal(before.detach().contiguous().reshape(-1).view(torch.uint8),
                                after.detach().contiguous().reshape(-1).view(torch.uint8)))
    if type(before) is not type(after):
        return False
    if isinstance(before, dict):
        return before.keys() == after.keys() and all(bitwise_equal(value, after[key]) for key, value in before.items())
    if isinstance(before, (tuple, list)):
        return len(before) == len(after) and all(bitwise_equal(x, y) for x, y in zip(before, after))
    return before == after


def encode(value, settings):
    """Return verified bytes; no disk writes or modification of the input."""
    import torch
    validate_profile(settings)
    plain = io.BytesIO()
    torch.save(value, plain)
    raw_bytes = plain.tell()
    if settings == SHUFFLE_PROFILE:
        import numpy as np
        raw = plain.getvalue()
        if raw_bytes > MAX_PLAIN_BYTES:
            raise ValueError('Full-vocabulary row exceeds lossless container size limit')
        padded = raw + b'\0' * (-raw_bytes % 4)
        shuffled = np.frombuffer(padded, dtype=np.uint8).reshape(-1, 4).T.copy().tobytes()
        checksum = hashlib.sha256(raw).digest()
        encoded = HEADER.pack(MAGIC, raw_bytes, checksum) + lzma.compress(shuffled, preset=0)
        restored = decode(encoded)
        if not bitwise_equal(value, restored):
            raise ValueError('Lossless serialization did not preserve exact tensor bits/metadata')
        return encoded, {'codec': settings['codec'], 'plain_torch_save_bytes': raw_bytes,
                         'plain_sha256': checksum.hex(), 'encoded_bytes': len(encoded),
                         'encoded_sha256': hashlib.sha256(encoded).hexdigest(),
                         'bitwise_roundtrip_verified': True}
    plain.seek(0)
    compressed = io.BytesIO()
    with zipfile.ZipFile(plain, 'r') as source:
        with zipfile.ZipFile(compressed, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=settings['level']) as target:
            for entry in source.infolist():
                target.writestr(entry.filename, source.read(entry.filename),
                                compress_type=zipfile.ZIP_DEFLATED, compresslevel=settings['level'])
    encoded = compressed.getvalue()
    restored = torch.load(io.BytesIO(encoded), map_location='cpu', weights_only=False)
    if not bitwise_equal(value, restored):
        raise ValueError('Lossless serialization did not preserve exact tensor bits/metadata')
    return encoded, {'codec': settings['codec'], 'plain_torch_save_bytes': raw_bytes,
                     'encoded_bytes': len(encoded), 'bitwise_roundtrip_verified': True}


def decode(encoded):
    """Strict bounded decode; verify the complete original archive before loading."""
    import numpy as np
    import torch
    if len(encoded) < HEADER.size:
        raise ValueError('Truncated lossless tensor header')
    magic, length, checksum = HEADER.unpack(encoded[:HEADER.size])
    if magic != MAGIC or not 0 < length <= MAX_PLAIN_BYTES:
        raise ValueError('Unknown or oversized lossless tensor container')
    expected = length + (-length % 4)
    decoder = lzma.LZMADecompressor(memlimit=64 * 1024**2)
    try:
        shuffled = decoder.decompress(encoded[HEADER.size:], max_length=expected + 1)
    except lzma.LZMAError as error:
        raise ValueError('Corrupt lossless tensor stream') from error
    if len(shuffled) != expected or not decoder.eof or decoder.unused_data:
        raise ValueError('Truncated, oversized or trailing lossless tensor data')
    raw = np.frombuffer(shuffled, dtype=np.uint8).reshape(4, -1).T.copy().tobytes()[:length]
    if hashlib.sha256(raw).digest() != checksum:
        raise ValueError('Lossless tensor SHA-256 mismatch')
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)


def load(path):
    """Read new compressed rows or unchanged historical Torch archives."""
    import torch
    with Path(path).open('rb') as stream:
        if stream.read(len(MAGIC)) == MAGIC:
            stream.seek(0)
            return decode(stream.read())
    return torch.load(path, map_location='cpu', weights_only=False)
