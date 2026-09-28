"""Opt-in serialization limits for NEW full-vocabulary rows only.

Legacy capture, optimizer batches, trajectories, witnesses and checkpoints are
unchanged. Compression overruns fail closed; they never truncate, lower dtype,
delete evidence, relax capacity limits or fall back to a lossy representation.
"""
from __future__ import annotations

from pathlib import Path

from .common import file_hash, read_json, write_new_bytes
from .lossless_tensor import encode, validate_profile


def settings(root):
    path = Path(root)/'resource_limits.json'
    if not path.exists():
        return None
    limits = read_json(path)
    if 'full_vocab_compression' not in limits:
        return None
    validate_profile(limits['full_vocab_compression'])
    for key in ('full_vocab_max_encoded_bytes_per_token', 'full_vocab_max_encoded_overhead_bytes',
                'minimum_free_bytes', 'maximum_run_bytes', 'checkpoint_reserve_bytes'):
        if type(limits.get(key)) is not int or limits[key] <= 0:
            raise ValueError(f'Explicit positive compressed capture limit required: {key}')
    return limits


def encoded_bound(limits, tokens, rows):
    if type(tokens) is not int or tokens < 0 or type(rows) is not int or rows < 0:
        raise ValueError('Exact token and row counts required for compressed capacity admission')
    return tokens * limits['full_vocab_max_encoded_bytes_per_token'] + rows * limits['full_vocab_max_encoded_overhead_bytes']


def save_row(root, path, value, limits):
    """Verify in memory, enforce admitted bounds, publish once under a disk lock."""
    import fcntl
    from .runtime import disk_gate
    root, path = Path(root).resolve(), Path(path).resolve()
    if (not path.is_relative_to(root) or path.relative_to(root).parts[0] not in ('old_logprobs', 'new_logprobs')
            or not path.name.startswith('row-') or path.suffix != '.pt'):
        raise ValueError('Compression is restricted to this run\'s temporary full-vocabulary rows')
    if path.exists():
        raise FileExistsError('Do not overwrite an existing full-vocabulary row')
    encoded, audit = encode(value, limits['full_vocab_compression'])
    maximum = encoded_bound(limits, len(value['log_probs']), 1)
    if len(encoded) > maximum:
        raise OSError('Compressed row exceeds measured admission bound; stop without lossy fallback or deletion')
    # Other ranks compress in parallel, but disk admission + publication is atomic
    # with respect to our own writers. Foreign disk usage remains a runtime risk.
    with (root/'.capture-write.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        disk_gate(root, len(encoded) + limits['checkpoint_reserve_bytes'],
                  minimum_free_bytes=limits['minimum_free_bytes'], maximum_run_bytes=limits['maximum_run_bytes'])
        if path.exists():
            raise FileExistsError('Concurrent capture already published this row')
        write_new_bytes(path, encoded)
        import hashlib
        if file_hash(path) != hashlib.sha256(encoded).hexdigest():
            raise OSError('Published compressed row differs from verified bytes; preserve file and stop')
    return {**audit, 'maximum_encoded_bytes': maximum, 'published_bytes_verified': True}
