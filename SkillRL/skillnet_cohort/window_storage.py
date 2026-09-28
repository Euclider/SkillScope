"""Seal compact window evidence; delete only exact, independently replay-proven rows.

Intermediate weights are not silently assumed to exist. Absent a per-row
regeneration proof, retain the raw file and let the ordinary disk gate stop work.
"""
from __future__ import annotations

from pathlib import Path
import re

from .common import file_hash, read_json, write_new_json


def seal_window(root, start, end):
    from phase2.utilities import read_evaluations
    from phase2.protocol import signal_directory
    root = Path(root).resolve()
    config = read_json(root / 'protocol.json')
    rows = read_evaluations(root)
    if rows.empty or set(rows['update']) != {start, end}:
        raise ValueError('Both exact paired endpoint evaluation sets must be complete')
    signal = signal_directory(root, end, start)
    commit, prediction = read_json(signal / 'committed.json'), read_json(signal / 'prediction.json')
    if (commit['features_sha256'] != file_hash(signal / 'skill_context_features.parquet')
            or commit['token_signals_sha256'] != file_hash(signal / 'token_signals.parquet')
            or prediction['features_sha256'] != commit['features_sha256']
            or prediction['protocol_sha256'] != file_hash(root / 'protocol.json')
            or prediction['target_gold_read'] is not False):
        raise ValueError('Changed window commitment')
    if config['windows'] != [{'start': start, 'end': end, 'role': 'test'}]:
        raise ValueError('Not the registered fixed five-update window')
    if not (root / 'window_metrics/summary.json').is_file():
        raise ValueError('Compact report must be written before sealing')
    files = [root / 'protocol.json', root / 'manifest.json']
    for directory in ('audits', 'window_signals', 'signals', 'evaluations', 'window_metrics', 'reports'):
        files.extend(path for path in (root / directory).rglob('*') if path.is_file())
    if any(path.is_symlink() or not path.resolve().is_relative_to(root) for path in files):
        raise ValueError('Seal cannot follow external artifact links')
    record = {'schema_version': 'skillnet.window.seal.v1', 'start': start, 'end': end,
              'files': [{'path': str(path.relative_to(root)), 'sha256': file_hash(path)} for path in sorted(files)]}
    write_new_json(root / 'sealed.json', record)
    return record


def reclaim_proven_rows(run_root, windows, start, end, permit):
    run_root = Path(run_root).resolve()
    if (str(run_root) != permit.get('run_root') or permit.get('reclaim_regenerable_full_vocab') is not True
            or not (run_root / 'launch.json').is_file() or end != start + 5):
        raise PermissionError('Cleanup requires this explicitly authorized new run and window')
    for window in windows:
        window = Path(window).resolve()
        if not window.is_relative_to(run_root / 'windows'):
            raise ValueError('Foreign window seal')
        seal = read_json(window / 'sealed.json')
        if (seal['start'], seal['end']) != (start, end):
            raise ValueError('Wrong sealed window')
        for item in seal['files']:
            path = window / item['path']
            if not path.resolve().is_relative_to(window) or file_hash(path) != item['sha256']:
                raise ValueError('Changed sealed result')
    if not windows:
        raise ValueError('No sealed measurement windows; retain all tensors')
    deleted, retained = [], []
    proof_root = run_root / 'regeneration_proofs'
    for update in range(start + 1, end + 1):
        for stage in ('old', 'new'):
            directory = run_root / f'{stage}_logprobs' / f'u{update:04d}'
            if directory.is_symlink() or directory.parent.is_symlink():
                raise ValueError('Refuse cleanup through shared/historical source links')
            for path in sorted(directory.glob('row-*.pt')):
                if path.is_symlink() or not re.fullmatch(r'row-[0-9]{6}\.pt', path.name):
                    raise ValueError('Unexpected cleanup target')
                relative = path.relative_to(run_root).as_posix()
                proof_path = proof_root / f'{stage}-u{update:04d}-{path.stem}.json'
                if not proof_path.exists():
                    retained.append({'path': relative, 'reason': 'no independently verified regeneration proof'})
                    continue
                proof = read_json(proof_path)
                if (proof.get('target') != relative or proof.get('target_sha256') != file_hash(path)
                        or proof.get('bitwise_equal') is not True or not proof.get('sources')
                        or not isinstance(proof.get('recipe'), list) or not proof['recipe']):
                    raise ValueError('Invalid exact regeneration proof')
                for source in proof['sources']:
                    source_path = run_root / source['path']
                    # Only retained batches/models/checkpoints can establish regeneration.
                    if (Path(source['path']).parts[0] not in ('batches', 'models', 'checkpoints')
                            or not source_path.resolve().is_relative_to(run_root)
                            or source_path.is_symlink() or file_hash(source_path) != source['sha256']):
                        raise ValueError('Missing/changed retained regeneration source')
                item = {'path': relative, 'sha256': proof['target_sha256'], 'bytes': path.stat().st_size,
                        'regeneration_proof_sha256': file_hash(proof_path)}
                # Durable intent before unlink, scoped to one validated row only.
                write_new_json(run_root / 'reclamation' / f'{stage}-u{update:04d}-{path.stem}.json', item)
                path.unlink()
                deleted.append(item)
    result = {'start': start, 'end': end, 'deleted': deleted, 'retained': retained,
              'historical_files_or_checkpoints_removed': False}
    write_new_json(run_root / 'reclamation' / f'u{start:04d}-u{end:04d}.json', result)
    return result
