"""Create a no-clobber, secret-scanned Phase1–2 source handoff archive.

No model/data/experiment artifacts are copied. The resulting manifest records
the exact current working-tree bytes, including legitimate uncommitted edits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from phase3.package import inventory as shared_inventory, scan
from phase3.common import require
from phase3.prepare import ROOT


SUFFIXES = {'.py', '.yaml', '.yml', '.json', '.md', '.txt', '.sh', '.toml',
            '.cfg', '.ini', '.in', '.lock', '.rst', '.jinja', '.pddl',
            '.twl2', '.lark', '.cu', '.cuh', '.cpp', '.h', '.hpp'}
OMIT = {'__pycache__', '.pytest_cache', 'artifacts', 'outputs', '.git',
        '.venv', 'node_modules'}
EXTRA_DIRECTORIES = ('tests/phase1', 'tests/phase2', 'tests/skillnet_cohort',
                     'examples/grpo_trainer', 'environment', 'docs/phase12',
                     'gigpo')
EXTRA_FILES = ('requirements-vllm-phase12.in', 'requirements-vllm-phase12.lock',
               'requirements-skillnet-phase12.txt', 'scripts/package_phase12_portable.py')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sources():
    found = set(shared_inventory())
    for directory in EXTRA_DIRECTORIES:
        for path in (ROOT / directory).rglob('*'):
            if path.is_symlink():
                raise ValueError(f'Symlink in handoff source: {path.relative_to(ROOT)}')
            if path.is_file() and not set(path.relative_to(ROOT).parts) & OMIT and path.suffix in SUFFIXES:
                found.add(path)
    found.update(ROOT / name for name in EXTRA_FILES)
    for path in found:
        require(path.is_file() and path.is_relative_to(ROOT) and not path.is_symlink(),
                f'Missing or unsafe handoff source: {path}')
        require(path.stat().st_size < 10 * 2**20, f'Unexpected large handoff source: {path.name}')
    return sorted(found)


def make_package(output, archive):
    output, archive = Path(output).resolve(), Path(archive).resolve()
    require(not output.is_relative_to(ROOT) and not archive.is_relative_to(ROOT),
            'Package targets must be outside the source checkout')
    require(not output.exists() and not archive.exists(),
            'Use new package paths; never overwrite a prior handoff')
    selected = sources()
    for path in selected:
        scan(path)  # Never print the matched credential or its surrounding text.
    output.mkdir(parents=True)
    for source in selected:
        target = output / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(ROOT / 'docs/phase12/CROSS_BENCHMARK_PORTABILITY.md', output / 'README.md')
    rows = [{'path': path.relative_to(output).as_posix(), 'sha256': sha256(path),
             'bytes': path.stat().st_size} for path in sorted(output.rglob('*')) if path.is_file()]
    manifest = {'schema_version': 'skillscope.phase12.portable-source.v1',
                'version_label': 'phase12-crossbench-source-2026-09-28',
                'source_scope': 'current working-tree bytes of allowlisted source and environment files',
                'files': rows, 'source_file_count': len(selected),
                'model_weights_included': False, 'benchmark_data_included': False,
                'run_artifacts_included': False, 'credentials_included': False,
                'non_alfworld_end_to_end_validated': False,
                'requires_benchmark_adapter': True}
    (output / 'RELEASE_MANIFEST.json').write_text(
        json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'w:gz') as bundle:
        bundle.add(output, arcname='SkillScope-phase12', recursive=True)
    return {'directory': str(output), 'archive': str(archive),
            'source_files': len(selected), 'archive_bytes': archive.stat().st_size,
            'archive_sha256': sha256(archive), 'manifest_sha256': sha256(output / 'RELEASE_MANIFEST.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_package(args.output, args.archive), sort_keys=True))


if __name__ == '__main__':
    main()
