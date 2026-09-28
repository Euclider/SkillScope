"""Package the public SkillScope research handoff without runtime/raw data.

The selection is deliberately broader than the Phase1–2 portable source
archive: it includes project notes, figures, protocols, and compact result
tables. Model snapshots, secrets, trajectories, optimizer tensors, caches,
locks, and benchmark datasets are not publication material.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

from phase3.package import SECRET, scan


OMIT_DIRS = {'.git', '.venv', '.pytest_cache', '.mypy_cache', '.ruff_cache',
             '__pycache__', 'node_modules', 'outputs', 'verl.egg-info'}
RAW_ARTIFACT_DIRS = {'trajectories', 'rollouts', 'checkpoints', 'models',
                     'optimizer_steps', 'batches', 'pre_forward_batches',
                     'forward_outputs', 'old_logprobs', 'new_logprobs',
                     'forward_progress', 'rollout_progress', 'evaluations'}
OMIT_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.parquet',
                 '.sqlite3', '.db', '.pyc', '.part'}
OMIT_ARTIFACT_CSV = {'sign_null_draws.csv', 'raw_features_and_semantic_utility.csv'}
ROOT_FILES = {'.gitignore', '.gitattributes', 'PROJECT_SHA256SUMS'}
MAX_FILE_BYTES = 20 * 2**20
REDACTED_VENDOR_DOC = ('SkillRL/agent_system/environments/env_package/'
                       'webshop/webshop/README_INSTALL_ARM-MAC.md')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _walk(directory: Path, *, artifact: bool = False):
    if not directory.is_dir():
        return
    for base, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = sorted(name for name in dirs
                         if name not in OMIT_DIRS
                         and not name.endswith('.locks')
                         and (not artifact or name not in RAW_ARTIFACT_DIRS))
        for name in sorted(files):
            yield Path(base) / name


def _public_source(path: Path, relative: Path) -> bool:
    if path.name == 'chromedriver':
        return False
    if path.name.startswith('.env') or path.suffix.lower() in OMIT_SUFFIXES:
        return False
    if path.name.endswith('.lock') and not path.name.startswith('requirements'):
        return False
    if 'datasets' in relative.parts and 'docs' in relative.parts:
        return False
    return True


def _public_artifact(path: Path) -> bool:
    if path.suffix == '.md':
        return True
    if path.suffix == '.csv':
        return path.name not in OMIT_ARTIFACT_CSV
    if path.suffix == '.json':
        return (bool({'reports', 'metrics', 'manifests'} & set(path.parts))
                or any(word in path.stem.lower()
                       for word in ('summary', 'manifest', 'aggregate',
                                    'analysis', 'registry', 'metric')))
    return False


def inventory(project: Path) -> list[tuple[Path, str]]:
    """Return verified source paths paired with project-relative publish paths."""
    project = Path(project).resolve()
    found: dict[str, Path] = {}
    for path in sorted(project.iterdir()):
        if path.is_file() and (path.suffix == '.md' or path.name in ROOT_FILES):
            found[path.relative_to(project).as_posix()] = path
    for path in _walk(project / 'deploy'):
        if _public_source(path, path.relative_to(project)):
            found[path.relative_to(project).as_posix()] = path
    source = project / 'SkillRL'
    for path in _walk(source):
        relative = path.relative_to(project)
        if 'artifacts' in relative.parts or not _public_source(path, relative):
            continue
        found[relative.as_posix()] = path
    for path in _walk(source / 'artifacts', artifact=True):
        if _public_artifact(path):
            found[path.relative_to(project).as_posix()] = path
    for relative, path in found.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Unsafe publication source: {relative}')
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f'Publication source exceeds 20 MiB: {relative}')
    return [(found[name], name) for name in sorted(found)]


def make_package(project: Path, output: Path, archive: Path) -> dict:
    project = Path(project).resolve()
    output, archive = Path(output).resolve(), Path(archive).resolve()
    if output.is_relative_to(project) or archive.is_relative_to(project):
        raise ValueError('Package targets must be outside the project')
    if output.exists() or archive.exists():
        raise ValueError('Use new package paths; never overwrite an earlier handoff')
    selected = inventory(project)
    for path, relative in selected:
        if relative != REDACTED_VENDOR_DOC:
            scan(path)  # Never print credential contents.
    output.mkdir(parents=True)
    rows = []
    redacted_paths = []
    for path, relative in selected:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == REDACTED_VENDOR_DOC:
            original = path.read_bytes()
            redacted = SECRET.sub(b'[REDACTED CREDENTIAL]', original)
            target.write_bytes(redacted)
            if redacted != original:
                redacted_paths.append(relative)
        else:
            shutil.copy2(path, target)
        scan(target)
        rows.append({'path': relative, 'sha256': sha256(target),
                     'bytes': target.stat().st_size})
    manifest = {
        'schema_version': 'skillscope.public-research-handoff.v1',
        'source_scope': ('current working-tree bytes of approved project materials '
                         'except paths listed under redacted_paths'),
        'files': rows, 'source_file_count': len(rows),
        'redacted_paths': redacted_paths,
        'excluded': ['model snapshots/weights', 'credentials', 'runtime locks/caches',
                     'intermediate tensors', 'raw trajectories', 'benchmark datasets'],
        'non_alfworld_end_to_end_validated': False,
    }
    (output / 'RELEASE_MANIFEST.json').write_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                   separators=(',', ':')) + '\n', encoding='utf-8')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'w:gz') as bundle:
        bundle.add(output, arcname='SkillScope-research', recursive=True)
    return {'directory': str(output), 'archive': str(archive),
            'source_files': len(rows), 'archive_bytes': archive.stat().st_size,
            'archive_sha256': sha256(archive),
            'manifest_sha256': sha256(output / 'RELEASE_MANIFEST.json')}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_package(args.project, args.output, args.archive),
                     sort_keys=True))


if __name__ == '__main__':
    main()
