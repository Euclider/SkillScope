"""Explicit allowlist export, integrity manifest and fail-closed secret scan.

Never uses the source Git index, never copies .git, credentials or run outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import tarfile
from pathlib import Path

from skillnet_cohort.common import file_hash, write_new_bytes
from .common import digest, require, strict_json, write_new
from .prepare import ROOT

SECRET = re.compile(rb'(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
PLACEHOLDER = re.compile(rb'sk-(?:x{16,}|test[-_][A-Za-z0-9_-]+|placeholder[-_][A-Za-z0-9_-]+)\Z', re.I)
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.venv', 'node_modules', 'artifacts', 'outputs'}
EXCLUDED_FILES = {'verl/workers/critic/dp_critic.py'}  # Existing unfinished optional critic; GRPO never initializes it.
SUFFIXES = {'.py', '.yaml', '.yml', '.json', '.md', '.txt', '.sh', '.toml', '.cfg', '.ini', '.pddl', '.twl2', '.lark'}


def inventory():
    directories = ['phase3', 'skillnet_cohort', 'phase1', 'phase2', 'verl', 'agent_system/memory',
        'agent_system/multi_turn_rollout', 'agent_system/reward_manager', 'agent_system/environments/env_package/alfworld',
        'agent_system/environments/prompts', 'memory_data/alfworld/skillnet37', 'configs', 'tests/phase3',
        'tests/skill_router', 'tests/experiment_settings', 'tests/skill_bank']
    files = set()
    for name in directories:
        for path in (ROOT / name).rglob('*'):
            relative = path.relative_to(ROOT)
            if path.is_symlink():
                raise ValueError('Symlink in source allowlist')
            if path.is_file() and not set(relative.parts) & EXCLUDED and relative.as_posix() not in EXCLUDED_FILES:
                if path.suffix in SUFFIXES or path.name in ('LICENSE', 'LICENSE.txt', 'version', 'Notice.txt'):
                    files.add(path)
    for name in ['agent_system', 'agent_system/environments', 'agent_system/environments/env_package']:
        files.update((ROOT / name).glob('*.py'))
    for name in ['LICENSE', 'Notice.txt', 'setup.py', 'pyproject.toml', 'requirements.txt',
                 'requirements-router.txt', 'requirements-skillnet-phase12.txt', 'requirements-phase3.txt', 'requirements-skillrl-embedding-router.txt',
                 'scripts/model_merger.py', 'scripts/inspect_skillrl_alignment.py', 'scripts/test_skillnet_router.py',
                 'scripts/test_skillrl_embedding_router.py',
                 'docs/experiments/skillrl-alignment-v1/upstream_sources.json',
                 'memory_data/alfworld/claude_style_skills.json']:
        files.add(ROOT / name)
    return sorted(files)


def scan(path):
    data = path.read_bytes()
    for match in SECRET.finditer(data):
        if not PLACEHOLDER.fullmatch(match.group()):
            # Never print a match or nearby text into terminal, logs or manifest.
            raise ValueError(f'Possible credential blocks export: {path.name}; inspect privately')


def package(output, archive):
    output, archive = Path(output).resolve(), Path(archive).resolve()
    require(not output.is_relative_to(ROOT) and not archive.is_relative_to(ROOT), 'Export outside original source workspace')
    require(not output.exists() and not archive.exists(), 'Export/archive must be new paths')
    source_files = inventory()
    for path in source_files:
        scan(path)
        require(path.stat().st_size < 10*2**20, 'Unexpected large file in source allowlist')
    output.mkdir(parents=True)
    for path in source_files:
        target = output / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for source, target in [('PHASE2_ALIGNMENT_V3.md', 'README.md'),
                           ('PHASE2_ALIGNMENT_V3.md', 'PHASE2_ALIGNMENT_V3.md'),
                           ('PHASE2_ALIGNMENT_V2.md', 'PHASE2_ALIGNMENT_V2.md'), ('START.md', 'START-v1.md'),
                           ('SETTING.md', 'SETTING-v1.md'), ('VALIDATION.md', 'VALIDATION.md'),
                           ('EDITOR_MODEL_SWITCH_2026-09-26.md', 'EDITOR_MODEL_SWITCH_2026-09-26.md'),
                           ('EDITOR_CURRENT_POLICY_EVIDENCE_V11.md', 'EDITOR_CURRENT_POLICY_EVIDENCE_V11.md'),
                           ('EDITOR_PREUPDATE_TRAIN_EVIDENCE_V12.md', 'EDITOR_PREUPDATE_TRAIN_EVIDENCE_V12.md'),
                           ('EDITOR_OLD_POLICY_BATCH_EVIDENCE_V13.md', 'EDITOR_OLD_POLICY_BATCH_EVIDENCE_V13.md')]:
        path = ROOT / 'docs/phase3' / source
        scan(path)
        shutil.copy2(path, output / target)
    write_new_bytes(output / '.gitignore', b'.venv/\n__pycache__/\n.pytest_cache/\n.env*\n*.sqlite3*\n*.pt\n*.safetensors\nruns/\nartifacts/\nphase3-assets*/\n')
    rows = [{'path': path.relative_to(output).as_posix(), 'sha256':file_hash(path), 'bytes':path.stat().st_size}
            for path in sorted(output.rglob('*')) if path.is_file()]
    manifest = {'schema_version':'skillrl.phase3.release.v1', 'files':rows, 'inventory_sha256':digest(rows),
                'source_git_index_modified':False, 'weights_or_experiment_data_included':False,
                'secret_scan':'passed', 'source_bytes_preserved':True,
                'excluded_optional_modules': sorted(EXCLUDED_FILES), 'supported_training_algorithm':'GRPO'}
    write_new(output / 'RELEASE_MANIFEST.json', manifest)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'w:gz') as bundle:
        bundle.add(output, arcname='SkillScope-phase3', recursive=True)
    return {'files':len(rows)+1, 'bytes':sum(row['bytes'] for row in rows), 'inventory_sha256':digest(rows),
            'archive_sha256':file_hash(archive), 'directory':str(output), 'archive':str(archive)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    a = p.parse_args()
    print(package(a.output, a.archive))


if __name__ == '__main__':
    main()
