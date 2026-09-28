"""Download pinned public inputs without overwriting existing data files."""
import argparse
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
DATA_URLS = (
    'https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip',
    'https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip',
    'https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip',
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=['model', 'router', 'data'])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    destination = a.output.resolve()
    if a.kind in ('model', 'router'):
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError('Use existing verified model or a new empty download directory')
        from huggingface_hub import snapshot_download
        if a.kind == 'router':
            from agent_system.memory.skillrl_embedding_router import MODEL_ID, MODEL_REVISION as ROUTER_REVISION, load_profile, verify_snapshot
            from agent_system.memory.skillnet_runtime import DEFAULT_EMBEDDING_ROUTER_PROFILE
            _, files = load_profile(DEFAULT_EMBEDDING_ROUTER_PROFILE)
            snapshot_download(MODEL_ID, revision=ROUTER_REVISION, local_dir=destination, allow_patterns=[*files, 'README.md'])
            verify_snapshot(destination, files)
            return
        snapshot_download('Qwen/Qwen3.5-4B', revision=MODEL_REVISION, local_dir=destination)
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='phase3-public-data-', dir=destination.parent) as temporary:
        for index, url in enumerate(DATA_URLS):
            archive = Path(temporary) / f'public-{index}.zip'
            urllib.request.urlretrieve(url, archive)
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    relative = Path(member.filename)
                    target = (destination / relative).resolve()
                    if relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(destination):
                        raise ValueError('Unsafe data archive member')
                    if (member.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('Symlink in data archive')
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.open(member) as source, target.open('xb') as out:
                            shutil.copyfileobj(source, out)
            archive.unlink()  # Only our downloaded temporary ZIP, never existing games.
    os.environ['ALFWORLD_DATA'] = str(destination)
    from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH
    (destination / 'logic').mkdir(exist_ok=True)
    for source, name in ((ALFRED_PDDL_PATH, 'alfred.pddl'), (ALFRED_TWL2_PATH, 'alfred.twl2')):
        target = destination / 'logic' / name
        if not target.exists():
            with open(source, 'rb') as inp, target.open('xb') as out:
                shutil.copyfileobj(inp, out)
    from skillnet_cohort.assets import game_inventory
    print({name: split['count'] for name, split in game_inventory(destination)['splits'].items()})


if __name__ == '__main__':
    main()
