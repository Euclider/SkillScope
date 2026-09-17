#!/usr/bin/env python3
"""Download the pinned base model or upstream ALFWorld text data on the target."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
DATA_URLS = [
    'https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip',
    'https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip',
    'https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip',
]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def verify(root, category):
    errors, count = [], 0
    for line in (HERE/'downloaded-assets-reference.jsonl').read_text().splitlines():
        item = json.loads(line)
        if item['category'] != category:
            continue
        p = root / item['path']
        count += 1
        if not p.is_file() or p.stat().st_size != item['bytes'] or sha256(p) != item['sha256']:
            errors.append(item['path'])
    if errors:
        print(json.dumps({'status': 'download_reference_mismatch', 'category': category,
                          'mismatch_count': len(errors), 'first_mismatches': errors[:20]}, indent=2))
        raise SystemExit('Keep the mismatch report; do not rewrite the reference manifest.')
    print(json.dumps({'status': 'download_matches_source', 'category': category, 'files': count}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('asset', choices=['model', 'data'])
    parser.add_argument('--root', type=Path, default=Path('/mnt/workspace/users/wangyifan'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.asset == 'model':
        destination = args.root / 'model/Qwen3.5-4B'
        if not args.verify_only:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id='Qwen/Qwen3.5-4B', revision=MODEL_REVISION, local_dir=str(destination))
        verify(args.root, 'model')
        return
    destination = args.root / 'skill-RL/data/alfworld'
    if not args.verify_only:
        destination.mkdir(parents=True, exist_ok=True)
        os.environ['ALFWORLD_DATA'] = str(destination)
        with tempfile.TemporaryDirectory(prefix='alfworld-download-', dir=args.root) as temporary:
            for url in DATA_URLS:
                archive_path = Path(temporary) / url.rsplit('/', 1)[1]
                print('Downloading', url, flush=True)
                urllib.request.urlretrieve(url, archive_path)
                with zipfile.ZipFile(archive_path) as archive:
                    for member in archive.infolist():
                        relative = Path(member.filename)
                        if relative.is_absolute() or '..' in relative.parts:
                            raise RuntimeError('Unsafe upstream zip member')
                        if (member.external_attr >> 16) & 0o170000 == 0o120000:
                            raise RuntimeError('Unexpected symlink in upstream data archive')
                        target = destination / relative
                        if member.is_dir():
                            target.mkdir(parents=True, exist_ok=True)
                            continue
                        if target.exists():
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, target.open('xb') as output:
                            shutil.copyfileobj(source, output)
                archive_path.unlink()
        from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH
        logic = destination / 'logic'
        logic.mkdir(exist_ok=True)
        for source, filename in [(ALFRED_PDDL_PATH, 'alfred.pddl'), (ALFRED_TWL2_PATH, 'alfred.twl2')]:
            if not (logic / filename).exists():
                with open(source, 'rb') as stream, (logic / filename).open('xb') as output:
                    shutil.copyfileobj(stream, output)
    verify(args.root, 'alfworld-text')


if __name__ == '__main__':
    main()
