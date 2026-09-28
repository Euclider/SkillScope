"""Publish a verified public handoff to GitHub without deleting remote files.

Dry-run is the default. Publishing requires the expected remote HEAD, a
separate release directory, and an explicit --publish switch. Credentials
are read from an authenticated GitHub CLI process and never logged.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path

import requests

from phase3.package import SECRET, scan
from scripts.package_full_research_handoff import (
    REDACTED_VENDOR_DOC, inventory, sha256,
)


MANIFEST_REMOTE_NAME = 'FULL_RESEARCH_RELEASE_MANIFEST.json'


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def verified_entries(project: Path, package: Path) -> dict[str, bytes]:
    """Read only files explicitly listed in the archive and verify their origin."""
    project, package = Path(project).resolve(), Path(package).resolve()
    manifest_path = package / 'RELEASE_MANIFEST.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema_version') != 'skillscope.public-research-handoff.v1':
        raise ValueError('Unsupported release manifest schema')
    source = dict((name, path) for path, name in inventory(project))
    listed = {row['path'] for row in manifest['files']}
    if listed != set(source) or len(listed) != len(manifest['files']):
        raise ValueError('Source inventory changed since package creation')
    staged = {path.relative_to(package).as_posix()
              for path in package.rglob('*') if path.is_file()}
    if staged != listed | {'RELEASE_MANIFEST.json'}:
        raise ValueError('Package contains unlisted or missing files')
    entries = {}
    for row in manifest['files']:
        name = row['path']
        data = (package / name).read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Package hash mismatch: {name}')
        original = source[name].read_bytes()
        if name == REDACTED_VENDOR_DOC:
            original = SECRET.sub(b'[REDACTED CREDENTIAL]', original)
        if data != original:
            raise ValueError(f'Source changed since package creation: {name}')
        scan(package / name)
        entries[name] = data
    entries[MANIFEST_REMOTE_NAME] = manifest_bytes
    return entries


def _api(session: requests.Session, base: str, method: str,
         path: str, payload: dict | None = None) -> dict:
    response = session.request(method, base + path, json=payload, timeout=180)
    if not 200 <= response.status_code < 300:
        raise RuntimeError(f'GitHub {method} {path}: HTTP {response.status_code}; '
                           f'{response.text[:240]}')
    return response.json()


def publish(project: Path, package: Path, *, repo: str, expected_head: str,
            gh_bin: str, receipt: Path, do_publish: bool) -> dict:
    entries = verified_entries(project, package)
    token = subprocess.check_output([gh_bin, 'auth', 'token'], text=True).strip()
    if not token:
        raise RuntimeError('GitHub CLI has no active credential')
    session = requests.Session()
    session.headers.update({'Authorization': 'Bearer ' + token,
                            'Accept': 'application/vnd.github+json',
                            'X-GitHub-Api-Version': '2022-11-28'})
    base = f'https://api.github.com/repos/{repo}'
    head = _api(session, base, 'GET', '/git/ref/heads/main')['object']['sha']
    if head != expected_head:
        raise RuntimeError('Remote main changed; refusing to publish from stale base')
    base_tree = _api(session, base, 'GET', '/git/commits/' + head)['tree']['sha']
    remote_tree = _api(session, base, 'GET', '/git/trees/' + base_tree + '?recursive=1')
    if remote_tree.get('truncated'):
        raise RuntimeError('Remote tree was truncated')
    current = {row['path']: row['sha'] for row in remote_tree['tree']
               if row['type'] == 'blob'}
    changed = [(name, data, blob_sha(data)) for name, data in entries.items()
               if current.get(name) != blob_sha(data)]
    summary = {'base_commit': head, 'package_files_verified': len(entries),
               'changed_existing': sum(name in current for name, _, _ in changed),
               'new_files': sum(name not in current for name, _, _ in changed),
               'upload_bytes': sum(len(data) for _, data, _ in changed),
               'publish_requested': do_publish}
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not do_publish:
        return summary
    if Path(receipt).exists():
        raise ValueError('Receipt path already exists; use a new release path')
    if not changed:
        raise RuntimeError('No changes to publish')
    known_blobs = set(current.values())
    working_tree = base_tree
    batch = []
    inline_bytes = 0

    def flush() -> None:
        nonlocal working_tree, batch, inline_bytes
        if batch:
            working_tree = _api(session, base, 'POST', '/git/trees',
                                {'base_tree': working_tree, 'tree': batch})['sha']
            batch, inline_bytes = [], 0

    for name, data, expected_sha in changed:
        staged_path = Path(package) / ('RELEASE_MANIFEST.json'
                                       if name == MANIFEST_REMOTE_NAME else name)
        row = {'path': name, 'mode': '100755' if os.access(staged_path, os.X_OK)
               else '100644', 'type': 'blob'}
        try:
            inline = data.decode('utf-8') if len(data) <= 64 * 1024 else None
        except UnicodeDecodeError:
            inline = None
        if expected_sha in known_blobs:
            row['sha'] = expected_sha
        elif inline is not None:
            row['content'] = inline
            inline_bytes += len(data)
        else:
            uploaded = _api(session, base, 'POST', '/git/blobs',
                            {'content': base64.b64encode(data).decode('ascii'),
                             'encoding': 'base64'})['sha']
            if uploaded != expected_sha:
                raise RuntimeError(f'GitHub blob hash mismatch: {name}')
            known_blobs.add(uploaded)
            row['sha'] = uploaded
        batch.append(row)
        if len(batch) >= 100 or inline_bytes >= 512 * 1024:
            flush()
    flush()
    commit = _api(session, base, 'POST', '/git/commits',
                  {'message': 'Publish complete SkillScope research handoff',
                   'tree': working_tree, 'parents': [head]})['sha']
    _api(session, base, 'PATCH', '/git/refs/heads/main',
         {'sha': commit, 'force': False})
    observed = _api(session, base, 'GET', '/git/ref/heads/main')['object']['sha']
    if observed != commit:
        raise RuntimeError('Remote main does not point at published commit')
    verified_tree = _api(session, base, 'GET',
                         '/git/trees/' + working_tree + '?recursive=1')
    if verified_tree.get('truncated'):
        raise RuntimeError('Published remote tree was truncated')
    after = {row['path']: row['sha'] for row in verified_tree['tree']
             if row['type'] == 'blob'}
    if any(after.get(name) != blob_sha(data) for name, data in entries.items()):
        raise RuntimeError('Published file hash audit failed')
    if any(after.get(name) != sha for name, sha in current.items()
           if name not in entries):
        raise RuntimeError('Remote files outside the package changed or disappeared')
    result = {**summary, 'commit': commit, 'tree': working_tree,
              'all_published_blob_hashes_match': True,
              'remote_files_deleted': False}
    receipt = Path(receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x') as stream:
        json.dump(result, stream, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'commit': commit, 'remote_blob_audit': 'PASS',
                      'receipt': str(receipt)}, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--repo', default='Euclider/SkillScope')
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--gh-bin', default='gh')
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    publish(args.project, args.package, repo=args.repo,
            expected_head=args.expected_head, gh_bin=args.gh_bin,
            receipt=args.receipt, do_publish=args.publish)


if __name__ == '__main__':
    main()
