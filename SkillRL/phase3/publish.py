"""Publish a verified export to the user-selected GitHub repository.

Uses authenticated gh, never reads/logs its token and never touches source Git.
No force push, branch deletion, issue/PR creation or repository setting change.
"""
import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

from .common import digest, require, strict_json, write_new
from .package import scan
from skillnet_cohort.common import file_hash

REPOSITORY = 'Euclider/SkillScope-phase3'


def api(endpoint, payload=None, method=None, optional=False):
    command = ['gh', 'api', endpoint]
    if method:
        command += ['--method', method]
    if payload is not None:
        command += ['--input', '-']
    result = subprocess.run(command, input=None if payload is None else json.dumps(payload),
                            text=True, capture_output=True, check=False)
    if result.returncode:
        if optional:
            return None
        raise RuntimeError('GitHub API operation failed; no credentials or response body logged')
    return json.loads(result.stdout)


def publish(directory, receipt, expected_parent=None):
    directory = Path(directory).resolve()
    manifest = strict_json((directory / 'RELEASE_MANIFEST.json').read_text())
    require(manifest['inventory_sha256'] == digest(manifest['files']), 'Changed export inventory')
    expected = {row['path'] for row in manifest['files']} | {'RELEASE_MANIFEST.json'}
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob('*') if path.is_file()}
    require(actual == expected and not any(path.is_symlink() for path in directory.rglob('*')), 'Unexpected export files/links')
    for row in manifest['files']:
        require(file_hash(directory / row['path']) == row['sha256'], 'Changed exported file')
    files = [directory / name for name in sorted(actual)]
    for path in files:
        scan(path)
    repo = api(f'repos/{REPOSITORY}')
    require(repo['full_name'] == REPOSITORY and repo['permissions']['push'], 'Wrong target or no push permission')
    head = api(f'repos/{REPOSITORY}/git/ref/heads/main', optional=True)
    if head is not None:
        require(expected_parent is not None and head['object']['sha'] == expected_parent,
                'Existing target requires the exact reviewed parent commit; no force/implicit overwrite')
        parent = expected_parent
    else:
        require(expected_parent is None and repo['size'] == 0, 'Cannot infer an empty target or missing branch')
        # Git Data API cannot create a tree in a completely empty repository.
        initial = api(f'repos/{REPOSITORY}/contents/README.md', {
            'message':'Initialize SkillScope Phase3 reproducibility package', 'branch':'main',
            'content':base64.b64encode((directory / 'README.md').read_bytes()).decode()}, method='PUT')
        parent = initial['commit']['sha']
    parent_commit = api(f'repos/{REPOSITORY}/git/commits/{parent}')
    previous = api(f'repos/{REPOSITORY}/git/trees/{parent_commit["tree"]["sha"]}?recursive=1')
    require(not previous['truncated'], 'Previous remote tree was truncated')
    require({row['path'] for row in previous['tree'] if row['type'] == 'blob'} <= expected,
            'Remote has files outside the new export; refusing to remove or overwrite unreviewed extras')
    entries, blobs = [], {}
    for path in files:
        data = path.read_bytes()
        name = path.relative_to(directory).as_posix()
        entries.append({'path':name, 'mode':'100755' if path.stat().st_mode & 0o111 else '100644',
                        'type':'blob', 'content':data.decode('utf-8')})
        blobs[name] = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    tree = api(f'repos/{REPOSITORY}/git/trees', {'base_tree':parent_commit['tree']['sha'], 'tree':entries}, method='POST')
    remote = api(f'repos/{REPOSITORY}/git/trees/{tree["sha"]}?recursive=1')
    require(not remote['truncated'], 'Remote tree verification was truncated')
    actual_blobs = {row['path']:row['sha'] for row in remote['tree'] if row['type']=='blob'}
    require(actual_blobs == blobs, 'Remote tree bytes differ from the verified export')
    commit = api(f'repos/{REPOSITORY}/git/commits', {'message':'Use versioned frozen SkillRL embedding routing across Phase3 branches; retain o3 editor',
        'tree':tree['sha'], 'parents':[parent]}, method='POST')
    # Non-force update rejects a concurrent sibling commit rather than losing it.
    api(f'repos/{REPOSITORY}/git/refs/heads/main', {'sha':commit['sha'], 'force':False}, method='PATCH')
    remote_head = api(f'repos/{REPOSITORY}/git/ref/heads/main')['object']['sha']
    require(remote_head == commit['sha'], 'Remote branch changed during publication')
    result = {'repository':REPOSITORY, 'commit':commit['sha'], 'tree':tree['sha'],
        'verified_files':len(files), 'inventory_sha256':manifest['inventory_sha256'],
        'all_remote_blob_hashes_match':True, 'source_repository_committed':False,
        'parent_commit':parent, 'force_push':False, 'remote_files_deleted':False}
    write_new(receipt, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--expected-parent', help='Required exact reviewed main commit for an existing repository')
    a = p.parse_args()
    require(a.execute, 'Explicit --execute required for GitHub publication')
    print(publish(a.directory, a.receipt, a.expected_parent))


if __name__ == '__main__':
    main()
