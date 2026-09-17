#!/usr/bin/env python3
"""Check the exported project and report inputs without running experiments."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def relocated(path):
    return ROOT / Path(path).relative_to('/home/wangyifan/skill-RL')


def main():
    count = 0
    for line in (HERE/'project-original-files.jsonl').read_text().splitlines():
        item = json.loads(line)
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise SystemExit('Unsafe relative path')
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise SystemExit('Project content mismatch: ' + str(relative))
        count += 1
    results = []
    for revision in sorted((ROOT/'SkillRL/artifacts/phase2').glob('*/reports/*/revision.json')):
        record = json.loads(revision.read_text())
        items = record.get('source_files', []) + record.get('audit_files', []) + record.get('protected_files', [])
        for item in items:
            if sha256(relocated(item['path'])) != item['sha256']:
                raise SystemExit('Historical input hash mismatch: ' + item['path'])
        report_matches = sha256(relocated(record['report']['path'])) == record['report']['sha256']
        if revision.parent.name.endswith('synthesis-v2') and not report_matches:
            raise SystemExit('The protected synthesis-v2 report changed')
        results.append({'revision': revision.parent.name, 'input_hashes_verified': len(items),
                        'recorded_report_hash_matches_current_file': report_matches})
    print(json.dumps({'status': 'project_snapshot_verified', 'files': count, 'revisions': results,
                      'raw_trajectory_integrity_checked': False,
                      'note': 'v1 refers to an older report; v3 current-report mismatch is preserved, not repaired.'}, indent=2))


if __name__ == '__main__':
    main()
