"""One user-authorized LOCAL reissue per frozen interrupted query, never an API retry.

Original attempts/decisions/hits remain byte-for-byte intact. A reissue appends
a new attempt and carries its authorization into the resulting cached record.
Ordinary routing, embedding numerics, cache keys and successful decisions do not
change. This adapter is installed only by the explicit evaluation entry below.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

from agent_system.memory.router_cache import RouterCache, RouterCacheError, RouterBudgetExceeded, canonical_json, digest, utc_now
from .common import REPO, file_hash, read_json

TABLES = {'protocol', 'decisions', 'attempts', 'cache_hits'}


def row_signature(row):
    return hashlib.sha256(canonical_json(list(row)).encode()).hexdigest()


@contextmanager
def read_only(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True, timeout=60)
    try:
        connection.execute('PRAGMA query_only=ON')
        yield connection
    finally:
        connection.close()


def local_protocol(protocol):
    return (protocol.get('version') == 'skillrl-embedding-state-batch-top1-v1'
        and protocol.get('config', {}).get('model') == 'Qwen/Qwen3-Embedding-0.6B'
        and protocol.get('device') == 'cpu' and protocol.get('external_api_calls') == 0
        and protocol.get('automatic_retries') == 0 and protocol.get('selection_count') == 1)


def load_authorization(path):
    from .seed_queue import verify_sources
    path = Path(path).resolve(); value = read_json(path); output = Path(value['recovery_root'])
    plan = read_json(output/'plan.json')
    if (path.parent != output or plan.get('approved') is not True
            or plan.get('router_resume') != {'path': str(path), 'sha256': file_hash(path)}
            or value.get('approved') is not True
            or value.get('schema_version') != 'skillnet.explicit_local_query_resume.v1'
            or value.get('external_api_calls') != 0 or value.get('max_reissues_per_key') != 1
            or len(value.get('allowed', [])) != 5
            or len({r['key'] for r in value['allowed']}) != 5):
        raise PermissionError('An explicit plan-bound five-query local-only authorization is required')
    verify_sources(plan)
    snapshot = Path(value['snapshot']['path'])
    if (snapshot.parent != output or snapshot.is_symlink()
            or file_hash(snapshot) != value['snapshot']['sha256']):
        raise ValueError('Original router ledger snapshot changed')
    value['_authorization_sha256'] = file_hash(path)
    return value


def audit_cache(path, authorization=None, *, check_all_decisions=True):
    """Read-only quiescent preflight, before any GPU work or query reissue.

    Reject unregistered interruptions. When a snapshot is supplied, every old
    row of all four tables must remain identical (new rows alone are allowed).
    """
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Router cache must not be a symlink')
    path = path.resolve()
    if not path.exists() and authorization is None:
        return {'status': 'FRESH_CACHE_NOT_CREATED', 'unresolved': [], 'successful_decisions': 0}
    with read_only(path) as c:
        if {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")} != TABLES:
            raise ValueError('Unexpected cache schema')
        if c.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
            raise ValueError('Router SQLite integrity failure')
        protocol_hash, encoded = c.execute('SELECT hash,data FROM protocol WHERE id=1').fetchone()
        protocol = json.loads(encoded)
        if digest(protocol) != protocol_hash or not local_protocol(protocol):
            raise PermissionError('Only the unchanged frozen CPU embedding router can be resumed')
        if check_all_decisions:
            for key, raw, expected in c.execute('SELECT key,record,hash FROM decisions'):
                record = json.loads(raw)
                if digest(record) != expected or record['cache_key'] != key or record['protocol_hash'] != protocol_hash:
                    raise ValueError('Successful cached decision failed integrity validation')
        allowed = {}
        if authorization is not None:
            if (str(path) != authorization['cache_path'] or protocol_hash != authorization['protocol_hash']):
                raise PermissionError('Wrong cache or router protocol for explicit resume')
            c.execute('ATTACH DATABASE ? AS frozen', (Path(authorization['snapshot']['path']).as_uri()+'?mode=ro',))
            for table, key, fields in (
                    ('protocol', 'id', ('hash', 'data')), ('decisions', 'key', ('record', 'hash')),
                    ('attempts', 'id', ('key', 'started', 'status', 'data')), ('cache_hits', 'id', ('key', 'at'))):
                differences = ' OR '.join('n.'+col+' IS NOT f.'+col for col in (key, *fields))
                if c.execute(f'SELECT COUNT(*) FROM frozen.{table} f LEFT JOIN main.{table} n ON n.{key}=f.{key} WHERE {differences}').fetchone()[0]:
                    raise ValueError('Original router ledger rows changed: '+table)
            allowed = {r['key']: r for r in authorization['allowed']}
            for key, item in allowed.items():
                row = c.execute('SELECT id,key,started,status,data FROM attempts WHERE id=?', (item['id'],)).fetchone()
                if row is None or row_signature(row) != item['row_sha256'] or row[1] != key:
                    raise ValueError('Authorized original reservation changed')
                visible = json.loads(row[4])['visible_input']
                if key != digest({'protocol_hash': protocol_hash, 'input_hash': digest(visible)}):
                    raise ValueError('Interrupted query input/key mismatch')
        unresolved = []
        for row in c.execute("SELECT a.id,a.key,a.started,a.status,a.data FROM attempts a LEFT JOIN decisions d ON a.key=d.key WHERE d.key IS NULL ORDER BY a.id"):
            item = allowed.get(row[1])
            if (item is None or row[3] != 'started' or row_signature(row) != item['row_sha256']):
                raise RouterCacheError('Unapproved interrupted router query; stop before GPU launch')
            unresolved.append({'id': row[0], 'key': row[1]})
        successful = c.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]
        if c.execute("SELECT COUNT(*) FROM attempts a LEFT JOIN decisions d ON a.key=d.key WHERE a.status='success' AND d.key IS NULL").fetchone()[0]:
            raise ValueError('Successful attempt has no decision')
        return {'status': 'PASS', 'protocol_hash': protocol_hash, 'successful_decisions': successful,
            'attempts': c.execute('SELECT COUNT(*) FROM attempts').fetchone()[0], 'unresolved': unresolved,
            'original_rows_unchanged': authorization is not None, 'external_api_calls': 0}


class ExplicitLocalResumeCache(RouterCache):
    def __init__(self, path, protocol, max_api_calls, *, authorization):
        self.authorization = authorization
        if (authorization.get('approved') is not True or authorization.get('max_reissues_per_key') != 1
                or authorization.get('external_api_calls') != 0 or not local_protocol(protocol) or Path(path).is_symlink()
                or str(Path(path).resolve()) != authorization['cache_path']
                or digest(protocol) != authorization['protocol_hash']
                or max_api_calls != authorization['max_local_calls']):
            raise PermissionError('Explicit resume cannot change the cache, local model, protocol or budget')
        self.allowed = {r['key']: r for r in authorization['allowed']}
        self.resume_metadata = {}
        super().__init__(path, protocol, max_api_calls)

    def reserve(self, key, visible_input):
        raise PermissionError('Single/API reservations are forbidden by the local-only resume adapter')

    def reserve_many_local(self, items):
        items = list(items)
        if len({key for key, _ in items}) != len(items):
            raise RouterCacheError('Duplicate batch reservation')
        metadata = {}
        with self._transaction() as c:
            count = c.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
            if count+len(items) > self.max_api_calls:
                raise RouterBudgetExceeded('Local embedding-call budget exhausted before batch forward')
            # Validate the entire batch before appending anything.
            for key, visible in items:
                if key != digest({'protocol_hash': self.protocol_hash, 'input_hash': digest(visible)}):
                    raise RouterCacheError('Query identity changed')
                if c.execute('SELECT 1 FROM decisions WHERE key=?', (key,)).fetchone():
                    raise RouterCacheError('Successful decisions must be reused, not resubmitted')
                rows = c.execute('SELECT id,key,started,status,data FROM attempts WHERE key=? ORDER BY id', (key,)).fetchall()
                if not rows:
                    continue
                item = self.allowed.get(key)
                if (item is None or len(rows) != 1 or rows[0][0] != item['id'] or rows[0][3] != 'started'
                        or row_signature(rows[0]) != item['row_sha256']
                        or json.loads(rows[0][4]).get('visible_input') != visible):
                    raise RouterCacheError('Prior failed/incomplete query is not authorized or its one reissue is consumed')
                metadata[key] = {'authorization_sha256': self.authorization['_authorization_sha256'],
                    'original_attempt_id': item['id'], 'original_attempt_row_sha256': item['row_sha256'],
                    'max_reissues_per_key': 1, 'external_api_calls': 0, 'automatic_retry': False}
            result = {}
            for key, visible in items:
                data = {'visible_input': visible}
                if key in metadata:
                    data['explicit_local_resume'] = metadata[key]
                cursor = c.execute("INSERT INTO attempts (key,started,status,data) VALUES (?,?,'started',?)",
                    (key, utc_now(), canonical_json(data)))
                result[key] = cursor.lastrowid
        for key, info in metadata.items():
            self.resume_metadata[result[key]] = info
        return result

    def finish(self, attempt, record):
        info = self.resume_metadata.get(attempt)
        if info is not None:
            record = {**record, 'explicit_local_resume': info}
        # Base finish changes only this newly appended attempt, never the old one.
        super().finish(attempt, record)


def install(path, root, update):
    """Scope the adapter to this explicit seed404 evaluator process only."""
    from agent_system.memory import skillrl_embedding_batch_router as batch
    value = load_authorization(path); root = Path(root).resolve()
    if str(root) != value['run_root'] or update not in (0, 5):
        raise PermissionError('Router resume authorization is scoped to the registered evaluation only')
    config = read_json(root/'protocol.json'); runtime = config['runtime']
    if (str(root) != value['run_root'] or update not in (0, 5)
            or file_hash(root/'protocol.json') != value['evaluation_protocol_sha256']
            or runtime['router_backend'] != 'skillrl_embedding_state_batch'
            or runtime['cache_path'] != value['cache_path'] or runtime['max_api_calls'] != 0
            or runtime['max_local_calls'] != value['max_local_calls']):
        raise PermissionError('Router resume authorization is scoped to the registered evaluation only')
    # No file or global paid-router implementation is modified. Each child has
    # its own Python interpreter; the base RouterCache still rejects retries.
    def factory(cache_path, protocol, max_local_calls):
        return ExplicitLocalResumeCache(cache_path, protocol, max_local_calls, authorization=value)
    batch.RouterCache = factory


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--router-resume-authorization', type=Path, required=True)
    custom, rest = parser.parse_known_args()
    context = argparse.ArgumentParser(add_help=False)
    context.add_argument('--root', type=Path, required=True); context.add_argument('--update', type=int, required=True)
    args, _ = context.parse_known_args(rest)
    install(custom.router_resume_authorization, args.root, args.update)
    sys.argv = ['phase2.evaluate', *rest]
    from phase2.evaluate import main as evaluate
    evaluate()


if __name__ == '__main__':
    main()
