"""A single authorized 60->600 second editor transition; no automatic retries.

Original settings, implementation inventory and failed attempt remain immutable.
A separate receipt binds the only allowed runtime changes and one new request
key. Both old and new requests consume the same branch-wide API budget.
"""
from dataclasses import asdict, replace
from pathlib import Path
import sqlite3
import os
import tempfile

from phase3.api import APIConfig
from phase3.common import ProtocolError, canonical, digest, require, strict_json, write_new
from skillnet_cohort.common import file_hash

RECEIPT = 'recovery/editor-timeout-v1.json'


def _load(root):
    root=Path(root)
    receipt=strict_json((root/RECEIPT).read_text())
    require(receipt['setting_sha256']==digest(strict_json((root/'setting.json').read_text())), 'Recovery setting changed')
    require(receipt['original_implementation_sha256']==digest(strict_json((root/'implementation.json').read_text())),
            'Original implementation changed')
    require(receipt['recovery_module_sha256']==file_hash(Path(__file__)), 'Recovery module changed after authorization')
    with sqlite3.connect(f'file:{root}/editor.sqlite3?mode=ro',uri=True) as db:
        profile=db.execute('select value from profile where id=1').fetchone()
        try:
            migration=db.execute('select receipt from runtime_migrations where name=?',('editor-timeout-v1',)).fetchone()
        except sqlite3.Error:
            migration=None
        require(profile==(canonical(receipt['new_editor_profile']),) and migration==(canonical(receipt),),
                'Recovery receipt is not committed in the API ledger')
    return receipt


def _check_backup(path,failed_key,original):
    require(path.is_file() and not path.is_symlink(), 'Invalid recovery backup')
    try:
        with sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True) as db:
            require(db.execute('PRAGMA integrity_check').fetchone()==('ok',), 'Corrupt recovery backup')
            require(db.execute('select value from profile where id=1').fetchone()==(canonical(asdict(original)),),
                    'Backup must contain the original API profile')
            require(db.execute('select count(*) from attempts').fetchone()==(1,), 'Backup has wrong attempt inventory')
            row=db.execute('select request,result from attempts where key=?',(failed_key,)).fetchone()
            require(row is not None and row[1] is not None,'Backup missing failed request')
            return row
    except sqlite3.Error:
        raise ProtocolError('Invalid recovery backup database') from None


def editor_config(setting,root):
    cfg=APIConfig(**setting['editor'])
    if not (Path(root)/RECEIPT).exists():
        return cfg
    receipt=_load(root)
    require(digest(setting)==receipt['setting_sha256'] and asdict(cfg)==receipt['original_editor_profile'],
            'Changed original editor setting')
    expected=replace(cfg,timeout_seconds=receipt['new_editor_profile']['timeout_seconds'])
    require(asdict(expected)==receipt['new_editor_profile'], 'Recovery may only change timeout')
    return expected


def verify_implementation(root,current):
    root=Path(root)
    if not (root/RECEIPT).exists():
        write_new(root/'implementation.json',current)
        return
    receipt=_load(root)
    require(current==receipt['authorized_implementation'], 'Unregistered implementation revision')


def authorize(root, failed_key, implementation_after, *, authorization, timeout_seconds=600):
    """Offline migration: preserves old attempt bytes, no API request is sent."""
    root=Path(root)
    require(bool(authorization.strip()) and timeout_seconds==600, 'Expected explicit 600-second recovery authorization')
    setting=strict_json((root/'setting.json').read_text())
    original=APIConfig(**setting['editor'])
    require(original.stage=='editor' and original.timeout_seconds==60, 'Unexpected original timeout/stage')
    revised=replace(original,timeout_seconds=timeout_seconds)
    path=root/'editor.sqlite3'
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)), 'Unsafe editor ledger')
    backup=root/'recovery/editor-before-timeout-v1.sqlite3'
    backup.parent.mkdir(parents=True,exist_ok=True)
    # Save a read-only logical copy before any migration. Never rewrite it.
    if not backup.exists():
        descriptor,name=tempfile.mkstemp(prefix='.editor-backup-',suffix='.partial',dir=backup.parent)
        os.close(descriptor)
        temporary=Path(name)
        try:
            with sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True) as source, sqlite3.connect(temporary) as destination:
                source.backup(destination)
            _check_backup(temporary,failed_key,original)
            temporary.chmod(0o400)
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            os.link(temporary,backup)  # Atomic publication, refuses replacement.
            directory=os.open(backup.parent,os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(directory)
            finally: os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
    backed_up=_check_backup(backup,failed_key,original)
    with sqlite3.connect(path,timeout=60) as db:
        db.execute('BEGIN IMMEDIATE')
        old=db.execute('select request,result from attempts where key=?',(failed_key,)).fetchone()
        require(old is not None and old[1] is not None, 'No recorded failed request to recover')
        require(old==backed_up,'Original failed row differs from verified backup')
        request,result=map(strict_json,old)
        checked=dict(result);checksum=checked.pop('record_sha256')
        require(digest(checked)==checksum and digest(request)==failed_key,'Changed original API evidence')
        require(result['status']=='failed' and result['failure']['type']=='APITimeoutError'
                and request['identity']['event_id']=='u0005' and request['profile']==asdict(original),
                'Recovery applies only to the original U5 timeout')
        replacement={**request,'profile':asdict(revised)}
        new_key=digest(replacement)
        keys={r[0] for r in db.execute('select key from attempts')}
        require(keys <= {failed_key,new_key}, 'Unregistered API attempts before recovery')
        receipt={'schema_version':'logicbench.editor_timeout_recovery.v1','authorization':authorization,
            'failed_request_key':failed_key,'failed_request_record_sha256':digest(list(old)),
            'new_request_key':new_key,'original_editor_profile':asdict(original),'new_editor_profile':asdict(revised),
            'setting_sha256':digest(setting),
            'original_implementation_sha256':digest(strict_json((root/'implementation.json').read_text())),
            'authorized_implementation':implementation_after,'recovery_module_sha256':file_hash(Path(__file__)),
            'sdk_automatic_retries':0,'max_new_attempts_for_failed_request':1,'old_usage_and_cost':'unknown; retained',
            'future_editor_timeout_seconds':timeout_seconds,'backup_sha256':file_hash(backup)}
        current=db.execute('select value from profile where id=1').fetchone()
        require(current is not None and current[0] in (canonical(asdict(original)),canonical(asdict(revised))),
                'Unexpected ledger profile')
        if (root/RECEIPT).exists():
            require(strict_json((root/RECEIPT).read_text())==receipt,'Changed recovery receipt')
        db.execute('CREATE TABLE IF NOT EXISTS runtime_migrations (name TEXT PRIMARY KEY, receipt TEXT NOT NULL)')
        db.execute('INSERT OR IGNORE INTO runtime_migrations VALUES (?,?)',('editor-timeout-v1',canonical(receipt)))
        require(db.execute('SELECT receipt FROM runtime_migrations WHERE name=?',('editor-timeout-v1',)).fetchone()[0]==canonical(receipt),
                'Changed runtime migration')
        db.execute('UPDATE profile SET value=? WHERE id=1',(canonical(asdict(revised)),))
    # Publish only after the transaction commits. If publication is interrupted,
    # rerunning authorize verifies and finishes this same migration, without API work.
    write_new(root/RECEIPT,receipt)
    return receipt


def checkpoint_reserve(root,setting):
    """A completed endpoint resumes editing, without reserving another checkpoint."""
    from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
    root=Path(root)
    for start in range(0,setting['updates'],5):
        end=start+5
        if (root/'windows'/f'u{start:04d}-u{end:04d}'/'complete.json').exists():
            continue
        native=root/'checkpoints'/f'global_step_{end}'
        if native.exists():
            require(validate_full_checkpoint(native)['world_size']==4
                    and (root/'metrics'/f'u{end:04d}.json').is_file(),'Incomplete endpoint cannot bypass storage reserve')
            return 0
        return setting['storage']['checkpoint_reserve_bytes']
    return 0
