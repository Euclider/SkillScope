from dataclasses import asdict
import sqlite3

import pytest

from phase3.api import APIConfig, JSONClient
from phase3.common import ProtocolError, write_new
from tests.phase3.test_bank_api import FakeClient


class APITimeoutError(Exception):
    pass


def failed_root(tmp_path):
    cfg=APIConfig('editor','gpt-5.5',64000,8192,2,sdk_version='3.3.1')
    setting={'editor':asdict(cfg)}
    write_new(tmp_path/'setting.json',setting)
    before={'phase3/api.py':'a'*64}
    write_new(tmp_path/'implementation.json',before)
    api=JSONClient(cfg,tmp_path/'editor.sqlite3',client=FakeClient(APITimeoutError()),token_counter=lambda _:1)
    args=dict(identity={'event_id':'u0005'},system='frozen',payload={'evidence':['same']},schema={},validate=lambda _:None)
    with pytest.raises(ProtocolError): api.request(**args)
    with sqlite3.connect(tmp_path/'editor.sqlite3') as db:
        row=db.execute('select key,request,result from attempts').fetchone()
    return setting,args,row


def test_authorized_timeout_migration_preserves_failure_and_shared_budget(tmp_path):
    from logicbench_phase3_recovery.runtime import authorize, editor_config, verify_implementation
    setting,args,old=failed_root(tmp_path)
    after={'phase3/api.py':'b'*64}
    receipt=authorize(tmp_path,old[0],after,authorization='User requested repair and resume',timeout_seconds=600)
    assert receipt['failed_request_key']==old[0]
    cfg=editor_config(setting,tmp_path)
    assert cfg.timeout_seconds==600 and cfg.max_api_calls==2
    verify_implementation(tmp_path,after)
    with pytest.raises(ProtocolError): verify_implementation(tmp_path,{'phase3/api.py':'c'*64})
    api=JSONClient(cfg,tmp_path/'editor.sqlite3',client=FakeClient({'ok':True}),token_counter=lambda _:1)
    api.request(**args)
    _,account=api.request(**args)
    assert account['api_calls']==0
    with sqlite3.connect(tmp_path/'editor.sqlite3') as db:
        assert db.execute('select key,request,result from attempts order by id').fetchone()==old
        assert db.execute('select count(*) from attempts').fetchone()[0]==2
    with pytest.raises(ProtocolError,match='budget'):
        api.request(**{**args,'identity':{'event_id':'u0010'}})
    assert authorize(tmp_path,old[0],after,authorization='User requested repair and resume',timeout_seconds=600)==receipt


def test_failed_recovery_is_not_automatically_retried(tmp_path):
    from logicbench_phase3_recovery.runtime import authorize,editor_config
    setting,args,old=failed_root(tmp_path)
    authorize(tmp_path,old[0],{'phase3/api.py':'b'*64},authorization='User requested repair and resume',timeout_seconds=600)
    client=FakeClient(APITimeoutError())
    api=JSONClient(editor_config(setting,tmp_path),tmp_path/'editor.sqlite3',client=client,token_counter=lambda _:1)
    with pytest.raises(ProtocolError): api.request(**args)
    with pytest.raises(ProtocolError,match='automatic retry'): api.request(**args)
    assert len(client.calls)==1


def test_recovery_cannot_change_other_editor_parameters(tmp_path):
    from logicbench_phase3_recovery.runtime import authorize,editor_config
    setting,args,old=failed_root(tmp_path)
    authorize(tmp_path,old[0],{'phase3/api.py':'b'*64},authorization='User requested repair and resume',timeout_seconds=600)
    altered={'editor':{**setting['editor'],'model':'o3'}}
    with pytest.raises(ProtocolError): editor_config(altered,tmp_path)


def test_completed_endpoint_resume_does_not_reserve_another_checkpoint(tmp_path,monkeypatch):
    from logicbench_phase3_recovery.runtime import checkpoint_reserve
    from phase1 import watch_qwen35_checkpoints
    setting={'updates':50,'storage':{'checkpoint_reserve_bytes':80}}
    assert checkpoint_reserve(tmp_path,setting)==80
    (tmp_path/'checkpoints/global_step_5').mkdir(parents=True)
    write_new(tmp_path/'metrics/u0005.json',{'complete':True})
    monkeypatch.setattr(watch_qwen35_checkpoints,'validate_full_checkpoint',lambda _: {'world_size':4})
    assert checkpoint_reserve(tmp_path,setting)==0
    write_new(tmp_path/'windows/u0000-u0005/complete.json',{'complete':True})
    assert checkpoint_reserve(tmp_path,setting)==80


def test_incomplete_backup_cannot_authorize_migration(tmp_path):
    from logicbench_phase3_recovery.runtime import authorize
    _,_,old=failed_root(tmp_path)
    p=tmp_path/'recovery/editor-before-timeout-v1.sqlite3';p.parent.mkdir();p.touch()
    with pytest.raises(ProtocolError):
        authorize(tmp_path,old[0],{'phase3/api.py':'b'*64},authorization='Repair',timeout_seconds=600)
    assert not (tmp_path/'recovery/editor-timeout-v1.json').exists()


def test_invalid_ledger_profile_does_not_publish_receipt(tmp_path):
    from logicbench_phase3_recovery.runtime import authorize
    _,_,old=failed_root(tmp_path)
    with sqlite3.connect(tmp_path/'editor.sqlite3') as db:
        db.execute("update profile set value='{}' where id=1")
    with pytest.raises(ProtocolError):
        authorize(tmp_path,old[0],{'phase3/api.py':'b'*64},authorization='Repair',timeout_seconds=600)
    assert not (tmp_path/'recovery/editor-timeout-v1.json').exists()


def test_receipt_requires_committed_database_and_can_finish_publication(tmp_path):
    from logicbench_phase3_recovery.runtime import authorize,editor_config
    setting,_,old=failed_root(tmp_path)
    args=dict(authorization='Repair',timeout_seconds=600)
    after={'phase3/api.py':'b'*64}
    receipt=authorize(tmp_path,old[0],after,**args)
    (tmp_path/'recovery/editor-timeout-v1.json').unlink()  # Simulate commit before receipt publication.
    assert authorize(tmp_path,old[0],after,**args)==receipt
    with sqlite3.connect(tmp_path/'editor.sqlite3') as db:
        db.execute('delete from runtime_migrations')
    with pytest.raises(ProtocolError):editor_config(setting,tmp_path)
