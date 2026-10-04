import json
import sqlite3
import pytest
from omegaconf import OmegaConf


def test_recovery_checkpoint_frequency_does_not_change_training_or_validation():
    from scripts.resume_two_arm_u50_router import recovery_checkpoints
    cfg = OmegaConf.create({'trainer': {'save_freq': 5, 'test_freq': 5,
        'total_training_steps': 150, 'max_actor_ckpt_to_keep': 2},
        'algorithm': {'adv_estimator': 'grpo'}})
    recovery_checkpoints(cfg)
    assert cfg.trainer.save_freq == 1
    assert cfg.trainer.test_freq == 5 and cfg.trainer.total_training_steps == 150
    assert cfg.trainer.max_actor_ckpt_to_keep == 2
    assert cfg.algorithm.adv_estimator == 'grpo'


def test_retry_preserves_failed_accounting_and_successful_cache(tmp_path):
    from scripts.resume_two_arm_u50_router import reconcile_failed_router
    ledger, bank = tmp_path/'ledger.sqlite3', tmp_path/'bank.sqlite3'
    with sqlite3.connect(ledger) as db:
        db.execute('create table local_attempts(key text primary key, bank text, result text)')
        db.executemany('insert into local_attempts values (?,?,?)', [
            ('failed','bank',json.dumps({'status':'failed'})),
            ('success','bank',json.dumps({'status':'success'}))])
    with sqlite3.connect(bank) as db:
        db.execute('create table attempts(id integer primary key, key text, status text, data text)')
        db.execute('create table decisions(key text primary key, record text)')
        db.executemany('insert into attempts values (?,?,?,?)',
            [(1,'failed','failed','original failure'), (2,'success','success','original success')])
        db.execute("insert into decisions values ('success','unchanged')")
    reconcile_failed_router(ledger, bank, ['failed'], tmp_path/'evidence')
    with sqlite3.connect(ledger) as db:
        rows = db.execute('select key,result from local_attempts order by key').fetchall()
        assert len(rows) == 2
        assert not db.execute("select 1 from local_attempts where key='failed'").fetchone()
        assert db.execute("select result from local_attempts where key='success'").fetchone() == (json.dumps({'status':'success'}),)
    with sqlite3.connect(bank) as db:
        assert db.execute("select record from decisions where key='success'").fetchone() == ('unchanged',)
        assert db.execute('select status,data from attempts where id=1').fetchone() == ('failed','original failure')
    with pytest.raises(ValueError):
        reconcile_failed_router(ledger, bank, ['success'], tmp_path/'illegal')
