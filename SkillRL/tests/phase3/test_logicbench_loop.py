import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from phase3.common import ProtocolError
from phase3.logicbench import initial_bank
from phase3.logicbench_loop_data import make_schedule, training_configuration, resume_data_action
from phase3.logicbench_loop_readout import aggregate_tokens, v13_payload


def setting():
    return json.loads(Path('configs/phase3_logicbench_50updates_s707.json').read_text())


def test_schedule_has_ten_windows_and_separate_context_splits():
    plan = make_schedule(setting())
    assert len(plan['windows']) == 10
    assert [(w['start'], w['end']) for w in plan['windows']] == [(i, i + 5) for i in range(0, 50, 5)]
    assert all(len(w['questions']) == len({q['context_id'] for q in w['questions']}) == 640 for w in plan['windows'])
    train = {q['context_id'] for w in plan['windows'] for q in w['questions']}
    gate, monitor = ({q['context_id'] for q in plan[key]} for key in ('gate', 'monitor'))
    assert len(gate) == 192 and len(monitor) == 64
    assert not train & (gate | monitor) and not gate & monitor
    assert plan == make_schedule(setting())


def test_segment_resume_preserves_horizon_and_current_bank_dataset(tmp_path):
    config = setting()
    bank = initial_bank('readout_d')
    path = bank.save(tmp_path / 'banks')
    cfg = training_configuration(config, tmp_path, bank, path, start=5, gpu_ids=[0, 1, 2, 3])
    assert cfg.trainer.total_training_steps == cfg.actor_rollout_ref.actor.optim.total_training_steps == 150
    assert cfg.phase3.segment_start == 5 and cfg.phase3.segment_end == 10
    assert cfg.phase3.domain == 'logicbench'
    assert cfg.trainer.resume_from_path.endswith('global_step_5')
    assert 'u0005-u0010' in cfg.data.train_files
    assert cfg.phase2.enabled is False
    assert cfg.data.train_batch_size * cfg.logicbench_phase12.repeats == 1024
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6
    assert cfg.trainer.max_actor_ckpt_to_keep == 2
    assert cfg.actor_rollout_ref.cohort_seed == cfg.actor_rollout_ref.rollout.seed == 707


def test_dataloader_reset_only_at_registered_window_boundary():
    cfg = SimpleNamespace(domain='logicbench', segment_start=5, segment_end=10, resume_update=5)
    assert resume_data_action(cfg, 5) == 'new_window'
    cfg.resume_update = 9
    assert resume_data_action(cfg, 9) == 'restore'
    with pytest.raises(ProtocolError):
        resume_data_action(cfg, 8)


def test_sign_balance_is_answer_then_question_equal_and_zero_supported():
    bank = initial_bank('readout_d')
    a, b = bank.skill_ids[:2]
    rows = [dict(skill_id=a, question_id='q1', trajectory_id='t1', decision_id='t1:s0',
                 D_sign_balance=-1., D_signed_gate=-2., D_real=0., D_original=0.,
                 D_signed=-2., C_upd_centered=0., M_delta_centered=1., M_delta_raw=2., P_int=2.) for _ in range(3)]
    rows += [{**rows[0], 'trajectory_id':'t2', 'decision_id':'t2:s0', 'D_sign_balance':1.},
             {**rows[0], 'question_id':'q2', 'trajectory_id':'t3', 'decision_id':'t3:s0', 'D_sign_balance':1.},
             {**rows[0], 'skill_id':b, 'question_id':'q3', 'trajectory_id':'t4', 'decision_id':'t4:s0', 'D_sign_balance':0.}]
    result = {r['skill_id']:r for r in aggregate_tokens(rows, bank)}
    assert result[a]['D_sign_balance'] == .5
    assert result[b]['supported'] and result[b]['D_sign_balance'] == 0.
    assert result[bank.skill_ids[2]]['supported'] is False


def test_v13_selects_distinct_failed_skills_and_all_failed_answers():
    bank = initial_bank('readout_d')
    rows = [{'skill_id':s.skill_id,'skill_version_sha256':s.version_sha256,'supported':True,
             'D_sign_balance':100-i} for i,s in enumerate(bank.skills)]
    evidence = []
    for i,s in enumerate(bank.skills[:7]):
        for j in range(8):
            evidence.append({'evidence_id':f'{i}-{j}','question_id':f'q{i}','context_id':f'c{i}',
                'split':'train','sampling_policy_update':5,'selected_skill_id':s.skill_id,
                'skill_version_sha256':s.version_sha256,'question':'Question','task_type':'BQA',
                'response':'no','success':i==0 or j==0})
    payload, selection = v13_payload(bank, evidence, rows, k=5)
    assert selection == list(bank.skill_ids[1:6])
    assert len(payload['evidence']) == 35
    assert all(not r['success'] for r in payload['evidence'])
    assert len({r['evidence_id'] for r in payload['evidence']}) == 35
    assert not any('D_sign_balance' in r for r in payload['evidence'])
    assert v13_payload(bank, [{**e,'success':True} for e in evidence], rows, k=5) == (None, [])


def test_capture_single_step_actual_batch_and_versions(tmp_path):
    import numpy as np
    import pandas as pd
    from phase3.logicbench_loop_capture import archive_batch
    from phase1.logicbench_single_step import LogicBenchQuestion, build_prompt
    from transformers import AutoTokenizer
    bank = initial_bank('readout_d')
    bank_path = bank.save(tmp_path / 'banks')
    q = json.loads(Path('data/logicbench/sra19/aug_split_v2/train.json').read_text())[0]
    skill = bank.skills[0]
    tokenizer = AutoTokenizer.from_pretrained(setting()['model_path'], local_files_only=True)
    question = LogicBenchQuestion(q['question_id'], q['question'], q['task_type'], q['answer'], '')
    chat = tokenizer.apply_chat_template([{'role':'user','content':build_prompt(question, skill.payload)}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)
    prompt = tokenizer(chat, add_special_tokens=False)['input_ids']
    response = tokenizer(q['answer'], add_special_tokens=False)['input_ids']
    tensors = {'prompts':torch.tensor([prompt]),'responses':torch.tensor([response]),
        'attention_mask':torch.ones(1,len(prompt)+len(response),dtype=torch.long),
        'response_mask':torch.ones(1,len(response),dtype=torch.long),
        'advantages':torch.ones(1,len(response)), 'old_log_probs':torch.zeros(1,len(response))}
    meta = {'global_update':1,'environment_step':0,'trajectory_id':'t1',
            'info':{'question_id':q['question_id'],'selected_skill_id':skill.skill_id,'reward':1.}}
    batch = SimpleNamespace(batch=tensors, non_tensor_batch={
        'question_id':np.asarray([q['question_id']]),'selected_skill_id':np.asarray([skill.skill_id]),
        'episode_rewards':np.asarray([1.]),'phase2_metadata':np.asarray([json.dumps(meta)])},
        meta_info={'temperature':1.})
    parquet = tmp_path/'train.parquet'
    pd.DataFrame([{'question_id':q['question_id'],'selected_skill_id':skill.skill_id,
        'bank_sha256':bank.manifest_sha256,'skill_version_sha256':skill.version_sha256}]).to_parquet(parquet)
    cfg = SimpleNamespace(phase3=SimpleNamespace(root=str(tmp_path),segment_start=0,
        bank_path=str(bank_path),bank_sha256=bank.manifest_sha256,branch_id=bank.branch_id,
        questions_per_update=1,repeats=1), data=SimpleNamespace(train_files=str(parquet)),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(multi_turn=SimpleNamespace(enable=False))))
    archive_batch(batch,update=1,config=cfg,tokenizer=tokenizer)
    saved = torch.load(tmp_path/'direction_batches/u0001.pt',weights_only=False)
    assert saved['evidence'][0]['success']
    assert saved['evidence'][0]['sampling_policy_update'] == 0
    assert torch.equal(saved['tensors']['advantages'],tensors['advantages'])
    assert saved['evidence'][0]['skill_version_sha256'] == skill.version_sha256
    with pytest.raises(ProtocolError):
        archive_batch(batch,update=1,config=cfg,tokenizer=tokenizer)


def test_router_accepts_explicit_python_and_rejects_relative():
    from phase3.embedding_routing import validate_settings
    config = setting()['router']
    validate_settings(config)
    with pytest.raises(ProtocolError):
        validate_settings({**config,'python_executable':'python'})


def test_readiness_reports_missing_gpu_without_claiming_running(tmp_path, monkeypatch):
    from phase3.logicbench_loop import readiness
    monkeypatch.setattr(torch.cuda,'device_count',lambda:0)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    result = readiness(setting(),tmp_path,[0,1,2,3])
    assert result['status'] == 'blocked'
    assert 'gpu_access' in result['blockers']
    assert result['training_started'] is False


def test_sealed_cleanup_never_touches_unsealed_or_external_path(tmp_path):
    from phase3.logicbench_loop import cleanup
    root = tmp_path/'run'
    root.mkdir()
    with pytest.raises(ProtocolError):
        cleanup(root,5)


def test_banked_window_parquet_resume_preserves_numpy_chat_cells(tmp_path):
    from tests.phase3.test_logicbench import pool
    from phase3.logicbench_loop_data import prepare_window
    bank = initial_bank('readout_d')
    schedule = make_schedule(setting())
    window = {**schedule['windows'][0],'questions':schedule['windows'][0]['questions'][:2]}
    provider = pool(tmp_path,bank)
    path = prepare_window(setting(),tmp_path,bank,window,schedule['monitor'][:1],provider)
    (path/'complete.json').unlink()
    assert prepare_window(setting(),tmp_path,bank,window,schedule['monitor'][:1],provider) == path


def test_evaluation_identity_rejects_changed_inherited_decoding(tmp_path):
    from phase3.logicbench_loop import bind_evaluation_identity, validate_evaluation_rows
    cfg = setting()
    generator = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(to_dict=lambda:{'top_k':50})),
        tokenizer=SimpleNamespace(eos_token_id=123),torch=SimpleNamespace(__version__='fake'))
    bind_evaluation_identity(tmp_path/'audit',tmp_path,cfg,generator)
    generator.model.generation_config.to_dict = lambda:{'top_k':1}
    with pytest.raises(FileExistsError):
        bind_evaluation_identity(tmp_path/'audit',tmp_path,cfg,generator)
    rows = [{'game_id':'q','eval_seed':0,'bank_sha256':'b'}]*16
    with pytest.raises(ProtocolError):
        validate_evaluation_rows(rows,{'q'},list(range(16)),'b')


def test_failed_stage_is_not_left_marked_as_running(tmp_path,monkeypatch):
    import subprocess
    from phase3.logicbench_loop import command
    def fail(*args,**kwargs):
        raise subprocess.CalledProcessError(17,args[0])
    monkeypatch.setattr(subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):
        command(tmp_path,'train-u0000-u0005',['fake-stage'])
    record = json.loads((tmp_path/'status.json').read_text())
    assert record['stage']=='failed' and record['failed_stage']=='train-u0000-u0005'
