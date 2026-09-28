"""Offline integration contracts; none of these tests perform RL or call APIs."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from phase3.bank import Bank
from phase3.common import ProtocolError, digest, write_new
from phase3.prepare import ARMS, partition_seen, validate_runtime
from phase3.report import actor_accounting


def runtime():
    return {'model_path':'/model', 'data_root':'/data', 'gpu_ids':list(range(8)),
        'rl_seed':707, 'optimizer_horizon_updates':150, 'initial_stop_update':20,
        'inference_profile':str(Path(__file__).resolve().parents[2]/'configs/phase3_vllm_v1.json'),
        'router':{'backend':'skillrl_embedding_state', 'model_path':'/embedding', 'device':'cpu',
                  'profile_sha256':'d1fc18a18537e65d4503d65d46187b8b63d21d9a884f6332262dbbb2dda778c8',
                  'max_local_calls':600000},
        'editor':{'max_input_tokens':100000, 'max_completion_tokens':8192, 'max_api_calls':30},
        'gate_games_per_task':1, 'gate_tolerance_pp':0, 'eval_seeds':[1404],
        'max_evidence_trajectories':10, 'readout_parity_atol':0.03,
        'storage':{'maximum_run_bytes':10**12, 'minimum_free_bytes':10**9, 'checkpoint_reserve_bytes':10**10}}


def test_runtime_requires_explicit_budgets():
    cfg = runtime()
    assert validate_runtime(cfg) == cfg
    cfg['router']['max_local_calls'] = None
    with pytest.raises(ValueError):
        validate_runtime(cfg)


@pytest.mark.parametrize('profile,utilization', [('v2', 0.38), ('v3', 0.36)])
def test_registered_memory_profile_keeps_phase3_recipe(monkeypatch, tmp_path, profile, utilization):
    import phase3.training as training
    cfg = runtime()
    cfg['inference_profile'] = str(Path(__file__).resolve().parents[2] / f'configs/phase3_vllm_memory_{profile}.json')
    cfg['router']['device'] = 'cuda:0'
    cfg['router']['execution'] = {'mode':'state_batch_fp32_micro_v2', 'intra_op_threads':1,
        'shared_gpu_physical_id':7, 'transport':'subprocess_pipe_v1', 'forward_microbatch_size':2}
    assert validate_runtime(cfg) == cfg
    monkeypatch.setattr(training, 'load', lambda _: ({}, cfg))
    preparation = tmp_path / 'assets/manifest.json'
    write_new(preparation.parent / 'split.json', {'evidence_seen':[{'game_id':'json_2.1.1/valid_seen/game.tw-pddl'}]})
    bank = Bank.initial('readout_d')
    path = bank.save(tmp_path / 'banks')
    actual = training.configuration(preparation, tmp_path/'run', 'readout_d', path, bank.manifest_sha256, 0)
    assert actual.actor_rollout_ref.rollout.gpu_memory_utilization == utilization
    assert actual.actor_rollout_ref.rollout.max_num_seqs == 16
    assert actual.env.skills_only_memory.step_routing.execution.shared_gpu_physical_id == 7
    assert actual.data.train_batch_size * actual.env.rollout.n == 128
    assert actual.trainer.test_freq == actual.trainer.save_freq == 5
    assert actual.phase3.penultimate_recovery_checkpoint is True
    assert actual.trainer.max_actor_ckpt_to_keep == 2
    recovered = training.configuration(preparation, tmp_path/'run', 'readout_d', path,
                                       bank.manifest_sha256, 0, resume_update=4)
    assert recovered.trainer.resume_mode == 'resume_path'
    assert recovered.trainer.resume_from_path.endswith('checkpoints/global_step_4')
    assert recovered.phase3.segment_start == 0 and recovered.phase3.segment_end == 5


def test_u20_is_an_interim_stop_under_an_unchanged_u150_optimizer_schedule(monkeypatch, tmp_path):
    from phase3 import run
    from phase3.common import ProtocolError
    cfg = runtime()
    manifest = {'initial_banks': {branch: 'a'*64 for branch in ARMS}}
    monkeypatch.setattr(run, 'load', lambda _: (manifest, cfg))
    first = run.plan(tmp_path/'assets/manifest.json', tmp_path/'run', 'readout_d')
    later = run.plan(tmp_path/'assets/manifest.json', tmp_path/'run', 'readout_d', 50)
    assert first['requested_stop_update'] == 20 and first['windows'][-1] == [15, 20]
    assert later['requested_stop_update'] == 50 and later['windows'][-1] == [45, 50]
    assert first['optimizer_horizon_updates'] == later['optimizer_horizon_updates'] == 150
    assert run.base_plan(manifest, cfg, tmp_path/'run', 'readout_d') == {
        key: value for key, value in later.items() if key not in
        ('requested_stop_update', 'windows', 'interim_unless_optimizer_horizon')}
    for invalid in (15, 21, 151, True):
        with pytest.raises(ProtocolError):
            run.stop_at(cfg, invalid)


def test_only_previous_generated_endpoint_is_retired_after_successor_seal(tmp_path, monkeypatch):
    from phase3 import run
    from phase1 import watch_qwen35_checkpoints
    root = tmp_path/'new-phase3-run'
    event = root/'events/u0010'
    old_native, old_export = root/'checkpoints/global_step_5', root/'models/u0005'
    new_native, new_export = root/'checkpoints/global_step_10', root/'models/u0010'
    for directory in (old_native, old_export, new_native, new_export):
        directory.mkdir(parents=True)
        (directory/'fixture').write_text(directory.name)
    write_new(root/'plan.json', {'root':str(root),
        'checkpoint_retention':'latest_native_and_latest_HF_export_after_seal'})
    write_new(event/'complete.json', {'selected_bank_sha256':'a'*64})
    identity = {'old_policy_sha256':'b'*64,'new_policy_sha256':'c'*64}
    write_new(event/'identity.json', identity)
    write_new(event/'running_metrics.json', {})
    readout = {'identity':identity}
    prediction = root/'predictions/u0005-u0010'
    write_new(prediction/'readout.json', readout)
    write_new(prediction/'complete.json', {'readout_sha256':digest(readout)})
    write_new(event/'shadow_action_bias.json', {'readout_sha256':digest(readout),
        'threshold_decision_applied':False})
    monkeypatch.setattr(run, 'model_identity', lambda path: 'b'*64 if Path(path) == old_export else 'c'*64)
    monkeypatch.setattr(watch_qwen35_checkpoints, 'validate_full_checkpoint', lambda path: {'world_size':8})
    run.retire_previous_endpoint(root, event, 5, 'b'*64, 'c'*64, new_export, new_native, 8)
    assert not old_native.exists() and not old_export.exists()
    assert new_native.exists() and new_export.exists()
    assert (event/'retention_intent.json').is_file() and (event/'retention_complete.json').is_file()
    run.retire_previous_endpoint(root, event, 5, 'b'*64, 'c'*64, new_export, new_native, 8)


def test_penultimate_checkpoint_retained_until_event_is_sealed(tmp_path, monkeypatch):
    from phase3 import run
    from phase1 import watch_qwen35_checkpoints
    root = tmp_path / 'new-phase3-run'
    event = root / 'events/u0005'
    u4 = root / 'checkpoints/global_step_4'
    u5 = root / 'checkpoints/global_step_5'
    u4.mkdir(parents=True)
    u5.mkdir(parents=True)
    (u4 / 'fixture').write_text('old')
    (u5 / 'fixture').write_text('new')
    monkeypatch.setattr(watch_qwen35_checkpoints, 'validate_full_checkpoint', lambda path: {'world_size': 8})
    with pytest.raises(ProtocolError, match='event sealing'):
        run.retire_penultimate_checkpoint(root, event, 5, u5, 8)
    assert u4.exists()
    write_new(event / 'complete.json', {'selected_bank_sha256': 'a' * 64})
    write_new(event / 'running_metrics.json', {})
    run.retire_penultimate_checkpoint(root, event, 5, u5, 8)
    assert not u4.exists() and u5.exists()
    assert (event / 'penultimate_retention_complete.json').is_file()
    run.retire_penultimate_checkpoint(root, event, 5, u5, 8)


def test_failed_bf16_prediction_requires_explicit_nonoverwriting_recovery(tmp_path):
    from phase3 import run
    root = tmp_path / 'run'
    event = root / 'events/u0005'
    original = root / 'predictions/u0000-u0005'
    write_new(original / 'source.json', {'backend': 'matched-offline-BF16-SDPA-full-vocabulary-v1'})
    failure = root / 'logs/predict-u0005.log'
    failure.parent.mkdir(parents=True)
    failure.write_text('Offline/live chosen-token parity exceeds the predeclared tolerance')
    with pytest.raises(ProtocolError, match='explicit non-overwriting recovery'):
        run.prediction_artifacts(root, event, 0, 5)
    target, log = run.prediction_artifacts(root, event, 0, 5, repair=True)
    assert target == root / 'predictions/u0000-u0005-fp32-autocast-v2'
    assert log == root / 'logs/predict-u0005-fp32-autocast-v2.log'
    assert run.prediction_artifacts(root, event, 0, 5) == (target, log)
    assert (original / 'source.json').is_file() and failure.is_file()
    with pytest.raises(FileExistsError):
        write_new(event / 'prediction_recovery.json', {'changed': True})


def test_resumed_window_uses_native_bf16_forward_and_preserves_failed_prediction(tmp_path):
    from phase3 import predict, run
    assert predict.endpoint_forward_precision(0) == (
        'float32', 'matched-offline-FP32-weights-BF16-autocast-SDPA-full-vocabulary-v2')
    assert predict.endpoint_forward_precision(5) == (
        'bfloat16', 'matched-offline-BF16-weights-FP32-logsoftmax-SDPA-full-vocabulary-v3')
    root = tmp_path / 'run'
    event = root / 'events/u0010'
    original = root / 'predictions/u0005-u0010'
    write_new(original / 'source.json', {'backend': predict.endpoint_forward_precision(0)[1]})
    failure = root / 'logs/predict-u0010.log'
    failure.parent.mkdir(parents=True)
    failure.write_text('Offline/live chosen-token parity exceeds the predeclared tolerance')
    with pytest.raises(ProtocolError, match='explicit non-overwriting recovery'):
        run.prediction_artifacts(root, event, 5, 10)
    target, log = run.prediction_artifacts(root, event, 5, 10, repair=True)
    assert target == root / 'predictions/u0005-u0010-bf16-weights-v3'
    assert log == root / 'logs/predict-u0010-bf16-weights-v3.log'
    assert run.prediction_artifacts(root, event, 5, 10) == (target, log)
    assert original.joinpath('source.json').is_file() and failure.is_file()


def test_seen_partition_disjoint_and_order_independent():
    from skillnet_cohort.assets import TASKS
    rows = [{'game_id':f'{task}/{i}', 'task_type':task} for task in TASKS for i in range(4)]
    a,b = partition_seen(rows,1)
    assert partition_seen(list(reversed(rows)),1) == (a,b)
    assert len(a) == 6 and len(b) == 18
    assert not {row['game_id'] for row in a} & {row['game_id'] for row in b}


@pytest.mark.parametrize('branch', ARMS)
@pytest.mark.parametrize('start', [0,5])
def test_phase3_actual_configuration(monkeypatch, tmp_path, branch, start):
    import phase3.training as training
    monkeypatch.setattr(training, 'load', lambda _: ({}, runtime()))
    preparation = tmp_path / 'assets/manifest.json'
    write_new(preparation.parent / 'split.json', {'evidence_seen':[{'game_id':'json_2.1.1/valid_seen/game.tw-pddl'}]})
    bank = Bank.initial(branch)
    path = bank.save(tmp_path / 'banks')
    cfg = training.configuration(preparation,tmp_path/'run',branch,path,bank.manifest_sha256,start)
    assert cfg.trainer.total_training_steps == 150
    assert cfg.actor_rollout_ref.actor.optim.total_training_steps == 150
    assert cfg.algorithm.adv_estimator == 'grpo'
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6
    assert cfg.data.train_batch_size * cfg.env.rollout.n == 128
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 128
    assert cfg.trainer.n_gpus_per_node == 8
    assert cfg.phase3.segment_end == start+5
    assert not cfg.phase2.enabled and not cfg.env.phase1_archive.enabled
    assert cfg.env.seed == 707 + 16*(start//5)
    assert cfg.actor_rollout_ref.rollout.name == 'vllm_v1'
    assert cfg.actor_rollout_ref.rollout.inference_profile.settings.seed == 707
    assert cfg.env.skills_only_memory.step_routing.bank_sha256 == bank.manifest_sha256
    assert cfg.ray_init.num_cpus == 32
    assert cfg.actor_rollout_ref.actor.fsdp_config.cpu_shard_init
    if start:
        assert cfg.trainer.resume_mode == 'resume_path'
        assert cfg.trainer.resume_from_path.endswith('global_step_5')


def test_direction_capture_first_batch_including_shadow_only_failure_arm(tmp_path):
    from phase3.capture import archive_batch
    cfg = OmegaConf.create({'phase3':{'root':str(tmp_path),'segment_start':5,'selector':'gated_d',
        'bank_sha256':'a'*64,'branch_id':'readout_d'}, 'actor_rollout_ref':{'rollout':{'multi_turn':{'enable':True}}}})
    batch = SimpleNamespace(batch={
        'input_ids':torch.tensor([[1,2,3]]), 'attention_mask':torch.tensor([[1,1,1]]),
        'position_ids':torch.tensor([[0,1,2]]), 'responses':torch.tensor([[2,3]]),
        'advantages':torch.tensor([[1.,0.]]), 'old_log_probs':torch.tensor([[-1.,-2.]]),
        'loss_mask':torch.tensor([[1,1,0]])},
        non_tensor_batch={'phase3_metadata':np.asarray([json.dumps({'global_update':6,'info':{'bank_sha256':'a'*64}})], dtype=object)},
        meta_info={'temperature':1.})
    archive_batch(batch,update=7,config=cfg)
    assert not list(tmp_path.rglob('*.pt'))
    archive_batch(batch,update=6,config=cfg)
    saved=torch.load(tmp_path/'direction_batches/u0006.pt',weights_only=False)
    assert saved['tensors']['actual_loss_mask'].tolist() == [[1,0]]
    assert 'logits' not in saved['tensors'] and len(saved['metadata']) == 1
    with pytest.raises(ValueError):
        archive_batch(batch,update=6,config=cfg)
    cfg.phase3.selector='failure_driven'
    cfg.phase3.root=str(tmp_path/'failure-arm')
    archive_batch(batch,update=6,config=cfg)
    assert (tmp_path/'failure-arm/direction_batches/u0006.pt').is_file()


def test_actor_totals_include_training_monitor_gate_final(tmp_path):
    steps=[{'prompt_tokens':10,'completion_tokens':2}]
    write_new(tmp_path/'episodes/u0001/train/a.json',{'split':'train','steps':steps})
    write_new(tmp_path/'episodes/u0005/valid_seen/b.json',{'split':'valid_seen','steps':steps})
    row={'steps':steps}
    for path in ('events/u0005/evaluations/bank/episodes/c.json','final/valid_unseen/episodes/d.json'):
        write_new(tmp_path/path,{'result':row,'result_sha256':digest(row)})
    report=actor_accounting(tmp_path)
    assert report['total'] == {'episodes':4,'steps':4,'prompt_tokens':40,'completion_tokens':8}
    assert set(report['by_stage']) == {'training','seen_monitor','paired_gate','final'}


@pytest.mark.parametrize('selector,passive', [('failure_driven', True), ('reward_sign_balance', False)])
def test_passive_failure_shadow_cost_is_not_charged_to_operational_selector(tmp_path, selector, passive):
    from phase3.report import summarize
    write_new(tmp_path/'plan.json', {'branch':'fixture', 'selector':selector, 'router_backend':'external_llm'})
    write_new(tmp_path/'predictions/u0000-u0005/complete.json', {
        'forward_calls':12, 'forward_input_tokens':1000, 'wall_seconds':3.5})
    result = summarize(tmp_path)['readout']
    assert result['windows'] == 1 and result['forward_calls'] == 12
    assert result['operational_selector_cost']['forward_calls'] == (0 if passive else 12)
    assert result['passive_shadow_cost']['forward_calls'] == (12 if passive else 0)


def test_unproven_vocab_rows_are_never_deleted(tmp_path):
    from skillnet_cohort.common import file_hash, write_new_bytes, write_new_json
    from skillnet_cohort.window_storage import reclaim_proven_rows
    root=tmp_path/'new-run'
    write_new_json(root/'launch.json',{})
    window=root/'windows/u0000-u0005-valid_seen'
    write_new_json(window/'report.json',{'complete':True})
    write_new_json(window/'sealed.json',{'start':0,'end':5,'files':[
        {'path':'report.json','sha256':file_hash(window/'report.json')}]})
    row=root/'old_logprobs/u0001/row-000000.pt'
    write_new_bytes(row,b'synthetic full vocabulary bytes')
    permit={'run_root':str(root),'reclaim_regenerable_full_vocab':True}
    result=reclaim_proven_rows(root,[window],0,5,permit)
    assert result['deleted'] == [] and len(result['retained']) == 1 and row.exists()
    with pytest.raises(PermissionError):
        reclaim_proven_rows(root,[window],0,5,{**permit,'reclaim_regenerable_full_vocab':False})


def test_cleanup_rejects_changed_seal(tmp_path):
    from skillnet_cohort.common import write_new_json
    from skillnet_cohort.window_storage import reclaim_proven_rows
    root=tmp_path/'new-run'; window=root/'windows/w'
    write_new_json(root/'launch.json',{})
    write_new_json(window/'report.json',{})
    write_new_json(window/'sealed.json',{'start':0,'end':5,'files':[{'path':'report.json','sha256':'0'*64}]})
    with pytest.raises(ValueError,match='Changed sealed'):
        reclaim_proven_rows(root,[window],0,5,{'run_root':str(root),'reclaim_regenerable_full_vocab':True})


def test_phase3_evaluation_uses_registered_vllm_profile(tmp_path, monkeypatch):
    from phase3.evaluate import evaluate_bank
    from skillnet_cohort.common import file_hash
    from phase3 import routing
    from skillnet_cohort import inference
    from phase1 import eval_skill_margin
    bank = Bank.initial('readout_d')
    game = tmp_path/'data/json_2.1.1/valid_seen/unit/game.tw-pddl'
    game.parent.mkdir(parents=True)
    game.write_text('synthetic fixture')
    calls = []
    monkeypatch.setattr(routing, 'BranchRouter', lambda *_: SimpleNamespace(memory='fixture'))
    def make_policy(checkpoint, settings, max_prompt_tokens):
        assert settings['inference_profile']['settings']['backend'] == 'vllm_v1'
        assert settings['inference_profile']['settings']['seed'] == 707
        assert max_prompt_tokens == 4096
        calls.append('vllm')
        return 'vllm-policy'
    monkeypatch.setattr(inference, 'make_policy', make_policy)
    def run_episode(**kwargs):
        assert kwargs['policy'] == 'vllm-policy'
        assert kwargs['environment_seed'] == 1707
        return {'success':True, 'steps':[{'selected_skill_id':bank.skills[0].skill_id,
            'prompt_text':'discarded', 'prompt_tokens':1, 'completion_tokens':1,
            'is_action_valid':True}]}
    monkeypatch.setattr(eval_skill_margin, 'run_episode', run_episode)
    rows = evaluate_bank(bank=bank, checkpoint='/fixture/model', policy_sha256='a'*64,
        games=[{'game_id':game.relative_to(tmp_path/'data').as_posix(),
                'game_sha256':file_hash(game), 'task_type':'pick_and_place'}],
        seeds=[1707], data_root=tmp_path/'data', output=tmp_path/'evaluation', router_api=object(),
        inference_profile=runtime()['inference_profile'])
    assert calls == ['vllm'] and rows[0]['success']
    assert not any('prompt_text' in step for step in rows[0]['steps'])
    plan = json.loads((tmp_path/'evaluation/plan.json').read_text())
    assert len(plan['jobs'][0]['inference_profile_sha256']) == 64


def test_unapproved_v3_cannot_start_formal_phase3(tmp_path, monkeypatch):
    from phase3 import run
    monkeypatch.setattr(run, 'load', lambda _: ({}, runtime()))
    settings = json.loads((Path(__file__).resolve().parents[2]/'configs/phase3_setting_embedding_v3.json').read_text())
    settings['execution']['approved'] = False
    write_new(tmp_path/'assets/setting.json', settings)
    with pytest.raises(ValueError, match='awaits launch approval'):
        run.execute(tmp_path/'assets/manifest.json', tmp_path/'run', 'readout_d')
