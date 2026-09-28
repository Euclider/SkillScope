from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace as NS
import sys

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from skillnet_cohort.common import REPO, file_hash, read_json, write_new_json, write_new_bytes, load_preparation
from skillnet_cohort.inference import PROFILE, apply_training, registration, validate
from skillnet_cohort.runtime import runtime_settings
from skillnet_cohort.vllm_backend import normalized_text_weights, text_config, VLLMPolicy
from skillnet_cohort.training import configuration
from verl import DataProto
from verl.workers.rollout.vllm_v1 import VLLMV1Rollout, pack_outputs
from verl.workers.sharding_manager import fsdp_vllm_v1 as bridge_module


def test_new_preparation_binds_all_generation_paths_without_changing_grpo(tmp_path):
    base = REPO / 'docs/experiments/phase12-daybudget-v2/preparation-s404-8gpu'
    manifest = read_json(base / 'manifest.json')
    for asset in manifest['assets']:
        if asset['path'] == 'spec.json':
            spec = read_json(base / asset['path'])
            spec['inference_profile'] = registration(PROFILE)
            write_new_json(tmp_path / asset['path'], spec)
        else:
            write_new_bytes(tmp_path / asset['path'], (base / asset['path']).read_bytes())
        asset['sha256'] = file_hash(tmp_path / asset['path'])
    write_new_json(tmp_path / 'manifest.json', manifest)
    load_preparation(tmp_path / 'manifest.json')
    cfg = configuration(tmp_path / 'manifest.json', tmp_path / 'run')
    assert cfg.actor_rollout_ref.rollout.name == 'vllm_v1'
    assert cfg.actor_rollout_ref.rollout.micro_batch_size == 16
    assert cfg.actor_rollout_ref.actor.strategy == 'fsdp'
    assert cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6
    assert cfg.data.train_batch_size == 16 and cfg.env.rollout.n == 8
    assert cfg.trainer.total_training_steps == 5
    assert cfg.phase2.enabled and cfg.env.phase1_archive.enabled
    runtime = runtime_settings(spec, tmp_path / 'cache', max_local_calls=10)
    assert runtime['inference_profile'] == spec['inference_profile']


def test_profile_tampering_fails_and_no_execution_authority():
    binding = registration(PROFILE)
    assert not binding['settings']['execution_authorized']
    binding['settings']['gpu_memory_utilization'] = .9
    with pytest.raises(ValueError, match='changed'):
        validate({'inference_profile': binding})


def test_weight_name_mapping_preserves_text_exports_and_lm_head():
    values = [('model.language_model.layers.0.norm.weight', 1), ('lm_head.weight', 2),
              ('model.visual.weight', 3), ('model.embed_tokens.weight', 4)]
    assert list(normalized_text_weights(values)) == [('model.layers.0.norm.weight', 1),
                                                    ('lm_head.weight', 2), ('model.embed_tokens.weight', 4)]


def test_text_config_does_not_modify_original():
    probe = NS(model_type='dummy_qwen3_5')
    assert text_config(probe) is probe
    original = NS(text_config=NS(model_type='qwen3_5_text', architectures=['Original']))
    assert text_config(original).architectures == ['SkillScopeQwen35Text']
    assert original.text_config.architectures == ['Original']
    with pytest.raises(ValueError, match='Qwen3.5'):
        text_config(NS(model_type='unknown'))


def prompts():
    return DataProto(TensorDict({'input_ids': torch.tensor([[0, 0, 3, 0], [0, 4, 5, 6]]),
        'attention_mask': torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]),
        'position_ids': torch.tensor([[0, 0, 0, 1], [0, 0, 1, 2]])}, batch_size=[2]),
        meta_info={'eos_token_id': 0, 'pad_token_id': 0})


def outputs():
    return [NS(outputs=[NS(token_ids=ids, logprobs=[{t:NS(logprob=-.5)} for t in ids])])
            for ids in ([7, 0], [8, 9, 0])]


def test_packing_uses_actual_length_with_pad_equals_eos():
    packed = pack_outputs(prompts(), outputs(), response_length=4, pad_token_id=0)
    assert packed.batch['responses'].tolist() == [[7,0,0,0], [8,9,0,0]]
    assert packed.batch['attention_mask'][:, -4:].tolist() == [[1,1,0,0], [1,1,1,0]]
    assert packed.batch['rollout_log_probs'][0].tolist() == [-.5,-.5,0,0]
    assert packed.batch['position_ids'][1,-4:].tolist() == [3,4,5,6]
    with pytest.raises(ValueError, match='count'):
        pack_outputs(prompts(), outputs()[:1], response_length=4, pad_token_id=0)
    with pytest.raises(ValueError, match='Missing'):
        pack_outputs(prompts(), outputs(), response_length=1, pad_token_id=0)


def test_rollout_batches_and_preserves_masked_prompt_tokens(monkeypatch):
    seen = []
    monkeypatch.setitem(sys.modules, 'vllm', NS(SamplingParams=lambda **kw:NS(**kw)))
    rollout = VLLMV1Rollout.__new__(VLLMV1Rollout)
    rollout.config = OmegaConf.create({'prompt_length':4096, 'response_length':4, 'temperature':1.,
        'top_p':1., 'top_k':0, 'val_kwargs':{'temperature':.4, 'top_p':1., 'top_k':0}})
    rollout.tokenizer = NS(eos_token_id=0)
    def generate(inputs, params, **kwargs):
        seen.append((inputs, params))
        return outputs()
    rollout.inference_engine = NS(generate=generate)
    batch = prompts()
    batch.meta_info['validate'] = True
    rollout.generate_sequences(batch)
    assert seen[0][0] == [{'prompt_token_ids':[3,0]}, {'prompt_token_ids':[4,5,6]}]
    assert seen[0][1].temperature == .4 and seen[0][1].top_k == -1
    assert seen[0][1].n == 1
    assert batch.batch['input_ids'][0].tolist() == [0,0,3,0]


def test_training_engine_uses_registered_seed_not_hardcoded_404(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 3)
    monkeypatch.setattr('skillnet_cohort.vllm_backend.build_engine', lambda *a, **k: seen.append(k))
    for seed in (404, 505, 606):
        cfg = OmegaConf.create({'tensor_model_parallel_size': 1, 'seed': seed,
                               'inference_profile': registration(PROFILE)})
        VLLMV1Rollout('/not-loaded', cfg, None)
    assert [item['seed'] for item in seen] == [407, 508, 609]


def test_eval_seed_is_per_request_and_token_budget_is_enforced(monkeypatch):
    monkeypatch.setitem(sys.modules, 'vllm', NS(SamplingParams=lambda **kw:NS(**kw)))
    policy = VLLMPolicy.__new__(VLLMPolicy)
    policy.tokenizer = NS(apply_chat_template=lambda *a, **k:'rendered', encode=lambda *a, **k:[1,2])
    seen = []
    def generate(prompts, params, **kwargs):
        seen.extend(p.seed for p in params)
        return [NS(prompt_token_ids=[1,2], outputs=[NS(text='go', token_ids=[3])]) for p in prompts]
    policy.engine = NS(generate=generate)
    requests = [dict(prompt='hello', seed=seed, temperature=.4, top_p=1., max_new_tokens=512) for seed in (101, 202)]
    assert policy.generate_batch(requests) == [('go',2,1), ('go',2,1)]
    assert seen == [101,202]
    with pytest.raises(ValueError, match='boundary'):
        policy.generate('x', 101, .4, 1., 513)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))

    def load_weights(self, weights):
        loaded = set()
        for name, value in weights:
            with torch.no_grad():
                getattr(self, name).copy_(value)
            loaded.add(name)
        return loaded


def test_weight_loader_rejects_missing_parameters():
    with pytest.raises(RuntimeError, match='Incomplete'):
        bridge_module.load_complete_weights(FakeModel(), [])


def test_live_weight_sync_sleep_wake_and_no_per_state_transfer(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda:None)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda:0)
    monkeypatch.setattr(torch.random, 'fork_rng', lambda **k:nullcontext())
    monkeypatch.setattr(bridge_module.FSDP, 'state_dict_type', lambda *a, **k:nullcontext())
    events = []
    source = torch.tensor([1., 2.])
    target = FakeModel()
    manager = bridge_module.FSDPVLLMV1ShardingManager.__new__(bridge_module.FSDPVLLMV1ShardingManager)
    manager.module = NS(state_dict=lambda:{'weight':NS(full_tensor=lambda:source.clone())}, train=lambda:None)
    manager.inference_engine = NS(sleep=lambda **k:events.append('sleep'),
        wake_up=lambda **k:events.append(k['tags'][0]), reset_prefix_cache=lambda:True,
        apply_model=lambda fn:[fn(target)])
    manager.awake, manager.dirty, manager.offload_param = False, True, False
    manager.weight_version, manager.sync_count = 0, 0
    with manager:
        assert torch.equal(target.weight, source)
    assert manager.awake and manager.sync_count == 1
    with manager:
        pass
    assert manager.sync_count == 1 and events == ['weights', 'kv_cache']
    manager.suspend()
    source.add_(3)
    manager.mark_dirty()
    with manager:
        assert torch.equal(target.weight, source)
    assert manager.sync_count == 2
    with pytest.raises(RuntimeError):
        with manager:
            raise RuntimeError('fail closed')
    assert not manager.awake


def test_factory_never_falls_back_to_hf_on_vllm_failure(monkeypatch):
    from skillnet_cohort import inference
    monkeypatch.setattr('skillnet_cohort.vllm_backend.VLLMPolicy', lambda *a:(_ for _ in ()).throw(RuntimeError('bad version')))
    with pytest.raises(RuntimeError, match='bad version'):
        inference.make_policy('/not-loaded', {'inference_profile':registration(PROFILE)})
