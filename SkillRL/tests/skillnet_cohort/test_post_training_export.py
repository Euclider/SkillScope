"""Post-training continuation safety: CPU only, no ALFWorld or GPU experiment."""
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from phase2 import export_model
from skillnet_cohort.common import REPO, write_new_json
from skillnet_cohort.post_training_recovery import consumed_seconds


def test_merger_module_resolves_local_verl_without_pythonpath():
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = subprocess.run([sys.executable, '-B', '-m', 'scripts.model_merger', 'merge', '--help'],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert 'output_dtype' in result.stdout
    command = export_model.merger_command(Path('/native'), Path('/staging'))
    assert command[:5] == [sys.executable, '-B', '-m', 'scripts.model_merger', 'merge']
    assert command[-2:] == ['--output_dtype', 'float32']


def test_export_preserves_failed_staging_and_requires_parity(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from phase2 import export_verify
    native, target = tmp_path/'native', tmp_path/'models/u0005'
    old_partial = target.with_name('u0005.partial')
    old_partial.mkdir(parents=True)
    staging = tmp_path/'new-attempt/u0005.partial'
    monkeypatch.setattr(export_model, 'validate_full_checkpoint', lambda p: {'world_size': 8})
    calls = []
    def merge(command, **kwargs):
        calls.append(command)
        save_file({f'tensor{i}': torch.ones(1) for i in range(400)}, staging/'model.safetensors')
    monkeypatch.setattr(export_model.subprocess, 'run', merge)
    monkeypatch.setattr(export_verify, 'verify', lambda c, t: {'status': 'PASS', 'all_parameters_bitwise_equal': True})
    export_model.export(native, target, temporary=staging)
    assert old_partial.is_dir() and not list(old_partial.iterdir())
    assert not staging.exists()
    assert json.loads((target/'phase2_export.json').read_text())['native_weight_parity']['status'] == 'PASS'
    assert len(calls) == 1


def test_export_does_not_overwrite_or_publish_failed_verification(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from phase2 import export_verify
    native, target, staging = tmp_path/'native', tmp_path/'u0005', tmp_path/'staging'
    monkeypatch.setattr(export_model, 'validate_full_checkpoint', lambda p: {'world_size': 8})
    target.mkdir()
    with pytest.raises(FileExistsError):
        export_model.export(native, target, temporary=staging)
    other = tmp_path/'other'
    staging.mkdir()
    with pytest.raises(FileExistsError):
        export_model.export(native, other, temporary=staging)
    fresh = tmp_path/'fresh'
    def merge(*args, **kwargs):
        save_file({f'tensor{i}': torch.ones(1) for i in range(400)}, fresh/'model.safetensors')
    def reject(*args):
        raise ValueError('parity mismatch')
    monkeypatch.setattr(export_model.subprocess, 'run', merge)
    monkeypatch.setattr(export_verify, 'verify', reject)
    with pytest.raises(ValueError, match='parity mismatch'):
        export_model.export(native, other, temporary=fresh)
    assert not other.exists() and (fresh/'model.safetensors').is_file()
    assert not (fresh/'phase2_export.json').exists()


def _save_tiny_native(rank, directory):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardedStateDictConfig, StateDictType
    from transformers import AutoModelForCausalLM, Qwen3_5TextConfig
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{directory}/rendezvous',
        rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(404)
        config = Qwen3_5TextConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            layer_types=['full_attention']*2, tie_word_embeddings=False)
        model = AutoModelForCausalLM.from_config(config)
        actor = Path(directory)/'native/actor'
        if rank == 0:
            actor.mkdir(parents=True)
            config.save_pretrained(actor/'huggingface')
            torch.save(model.state_dict(), Path(directory)/'reference.pt')
        dist.barrier()
        mesh = init_device_mesh('cpu', (2,), mesh_dim_names=('fsdp',))
        wrapped = FSDP(model, device_id=torch.device('cpu'), device_mesh=mesh, use_orig_params=False)
        with FSDP.state_dict_type(wrapped, StateDictType.SHARDED_STATE_DICT,
                                  ShardedStateDictConfig(offload_to_cpu=True)):
            torch.save(wrapped.state_dict(), actor/f'model_world_size_2_rank_{rank}.pt')
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_real_two_rank_dtensor_export_is_bitwise_exact_and_rejects_corruption(tmp_path, monkeypatch):
    from safetensors.torch import load_file, save_file
    from phase2.export_verify import verify
    from scripts import model_merger
    torch.multiprocessing.start_processes(_save_tiny_native, args=(str(tmp_path),),
        nprocs=2, join=True, start_method='spawn')
    native, target = tmp_path/'native', tmp_path/'export'
    monkeypatch.setattr(model_merger, 'hf_processor', lambda *a: None)
    monkeypatch.setattr(model_merger, 'hf_tokenizer', lambda *a: None)
    config = model_merger.ModelMergerConfig(operation='merge', backend='fsdp',
        local_dir=str(native/'actor'), hf_model_config_path=str(native/'actor/huggingface'),
        target_dir=str(target), output_dtype='float32')
    model_merger.FSDPModelMerger(config).merge_and_save()
    monkeypatch.setattr('phase1.watch_qwen35_checkpoints.validate_full_checkpoint', lambda p: {'world_size': 2})
    result = verify(native, target)
    assert result['all_parameters_bitwise_equal'] and result['optimizer_updates'] == 0
    exported = load_file(target/'model.safetensors')
    reference = torch.load(tmp_path/'reference.pt', weights_only=True)
    assert set(exported) == set(reference)
    assert all(torch.equal(exported[k], reference[k]) for k in reference)
    name = sorted(exported)[0]
    exported[name].reshape(-1)[0] += 1
    save_file(exported, target/'model.safetensors')
    with pytest.raises(ValueError, match='differs'):
        verify(native, target)


def test_post_training_pipeline_never_retrains_or_reexports_u0(tmp_path, monkeypatch):
    from skillnet_cohort.run import Pipeline
    p = Pipeline.__new__(Pipeline)
    p.root, p.audit_root = tmp_path/'source', tmp_path/'attempt'
    p.preparation, p.authorization = tmp_path/'preparation.json', tmp_path/'permit.json'
    write_new_json(p.preparation, {})
    write_new_json(p.authorization, {})
    write_new_json(p.root/'metrics/u0005.json', {})
    write_new_json(p.root/'launch.json', {'retained': True})
    p.recovery = p.post_training = {'skip_training': True}
    p.router_backend, p.deadline = 'skillrl_embedding_state', None
    p.spec = {'training': {'iterations': 5}, 'seed': 404,
              'evaluation': {'splits': ['valid_seen', 'valid_unseen'], 'utility_splits': ['valid_unseen']}}
    p.permit = {'gpu_ids': list(range(8))}
    p.budget = lambda: {'paid_router_cost': 0}
    commands, evaluations = [], []
    p.command = lambda args, label, **kw: commands.append((args, label))
    p.evaluate = lambda *args: evaluations.append(args) or tmp_path/'evaluations'
    monkeypatch.setattr('phase1.watch_qwen35_checkpoints.validate_full_checkpoint', lambda p: {'world_size': 8})
    monkeypatch.setattr('skillnet_cohort.support.build_support', lambda *args: {'anchor_sets': []})
    with pytest.raises(ValueError, match='No naturally supported'):
        p.run()
    assert len(commands) == 1 and commands[0][1] == 'export-u0005'
    assert evaluations == [(0, 'valid_seen', 'performance'), (0, 'valid_unseen', 'performance'),
                           (0, 'valid_unseen', 'anchors', None)]
    with pytest.raises(FileExistsError, match='already attempted'):
        p.run()


def test_post_training_budget_counts_prior_active_once(tmp_path):
    prior = tmp_path/'recovery-v2'
    write_new_json(prior/'seed-404-exit.json', {'exit_code': 1, 'completed': False, 'elapsed_seconds': 41864.4})
    write_new_json(prior/'seed-404/supervisor_launch.json', {'started_unix': 100.25})
    write_new_json(prior/'queue_launch.json', {'actual_attempt_started_unix': 100.0})
    assert consumed_seconds(tmp_path) == pytest.approx(41864.65)
