"""CPU regression for one-rank rendezvous and the explicit seed505 continuation."""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from skillnet_cohort.common import REPO, file_hash, read_json, write_new_json
from skillnet_cohort import first_calls_port_recovery as recovery


def config(**changes):
    parallel = dict(world_size=1, world_size_across_dp=1, tensor_parallel_size=1,
                    pipeline_parallel_size=1, data_parallel_size=1, nnodes=1)
    parallel.update(changes)
    return NS(parallel_config=NS(**parallel), device_config=NS(device='cuda'))


def test_actual_vllm_executor_resolves_custom_import_and_preserves_worker_methods(tmp_path, monkeypatch):
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from skillnet_cohort.vllm_file_executor import FileStoreUniProcExecutor
    value = config(distributed_executor_backend=recovery.EXECUTOR)
    assert Executor.get_class(value) is FileStoreUniProcExecutor
    assert FileStoreUniProcExecutor._init_executor is UniProcExecutor._init_executor
    assert FileStoreUniProcExecutor.execute_model is UniProcExecutor.execute_model
    assert FileStoreUniProcExecutor.sample_tokens is UniProcExecutor.sample_tokens
    obj = FileStoreUniProcExecutor.__new__(FileStoreUniProcExecutor)
    obj.vllm_config = value
    record = tmp_path/'rendezvous.json'; monkeypatch.setenv(recovery.RECORD_ENV, str(record))
    method, rank, local_rank = obj._distributed_args()
    assert method.startswith('file:///tmp/skillscope-eval-rdzv-')
    assert (rank, local_rank) == (0, 0)
    assert read_json(record)['tcp_rendezvous_port'] is None
    assert not Path(method.removeprefix('file://')).exists()


@pytest.mark.parametrize('field', ['world_size', 'world_size_across_dp', 'tensor_parallel_size',
                                   'pipeline_parallel_size', 'data_parallel_size', 'nnodes'])
def test_multi_rank_or_multi_node_fails_before_rendezvous(tmp_path, field):
    from skillnet_cohort.vllm_file_executor import file_rendezvous_args
    with pytest.raises(ValueError, match='one local'):
        file_rendezvous_args(config(**{field: 2}), tmp_path/'record.json')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('device', ['cpu', 'cuda:1', 'cuda:4'])
def test_wrong_visible_device_fails_before_rendezvous(tmp_path, device):
    from skillnet_cohort.vllm_file_executor import file_rendezvous_args
    value = config(); value.device_config.device = device
    with pytest.raises(ValueError):
        file_rendezvous_args(value, tmp_path/'record.json')


def test_eight_unique_paths_and_existing_attempt_not_reused(tmp_path, monkeypatch):
    import vllm.v1.executor.uniproc_executor as uni
    from skillnet_cohort.vllm_file_executor import file_rendezvous_args
    monkeypatch.setattr(uni, 'get_open_port', lambda: pytest.fail('No TCP port probe is permitted'))
    methods = [file_rendezvous_args(config(), tmp_path/f'{i}.json')[0] for i in range(8)]
    assert len(set(methods)) == 8
    old = (tmp_path/'0.json').read_bytes()
    with pytest.raises(FileExistsError):
        file_rendezvous_args(config(), tmp_path/'0.json')
    assert (tmp_path/'0.json').read_bytes() == old


@pytest.mark.parametrize('kind', ['relative', 'missing_parent', 'symlink'])
def test_rendezvous_audit_path_is_explicit_and_not_clobbered(tmp_path, kind):
    from skillnet_cohort.vllm_file_executor import file_rendezvous_args
    path = tmp_path/'record.json'
    if kind == 'relative': path = Path('relative-record.json')
    if kind == 'missing_parent': path = tmp_path/'missing/record.json'
    if kind == 'symlink': path.symlink_to(tmp_path/'missing')
    with pytest.raises(FileExistsError):
        file_rendezvous_args(config(), path)


def test_real_eight_process_gloo_file_stores_while_tcp_port_is_occupied(tmp_path):
    from skillnet_cohort.vllm_file_executor import file_rendezvous_args
    methods = [file_rendezvous_args(config(), tmp_path/f'cpu-{i}.json')[0] for i in range(8)]
    code = '''import sys, torch
import torch.distributed as dist
from datetime import timedelta
dist.init_process_group('gloo', init_method=sys.argv[1], world_size=1, rank=0, timeout=timedelta(seconds=30))
x=torch.tensor([int(sys.argv[2])]); dist.all_reduce(x)
assert x.item()==int(sys.argv[2])
dist.destroy_process_group()
print('FILESTORE_PASS', flush=True)
'''
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0)); occupied.listen()
        # MASTER_PORT is deliberately unusable; file:// does not use it.
        env.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(occupied.getsockname()[1]))
        children = [subprocess.Popen([sys.executable, '-B', '-c', code, method, str(i)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                    for i, method in enumerate(methods)]
        try:
            for child in children:
                stdout, stderr = child.communicate(timeout=50)
                assert child.returncode == 0, stderr
                assert 'FILESTORE_PASS' in stdout
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate(); child.wait(timeout=10)


def test_builder_changes_only_executor_not_model_sampling_or_engine_settings(monkeypatch):
    from skillnet_cohort.inference import PROFILE, registration
    from skillnet_cohort.vllm_backend import build_engine
    calls = []; registrations = []
    fake = NS(LLM=lambda **kw: calls.append(kw) or NS(),
              ModelRegistry=NS(register_model=lambda *a: registrations.append(a)))
    monkeypatch.setitem(sys.modules, 'vllm', fake)
    profile = registration(PROFILE)
    build_engine('/same-retained-model', profile)
    recovery.build_file_engine('/same-retained-model', profile)
    before, after = calls
    assert before.pop('distributed_executor_backend') == 'uni'
    assert after.pop('distributed_executor_backend') == recovery.EXECUTOR
    assert before == after
    assert registrations[0] == registrations[1]


@pytest.fixture
def bound(tmp_path):
    output = tmp_path/'recovery-v3'; output.mkdir()
    source = tmp_path/'followup-s505/plan.json'; root = source.parent/'seed-505'
    write_new_json(root/'protocol.json', {'runtime': {'router_backend': 'skillrl_embedding_state_batch', 'max_api_calls': 0}})
    write_new_json(source, {'jobs': [{'seed': 505, 'run_root': str(root)}]})
    previous = tmp_path/'recovery-v2/plan.json'; write_new_json(previous, {})
    tests = tmp_path/'tests.xml'; tests.write_text('passed')
    dependency = tmp_path/'installed.py'; dependency.write_text('unchanged')
    plan = {'schema_version': recovery.SCHEMA, 'approved': True, 'root': str(output),
        'seeds': [505], 'automatic_retry': False, 'external_api_calls': 0,
        'training_reexecuted': False, 'seed606_storage_waiver': False,
        'assessment_plan': {'path': str(source), 'sha256': file_hash(source)},
        'previous_attempt': {'path': str(previous), 'sha256': file_hash(previous)},
        'engineering_tests': {'path': str(tests), 'sha256': file_hash(tests)},
        'dependency_sources': [{'path': str(dependency), 'sha256': file_hash(dependency)}],
        'executor': recovery.EXECUTOR, 'protocol_sha256': file_hash(root/'protocol.json')}
    write_new_json(output/'plan.json', plan)
    return output/'plan.json', plan, root


def test_valid_binding_is_scoped_to_existing_505_only(bound):
    path, plan, root = bound
    actual, source = recovery.binding(path)
    assert actual == plan and source['jobs'][0]['run_root'] == str(root)


@pytest.mark.parametrize('field,value', [('approved', False), ('seeds', [404,505]),
    ('automatic_retry', True), ('external_api_calls', 1), ('training_reexecuted', True),
    ('seed606_storage_waiver', True), ('executor', 'uni'), ('schema_version', 'unknown')])
def test_scope_expansion_rejected(bound, field, value):
    path, plan, _ = bound; plan[field] = value; path.write_text(json.dumps(plan))
    with pytest.raises(PermissionError): recovery.binding(path)


@pytest.mark.parametrize('target', ['assessment_plan', 'previous_attempt', 'engineering_tests', 'dependency_sources'])
def test_changed_source_or_test_evidence_rejected(bound, target):
    path, plan, _ = bound
    item = plan[target][0] if target == 'dependency_sources' else plan[target]
    Path(item['path']).write_text('changed')
    with pytest.raises(ValueError): recovery.binding(path)


@pytest.mark.parametrize('update,shard,gpu', [(1,0,'0'), (5,8,'8'), (0,1,'2')])
def test_evaluator_rejects_wrong_endpoint_or_gpu_before_model(bound, monkeypatch, update, shard, gpu):
    path, _, root = bound; monkeypatch.setenv('CUDA_VISIBLE_DEVICES', gpu)
    monkeypatch.setattr(recovery, 'build_file_engine', lambda *a: pytest.fail('Must fail before GPU'))
    with pytest.raises(PermissionError): recovery.evaluate(path, root, update, shard)


def test_evaluator_uses_existing_generate_implementation_and_restores_class(bound, monkeypatch):
    from skillnet_cohort import vllm_backend
    from phase2 import evaluate as original_evaluator
    path, plan, root = bound; monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    original = vllm_backend.VLLMPolicy; observed = []
    def engine(*args):
        write_new_json(Path(os.environ[recovery.RECORD_ENV]), {'file': 'test'})
        return NS(get_tokenizer=lambda: 'same-tokenizer')
    monkeypatch.setattr(recovery, 'build_file_engine', engine)
    def main():
        assert vllm_backend.VLLMPolicy.generate is original.generate
        assert vllm_backend.VLLMPolicy.generate_batch is original.generate_batch
        policy = vllm_backend.VLLMPolicy('retained', {})
        assert policy.tokenizer == 'same-tokenizer'
        observed.append(list(sys.argv))
    monkeypatch.setattr(original_evaluator, 'main', main)
    old_argv = sys.argv
    recovery.evaluate(path, root, 5, 2)
    assert vllm_backend.VLLMPolicy is original and sys.argv is old_argv
    assert observed == [['phase2.evaluate', '--root', str(root), '--update', '5', '--shards', '8', '--shard', '2']]
    assert read_json(Path(plan['root'])/'seed-505/rendezvous/u0005-shard2-ready.json')['state'] == 'ENGINE_READY'
    with pytest.raises(FileExistsError): recovery.evaluate(path, root, 5, 2)


def test_evaluator_restores_policy_on_error(bound, monkeypatch):
    from skillnet_cohort import vllm_backend
    from phase2 import evaluate as original_evaluator
    path, _, root = bound; monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    original = vllm_backend.VLLMPolicy
    def fail(): raise RuntimeError('preserved failure')
    monkeypatch.setattr(original_evaluator, 'main', fail)
    with pytest.raises(RuntimeError): recovery.evaluate(path, root, 0, 0)
    assert vllm_backend.VLLMPolicy is original


def assessment(tmp_path):
    obj = recovery.PortRecoveryAssessment.__new__(recovery.PortRecoveryAssessment)
    obj.root = tmp_path/'run'; obj.audit = tmp_path/'audit'; obj.recovery = {'root': str(tmp_path), 'retained': {}}
    obj.plan = {}; obj.disk = lambda *a: None
    return obj


def job(obj, rank=0, update=0):
    return (['phase2.evaluate', '--root', obj.root, '--update', update, '--shards', 8, '--shard', rank],
            f'utility-u{update:04d}-shard{rank}', rank)


@pytest.mark.parametrize('module', ['verl.trainer.main_ppo', 'phase2.export_model',
                                   'skillnet_cohort.first_calls_measure', 'phase2.aggregate'])
def test_training_export_or_completed_readout_cannot_be_launched(tmp_path, module):
    obj = assessment(tmp_path)
    with pytest.raises(PermissionError): obj.commands([([module], 'forbidden', 0)])


def test_cache_failure_blocks_before_any_gpu_launch(tmp_path, monkeypatch):
    obj = assessment(tmp_path); monkeypatch.setattr(recovery, 'binding', lambda p: None)
    def rejected(*a): raise PermissionError('unapproved cache interruption')
    monkeypatch.setattr(recovery, 'audit_cache', rejected)
    monkeypatch.setattr(recovery.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Must stop before GPU'))
    with pytest.raises(PermissionError): obj.commands([job(obj)])


def test_child_failure_stops_only_new_children_and_never_retries(tmp_path, monkeypatch):
    from skillnet_cohort import runtime_watch
    obj = assessment(tmp_path); monkeypatch.setattr(recovery, 'binding', lambda p: None)
    monkeypatch.setattr(recovery, 'audit_cache', lambda p: {'status':'PASS'})
    monkeypatch.setattr(runtime_watch, 'StageWatch', lambda *a: NS(tick=lambda *a, **kw: None))
    children = []; signals = []
    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 99000+len(children); self.returncode = None if not children else 1
            self.command = command; children.append(self)
            assert kwargs['start_new_session'] and '--evaluate' in command
        def poll(self): return self.returncode
        def wait(self, timeout=None): return self.returncode
    def kill(pid, sig):
        signals.append((pid,sig)); children[pid-99000].returncode = -15
    monkeypatch.setattr(recovery.subprocess, 'Popen', Child)
    monkeypatch.setattr(recovery.os, 'killpg', kill)
    with pytest.raises(RuntimeError, match='no automatic retry'):
        obj.commands([job(obj,0), job(obj,7)])
    assert len(children) == 2 and len(signals) == 1 and signals[0][0] == 99000
    assert children[1].command[-1] == '7'
    assert (obj.audit/'logs/utility-u0000-shard7.log').exists()


def test_resume_order_keeps_readout_and_404_untouched(tmp_path, monkeypatch):
    from skillnet_cohort import first_calls_run, first_calls_reuse, first_calls_report, window_storage
    obj = assessment(tmp_path); events = []
    obj.evaluate = lambda u: events.append(('utility',u))
    monkeypatch.setattr(first_calls_run, 'lock_prediction', lambda r: events.append(('lock',)))
    monkeypatch.setattr(first_calls_reuse, 'import_endpoint', lambda r,u: events.append(('reuse',u)))
    monkeypatch.setattr(first_calls_report, 'report', lambda r: events.append(('report',)))
    monkeypatch.setattr(window_storage, 'seal_window', lambda r,s,e: events.append(('seal',s,e)))
    monkeypatch.setattr(recovery, 'verify_retained', lambda r: events.append(('verify_old',)))
    monkeypatch.setattr(first_calls_report, 'publish', lambda r: events.append(('publish',)))
    obj.resume505()
    assert events == [('utility',0),('lock',),('reuse',5),('utility',5),('report',),('seal',0,5),('verify_old',),('publish',)]
    assert read_json(obj.root/'complete.json')['training_reexecuted'] is False


def test_invalid_attempt_directory_fails_without_preparing_anything(tmp_path):
    with pytest.raises(FileExistsError):
        recovery.prepare(tmp_path/'wrong/plan.json', tmp_path/'bad-output', tmp_path/'tests.xml')
    assert not list(tmp_path.iterdir())
