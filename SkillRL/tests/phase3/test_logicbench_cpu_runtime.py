import json
import os
from pathlib import Path

import pytest

from phase3.common import ProtocolError


def cpu_settings():
    config=json.loads(Path('configs/phase3_logicbench_50updates_s707.json').read_text())['router']
    return {**config,'device':'cpu','python_executable':'/home/wangyifan/.venvs/skillrl-embedding/bin/python',
        'runtime_variant':'cpu_torch_2_11_0_v1','execution':{'mode':'state_batch_fp32_v1','intra_op_threads':8}}


def test_cpu_runtime_is_explicit_and_cannot_leak_into_gpu():
    from phase3.embedding_routing import validate_settings
    config=cpu_settings()
    profile,_=validate_settings(config)
    assert profile.torch_version=='2.11.0+cpu'
    with pytest.raises(ProtocolError):
        validate_settings({**config,'device':'cuda:0'})


def test_cpu_pool_uses_selected_interpreter_in_isolated_sidecar(tmp_path,monkeypatch):
    from phase3 import gpu_encoder_service
    from phase3.embedding_routing import EmbeddingRouterPool
    made=[]
    class Proxy:
        def __init__(self,**kwargs): made.append(kwargs)
        def close(self): pass
    monkeypatch.setattr(gpu_encoder_service,'GPUEncoderProxy',Proxy)
    monkeypatch.setattr(gpu_encoder_service,'_SHARED_ENCODERS',{})
    pool=EmbeddingRouterPool(cpu_settings(),tmp_path/'cpu.sqlite3','readout_d')
    assert made[0]['device']=='cpu'
    assert made[0]['python_executable']==cpu_settings()['python_executable']
    assert pool.config.torch_version=='2.11.0+cpu'


def test_secret_file_only_loaded_when_environment_missing(tmp_path,monkeypatch):
    from phase3.logicbench_loop import load_editor_credential
    p=tmp_path/'editor.key';p.write_text('sk-fixture-secret\n');p.chmod(0o600)
    monkeypatch.delenv('SKILLRL_PHASE3_EDITOR_API_KEY',raising=False)
    assert load_editor_credential(p) is True
    assert os.environ['SKILLRL_PHASE3_EDITOR_API_KEY']=='sk-fixture-secret'
    monkeypatch.setenv('SKILLRL_PHASE3_EDITOR_API_KEY','existing')
    assert load_editor_credential(p) is False
    assert os.environ['SKILLRL_PHASE3_EDITOR_API_KEY']=='existing'
    monkeypatch.delenv('SKILLRL_PHASE3_EDITOR_API_KEY')
    p.chmod(0o644)
    with pytest.raises(ProtocolError): load_editor_credential(p)


def test_installed_sdk_constructs_editor_schema_request_offline(tmp_path):
    import httpx,openai
    from phase3.api import APIConfig,JSONClient
    captured=[]
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200,json={'id':'offline-fixture','object':'chat.completion','created':0,
            'model':'gpt-5.5','choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'{"ok":true}'}}],
            'usage':{'prompt_tokens':20,'completion_tokens':5,'total_tokens':25}})
    http=httpx.Client(transport=httpx.MockTransport(handler))
    client=openai.OpenAI(api_key='offline-fixture',base_url='https://offline.invalid/v1',http_client=http,max_retries=0)
    api=JSONClient(APIConfig(stage='editor',model='gpt-5.5',max_input_tokens=1,max_completion_tokens=8192,
        max_api_calls=1,sdk_version=openai.__version__),tmp_path/'editor.sqlite3',client=client,token_counter=lambda _:100)
    value,_=api.request(identity={'fixture':True},system='fixture',payload={'x':1},
        schema={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False},
        validate=lambda v:v['ok'])
    assert value=={'ok':True}
    assert captured[0]['reasoning_effort']=='medium'
    assert captured[0]['response_format']['json_schema']['strict'] is True
    assert captured[0]['max_completion_tokens']==8192 and captured[0]['store'] is False
    client.close()


def test_router_readiness_probe_does_not_inherit_editor_secret(tmp_path,monkeypatch):
    import subprocess,torch
    from phase3.logicbench_loop import readiness
    config=json.loads(Path('configs/phase3_logicbench_50updates_s707_cpu_v2.json').read_text())
    monkeypatch.setenv('SKILLRL_PHASE3_EDITOR_API_KEY','sk-private-test-fixture')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:0)
    def probe(*args,**kwargs):
        assert 'SKILLRL_PHASE3_EDITOR_API_KEY' not in kwargs['env']
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']==''
        return subprocess.CompletedProcess(args[0],0,stdout=json.dumps({'cuda':False,'torch':'2.11.0+cpu',
            'sentence_transformers':'6.0.1','transformers':'5.10.4','tokenizers':'0.22.2'}))
    monkeypatch.setattr(subprocess,'run',probe)
    monkeypatch.setattr('skillnet_cohort.runtime.disk_gate',lambda *args,**kwargs: {'free_bytes':200*2**30})
    assert readiness(config,tmp_path,[0,1,2,3])['blockers']==['gpu_access']
