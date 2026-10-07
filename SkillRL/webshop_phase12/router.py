import json
import os
import subprocess
import time
import sys
from pathlib import Path
from webshop_phase12.assets import ROOT, RUN_ROOT


class RouterClient:
    def __init__(self, label):
        if 'WEBSHOP_ROUTER_GPU' not in os.environ:raise ValueError('Frozen LLM router GPU must be explicitly allocated')
        self.label=label
        self.output_root=Path(os.environ['WEBSHOP_ROUTER_CACHE']).parent
        self.log = (self.output_root/f'router-service-{label}.log').open('a')
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=os.environ.get('WEBSHOP_ROUTER_GPUS',os.environ['WEBSHOP_ROUTER_GPU']),PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        self.process = subprocess.Popen([os.environ.get('WEBSHOP_ROUTER_PYTHON',sys.executable),'-B','-u','-m','webshop_phase12.llm_service'],
            cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        self.sleeping=False

    def control(self,command):
        self.process.stdin.write(json.dumps({'command':command})+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line or json.loads(line).get('ack')!=command:
            raise RuntimeError('Frozen router lifecycle operation failed; inspect '+self.log.name)

    def sleep(self):
        if not self.sleeping:self.control('sleep');self.sleeping=True

    def route_many(self, states):
        if not states:
            return []
        wake_seconds=0.
        if self.sleeping:
            wake_started=time.monotonic();self.control('wake');self.sleeping=False
            wake_seconds=time.monotonic()-wake_started
        started=time.monotonic()
        self.process.stdin.write(json.dumps({'states':states},ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Frozen LLM router failed, code={self.process.poll()}; inspect {self.log.name}')
        response=json.loads(line);bundles=response['bundles']
        self.last_server_seconds=response['wall_seconds']+wake_seconds
        print(json.dumps({'router_request_states':len(states),'router_wall_seconds':time.monotonic()-started,
            'router_cache_hits':sum(bool(b['skill_router_api']['cache_hit']) for b in bundles)}),file=sys.stderr,flush=True)
        return bundles

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"close":true}\n'); self.process.stdin.flush()
                self.process.wait(timeout=60)
            except (BrokenPipeError,subprocess.TimeoutExpired):
                self.process.terminate();self.process.wait(timeout=30)
        self.log.close()
