"""Resident, batched, prefix-cached local Qwen3.5 router. No network/API."""
import contextlib
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import multiprocessing
import atexit
import signal

from webshop_phase12.assets import BASE_MODEL,RUN_ROOT,WebshopBank,digest
from webshop_phase12.llm_router import FrozenLLMSkillRouter,choice_labels,fixed_prefix


def model_identity():
    receipt=RUN_ROOT/'frozen-router-model.json'
    identity=json.loads(receipt.read_text())
    # Detect changed files without hashing the 8 GB weights at every process start.
    for row in identity['files']:
        stat=(BASE_MODEL/row['path']).stat()
        if stat.st_size!=row['size'] or stat.st_mtime_ns!=row['mtime_ns']:
            raise ValueError('Frozen router model snapshot changed; create a new protocol')
    return identity


class LocalVLLM:
    def __init__(self):
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        self.tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,local_files_only=True)
        encoded=[self.tokenizer.encode(label,add_special_tokens=False) for label in choice_labels()]
        if not all(len(tokens)==1 for tokens in encoded) or len({tokens[0] for tokens in encoded})!=54:
            raise ValueError('Frozen tokenizer does not support 54 distinct one-token labels')
        self.token_ids=[tokens[0] for tokens in encoded]
        self.identity={'snapshot':model_identity(),'torch':importlib.metadata.version('torch'),
            'transformers':importlib.metadata.version('transformers'),'vllm':importlib.metadata.version('vllm'),
            'dtype':'bfloat16','max_model_len':32768,'max_num_seqs':64,
            'max_num_batched_tokens':8192,'prefix_caching':True,'mamba_cache_mode':'align',
            'enforce_eager':True,'gpu_memory_utilization':0.22,'seed':0,'sleep_mode':'level1-offload-frozen-weights'}
        self.engine=LLM(model=str(BASE_MODEL),tokenizer=str(BASE_MODEL),dtype='bfloat16',
            tensor_parallel_size=1,max_model_len=32768,max_num_seqs=64,max_num_batched_tokens=8192,
            gpu_memory_utilization=.22,enable_prefix_caching=True,mamba_cache_mode='align',
            enable_chunked_prefill=True,enforce_eager=True,seed=0,disable_log_stats=False,
            limit_mm_per_prompt={'image':0,'video':0},enable_sleep_mode=True)
        self.sampling=SamplingParams(temperature=0.,max_tokens=1,allowed_token_ids=self.token_ids,
            ignore_eos=True,seed=0)
        self.stats={'batches':0,'unique_states':0,'input_tokens':0,'output_tokens':0,'inference_seconds':0.}
        # Qwen3.5 supports align-mode caching, not all-token Mamba caching.
        # Explicitly prefill a shared block-aligned catalogue prefix once.
        from webshop_phase12.llm_router import build_router_prompt
        from webshop_phase12.visible_state import VisibleMemory
        sample=self.tokenizer.apply_chat_template([{'role':'user','content':build_router_prompt(WebshopBank(),VisibleMemory('prefix probe',50).state())}],
            add_generation_prompt=True,tokenize=False,enable_thinking=False)
        prefix=sample.split('SHOPPING GOAL:')[0]
        tokens=self.tokenizer.encode(prefix,add_special_tokens=False)[:-1]
        block=self.engine.llm_engine.vllm_config.cache_config.block_size
        self.warm_prefix=tokens[:len(tokens)//block*block]
        self.warm_catalogue()

    def warm_catalogue(self):
        if self.warm_prefix:
            started=time.monotonic()
            self.engine.generate([{'prompt_token_ids':self.warm_prefix}],self.sampling,use_tqdm=False)
            print(json.dumps({'catalogue_prefix_warm_tokens':len(self.warm_prefix),
                'catalogue_prefix_warm_seconds':time.monotonic()-started}),file=sys.stderr,flush=True)

    def close(self):
        if getattr(self,'_closed',False):return
        self._closed=True
        self.engine.llm_engine.engine_core.shutdown(timeout=20)

    def sleep(self):
        self.engine.sleep(level=1)

    def wake(self):
        self.engine.wake_up()
        self.warm_catalogue()

    def select_many(self,prompts):
        rendered=[self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],
            add_generation_prompt=True,tokenize=False,enable_thinking=False) for prompt in prompts]
        # Tokenize once and pass IDs to vLLM; never truncate.
        encoded=self.tokenizer(rendered,add_special_tokens=False)['input_ids']
        if any(len(tokens)+1>32768 for tokens in encoded):
            raise ValueError('Frozen router prompt exceeds 32768 tokens; no truncation')
        if any(tokens[:len(self.warm_prefix)]!=self.warm_prefix for tokens in encoded):
            raise ValueError('Router prompt no longer has the registered fixed prefix')
        started=time.monotonic()
        outputs=self.engine.generate([{'prompt_token_ids':tokens} for tokens in encoded],
            self.sampling,use_tqdm=False)
        elapsed=time.monotonic()-started
        result=[]
        for expected,output in zip(encoded,outputs):
            ids=output.outputs[0].token_ids
            if len(ids)!=1 or ids[0] not in self.token_ids:raise ValueError('Invalid constrained router output')
            result.append({'label':choice_labels()[self.token_ids.index(ids[0])],
                'input_tokens':len(expected),'output_tokens':1,'latency_ms':elapsed*1000/len(prompts),
                'batch_size':len(prompts),'batch_latency_ms':elapsed*1000,
                'prefix_cached_tokens':int(output.num_cached_tokens or 0)})
        self.stats['batches']+=1;self.stats['unique_states']+=len(prompts)
        self.stats['input_tokens']+=sum(map(len,encoded));self.stats['output_tokens']+=len(prompts)
        self.stats['inference_seconds']+=elapsed
        print(json.dumps({'router_throughput':self.stats,'last_batch_states':len(prompts),
            'last_batch_seconds':elapsed,'states_per_second':len(prompts)/elapsed}),file=sys.stderr,flush=True)
        return result


def _replica_worker(gpu,connection,factory):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    backend=None
    try:
        backend=factory()
        connection.send({'identity':backend.identity})
        while True:
            request=connection.recv()
            if request is None:break
            if isinstance(request,dict):
                getattr(backend,request['command'])()
                connection.send({'ack':request['command']});continue
            connection.send({'outputs':backend.select_many(request)})
    except EOFError:pass
    except BaseException as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        connection.send({'error':type(error).__name__})
    finally:
        if backend is not None and hasattr(backend,'close'):backend.close()
        connection.close()


class ReplicaPool:
    """Parallel frozen replicas; caller deduplicates and caches before dispatch."""
    def __init__(self,gpus,*,factory=LocalVLLM):
        self.processes=[];self.connections=[];self._closed=False
        context=multiprocessing.get_context('spawn')
        for gpu in gpus:
            parent,child=context.Pipe()
            process=context.Process(target=_replica_worker,args=(gpu,child,factory))
            process.start();child.close()
            self.processes.append(process);self.connections.append(parent)
        try:
            identities=[self._receive(connection)['identity'] for connection in self.connections]
            if any(identity!=identities[0] for identity in identities):
                raise ValueError('Frozen-router replicas loaded different protocols/models')
            self.identity=identities[0]
            self.execution={'parallel_replicas':len(gpus),'dispatch':'length-balanced-original-order-restored'}
        except BaseException:
            self.close();raise

    def _receive(self,connection):
        response=connection.recv()
        if 'error' in response:raise RuntimeError('Frozen router replica failed: '+response['error'])
        return response

    def select_many(self,prompts):
        # Long inputs first, assigned to the currently shortest work queue.
        groups=[[] for _ in self.connections];loads=[0]*len(groups)
        for index in sorted(range(len(prompts)),key=lambda i:(-len(prompts[i]),i)):
            worker=min(range(len(groups)),key=lambda i:(loads[i],i))
            groups[worker].append(index);loads[worker]+=len(prompts[index])
        for connection,indices in zip(self.connections,groups):
            if indices:connection.send([prompts[i] for i in indices])
        result=[None]*len(prompts)
        for connection,indices in zip(self.connections,groups):
            if not indices:continue
            outputs=self._receive(connection)['outputs']
            if len(outputs)!=len(indices):raise ValueError('Router replica returned wrong result count')
            for index,output in zip(indices,outputs):result[index]=output
        return result

    def control(self,command):
        for connection in self.connections:connection.send({'command':command})
        for connection in self.connections:
            if self._receive(connection).get('ack')!=command:raise ValueError('Router replica lifecycle acknowledgement failed')

    def sleep(self):self.control('sleep')
    def wake(self):self.control('wake')

    def close(self):
        if self._closed:return
        self._closed=True
        for process,connection in zip(self.processes,self.connections):
            if process.is_alive():
                try:connection.send(None)
                except (BrokenPipeError,EOFError):pass
        deadline=time.monotonic()+45
        for process,connection in zip(self.processes,self.connections):
            process.join(timeout=max(0,deadline-time.monotonic()))
            if process.is_alive():process.terminate();process.join(timeout=10)
            connection.close()


def load_backend():
    gpus=[int(gpu) for gpu in os.environ.get('WEBSHOP_ROUTER_GPUS',os.environ['WEBSHOP_ROUTER_GPU']).split(',')]
    return ReplicaPool(gpus) if len(gpus)>1 else LocalVLLM()


def main():
    # vLLM uses subprocesses that can write directly to fd 1. Reserve an
    # exclusive protocol pipe, redirect all library/child stdout to the log.
    protocol_output=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    def stop(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop)
    with contextlib.redirect_stdout(sys.stderr):
        backend=load_backend()
        atexit.register(backend.close)
        router=FrozenLLMSkillRouter(os.environ['WEBSHOP_ROUTER_CACHE'],backend)
        print(json.dumps({'router_ready':True,'protocol':router.protocol,'pid':os.getpid()}),file=sys.stderr,flush=True)
    try:
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('close'):break
            if request.get('command')=='ready':
                print(json.dumps({'ack':'ready'}),file=protocol_output,flush=True)
                continue
            if request.get('command') in ('sleep','wake'):
                getattr(backend,request['command'])()
                print(json.dumps({'ack':request['command']}),file=protocol_output,flush=True)
                continue
            started=time.monotonic()
            with contextlib.redirect_stdout(sys.stderr):bundles=router.route_many(request['states'])
            print(json.dumps({'bundles':bundles,'wall_seconds':time.monotonic()-started},ensure_ascii=False),file=protocol_output,flush=True)
    finally:
        router.close()
        if hasattr(backend,'close'):backend.close()
        protocol_output.close()


if __name__=='__main__':main()
