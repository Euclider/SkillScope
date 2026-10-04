"""Synthetic eight-GPU cache handoff plus real frozen-router parity; no RL."""
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch

BASE = Path('/mnt/workspace/users/wangyifan/phase3-speed-742f1aa-BlbHzX')
CANDIDATE = BASE / 'candidate-router-handoff-v2'
OUTPUT = BASE / 'router-handoff-acceptance-v2'
sys.path.insert(0, str(CANDIDATE))
from phase3.common import require, write_new
from alfworld_rl_speed.launch import fingerprint
from skillnet_cohort.common import file_hash
from verl.utils.fsdp_utils import release_cuda_cache_after


def main():
    OUTPUT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    frozen = fingerprint()
    reports = []
    for rank in range(8):
        torch.cuda.set_device(rank)
        held = torch.empty(18 * 2**30, dtype=torch.uint8, device='cuda')
        parameter = torch.nn.Parameter(torch.ones(1, device='cuda'))
        optimizer = torch.optim.AdamW([parameter])
        optimizer.state[parameter]['exp_avg'] = torch.full((2**30,), .125, device='cuda')
        temporary = torch.empty(8 * 2**30, dtype=torch.uint8, device='cuda')
        del temporary
        torch.cuda.synchronize()
        before = torch.cuda.mem_get_info()[0]
        rng = torch.cuda.get_rng_state().clone()
        release_cuda_cache_after(lambda: None)()
        after = torch.cuda.mem_get_info()[0]
        require(after - before > 7 * 2**30, 'Cached pages were not released')
        require(torch.equal(torch.cuda.get_rng_state(), rng), 'Memory handoff changed CUDA RNG')
        require(bool((optimizer.state[parameter]['exp_avg'] == .125).all()), 'Adam values changed')
        require(optimizer.state[parameter]['exp_avg'].is_cuda, 'Optimizer was offloaded')
        reports.append({'gpu': rank, 'free_before': before, 'free_after': after,
                        'rng_unchanged': True, 'adam_values_unchanged': True})
        print('CACHE_HANDOFF_PASSED', rank, before, after, flush=True)
        del optimizer, parameter, held
        gc.collect(); torch.cuda.empty_cache()
    # Same actual index strings, frozen FP32 encoder, two-text microbatches.
    run = Path('/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v7-20260927/runs/skillrl_failure')
    cfg = json.loads((run / 'segments/u0025-u0030.json').read_text())['env']['skills_only_memory']['step_routing']
    from phase3.gpu_encoder_service import GPUEncoderProxy
    proxy = GPUEncoderProxy(model_path=cfg['model_path'], profile_sha256=cfg['profile_sha256'],
        intra_op_threads=1, shared_gpu_physical_id=7, ledger_path=OUTPUT/'smoke.sqlite3',
        forward_microbatch_size=2)
    from phase3.bank import Bank
    from phase3.routing import BranchMemory
    from agent_system.memory.skillrl_embedding_router import skill_texts
    bank = Bank.load(cfg['bank_path'], cfg['bank_sha256'])
    texts = skill_texts(BranchMemory(bank))
    try:
        baseline, _ = proxy.encode(texts)
        proxy.close()
        torch.cuda.set_device(7)
        held = torch.empty(18 * 2**30, dtype=torch.uint8, device='cuda')
        parameter = torch.nn.Parameter(torch.ones(1, device='cuda'))
        optimizer = torch.optim.AdamW([parameter])
        optimizer.state[parameter]['exp_avg'] = torch.full((2**30,), .125, device='cuda')
        temporary = torch.empty(8 * 2**30, dtype=torch.uint8, device='cuda')
        del temporary
        release_cuda_cache_after(lambda: None)()
        actual, _ = proxy.encode(texts)
        import numpy as np
        require(np.array_equal(actual, baseline), 'Router vectors changed under memory pressure')
        del held, optimizer, parameter
        gc.collect(); torch.cuda.empty_cache()
    finally:
        proxy.close()
    prior = BASE / 'finite-exp-acceptance-v3/receipt.json'
    receipt = json.loads(prior.read_text())
    changed = [k for k,v in frozen.items() if receipt['source_hashes'].get(k) != v]
    require(set(changed) == {'verl/utils/fsdp_utils.py', 'verl/workers/fsdp_workers.py'}
            and fingerprint() == frozen, 'Unexpected deployment delta')
    write_new(OUTPUT/'receipt.json', {'passed': True, 'source_hashes': frozen,
        'inherited_numerical_receipt_sha256': file_hash(prior), 'memory_reports': reports,
        'router_index_vectors_exact': True, 'router_text_count': len(texts),
        'scope': 'Synthetic 22GiB live including GPU Adam + 8GiB cached pressure; real router parity; no RL update'})
    print('ROUTER_HANDOFF_ACCEPTED', flush=True)


if __name__ == '__main__':
    main()
