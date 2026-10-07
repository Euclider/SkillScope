"""Explicit new-cohort input contract; legacy dense runs remain reproducible."""
import json
from collections import defaultdict
from pathlib import Path


DENSE='hf-dense-v1'
COMPACT='hf-exact-length-v2'


def generation_groups(attention_mask,limit):
    if limit<1 or attention_mask.ndim!=2:
        raise ValueError('Expected positive microbatch limit and 2D attention mask')
    mask=attention_mask.bool()
    lengths=mask.sum(-1)
    if not (lengths>0).all() or (mask[:,:-1]&~mask[:,1:]).any():
        raise ValueError('Generation requires nonempty, contiguous left-padded prompts')
    groups=defaultdict(list)
    for index,length in enumerate(lengths.cpu().tolist()):groups[length].append(index)
    return [rows[start:start+limit] for _,rows in sorted(groups.items())
            for start in range(0,len(rows),limit)]


def forward_contract(config):
    contract=config.get('webshop_phase12',{}).get('forward_contract',DENSE)
    if contract not in (DENSE,COMPACT):raise ValueError('Unknown WebShop forward contract')
    parts=config.get('actor_rollout_ref',{})
    flags=[bool(parts.get(key,{}).get('trim_common_padding',False)) for key in ('actor','ref','rollout')]
    if contract==COMPACT:
        if not all(flags):raise ValueError('Compact contract requires actor/ref/rollout trim together')
        if parts['actor'].get('ppo_micro_batch_size_per_gpu')!=1:
            raise ValueError('Compact native actor scoring requires one row per microbatch')
        if parts['ref'].get('log_prob_micro_batch_size_per_gpu')!=1:
            raise ValueError('Compact reference scoring requires one row per microbatch')
        if parts['rollout'].get('n',1)!=1:
            raise ValueError('Compact GRPO trajectories must be replicated outside generation')
    elif any(flags):raise ValueError('Dense contract cannot silently enable compact forwards')
    return contract


def load_forward_contract(seed_dir):
    manifest=Path(seed_dir)/'phase2/batches/u0001/manifest.json'
    return forward_contract(json.loads(manifest.read_text())['config'])
