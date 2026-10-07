"""Explicit new-cohort input contract; legacy dense runs remain reproducible."""
import json
from collections import defaultdict
from pathlib import Path


DENSE='hf-dense-v1'
COMPACT='hf-exact-length-v2'
BUCKET='hf-bucket256-v3'
ACTIVE='hf-bucket256-active-v4'


def generation_groups(attention_mask,limit,*,prompt_multiple=1):
    if limit<1 or attention_mask.ndim!=2:
        raise ValueError('Expected positive microbatch limit and 2D attention mask')
    mask=attention_mask.bool()
    lengths=mask.sum(-1)
    if not (lengths>0).all() or (mask[:,:-1]&~mask[:,1:]).any():
        raise ValueError('Generation requires nonempty, contiguous left-padded prompts')
    groups=defaultdict(list)
    from verl.workers.actor.padded_forward import prompt_bucket_width
    for index,length in enumerate(lengths.cpu().tolist()):
        groups[prompt_bucket_width(length,prompt_multiple)].append(index)
    return [rows[start:start+limit] for _,rows in sorted(groups.items())
            for start in range(0,len(rows),limit)]


def forward_contract(config):
    contract=config.get('webshop_phase12',{}).get('forward_contract',DENSE)
    if contract not in (DENSE,COMPACT,BUCKET,ACTIVE):raise ValueError('Unknown WebShop forward contract')
    parts=config.get('actor_rollout_ref',{})
    flags=[bool(parts.get(key,{}).get('trim_common_padding',False)) for key in ('actor','ref','rollout')]
    if contract!=DENSE:
        if not all(flags):raise ValueError('Compact contract requires actor/ref/rollout trim together')
        if parts['actor'].get('ppo_micro_batch_size_per_gpu')!=1:
            raise ValueError('Compact native actor scoring requires one row per microbatch')
        if parts['ref'].get('log_prob_micro_batch_size_per_gpu')!=1:
            raise ValueError('Compact reference scoring requires one row per microbatch')
        if parts['rollout'].get('n',1)!=1:
            raise ValueError('Compact GRPO trajectories must be replicated outside generation')
        if parts['rollout'].get('log_prob_micro_batch_size_per_gpu',1)!=1:
            raise ValueError('Compact native old-logprob scoring requires one row per microbatch')
        if contract==ACTIVE and not all(parts[key].get('active_response_logits_only',False) for key in ('actor','ref')):
            raise ValueError('Active contract requires actor/reference active-position projection together')
        multiple=256 if contract in (BUCKET,ACTIVE) else 1
        if any(parts[key].get('prompt_padding_multiple',1)!=multiple for key in ('actor','ref','rollout')):
            raise ValueError('Actor/reference/generation prompt bucket sizes differ')
    elif any(flags):raise ValueError('Dense contract cannot silently enable compact forwards')
    return contract


def prompt_multiple(contract):
    return 256 if contract in (BUCKET,ACTIVE) else 1


def load_forward_contract(seed_dir):
    manifest=Path(seed_dir)/'phase2/batches/u0001/manifest.json'
    return forward_contract(json.loads(manifest.read_text())['config'])
