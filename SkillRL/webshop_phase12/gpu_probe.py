"""Real frozen-model batch/cache parity and throughput gate before RL."""
import argparse
import json
import time
import atexit
from pathlib import Path

from webshop_phase12.assets import WebshopBank,ROOT
from webshop_phase12.llm_router import build_router_prompt,FrozenLLMSkillRouter
from webshop_phase12.llm_service import load_backend
from webshop_phase12.visible_state import VisibleMemory


def main(out):
    out.mkdir(exist_ok=False)
    bank=WebshopBank();backend=load_backend()
    atexit.register(backend.close)
    first=VisibleMemory('Find a blue cotton shirt under $30',50)
    first.observe('Search',['search[<your query>]'],{'page_type':'index'})
    second=VisibleMemory('Find a blue cotton shirt under $30',50)
    second.observe('Blue shirt\nPrice: $15.00',['click[Description]','click[blue]','click[Buy Now]'],{'page_type':'item_page','product_id':'B000000001'})
    fixture=json.loads((ROOT/'tests/webshop_phase12/fixtures/oversized_menu_state.json').read_text())
    third=VisibleMemory(fixture['task_description'],50)
    third.observe(fixture['current_observation'],fixture['admissible_actions'],{'page_type':'item_page','product_id':'B000000002'})
    states=[first.state(),second.state(),third.state()]
    prompts=[build_router_prompt(bank,state) for state in states]
    batched=backend.select_many(prompts)
    singles=[backend.select_many([prompt])[0] for prompt in prompts]
    reversed_batch=backend.select_many(list(reversed(prompts)))
    labels=[row['label'] for row in batched]
    if labels!=[row['label'] for row in singles] or labels!=[row['label'] for row in reversed(reversed_batch)]:
        raise ValueError('Frozen router choices changed with batch grouping/order')
    if not any(row['prefix_cached_tokens']>0 for row in batched):
        raise ValueError('Fixed catalogue prefix was not reused; speed gate failed')
    backend.sleep();backend.wake()
    after_wake=backend.select_many(prompts)
    if labels!=[row['label'] for row in after_wake]:
        raise ValueError('Frozen router choice changed after sleep/wake')
    # Exercise persistent state deduplication separately from prefix caching.
    router=FrozenLLMSkillRouter(out/'router.sqlite3',backend)
    started=time.monotonic();warm=router.route_many(states*8);first_seconds=time.monotonic()-started
    started=time.monotonic();hot=router.route_many(states*8);hot_seconds=time.monotonic()-started
    if router.cache.stats()['successful_decisions']!=3 or not all(row['skill_router_api']['cache_hit'] for row in hot):
        raise ValueError('Frozen-router exact-state cache contract failed')
    receipt={'status':'passed','batched':batched,'single':singles,'batch_order_choices_equal':True,
        'catalogue_prefix_cache_reused':True,'sleep_wake_choices_equal':True,
        'deduplicated_states':3,'requested_states':24,
        'first_request_seconds':first_seconds,'hot_request_seconds':hot_seconds,
        'backend':backend.identity,'execution':getattr(backend,'execution',{'parallel_replicas':1}),
        'protocol':router.protocol,'cache':router.cache.stats()}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    if hasattr(backend,'close'):backend.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();main(args.out)
