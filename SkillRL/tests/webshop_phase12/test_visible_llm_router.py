import json
import pytest

from webshop_phase12.assets import WebshopBank, BASE_MODEL, digest


def trace():
    return [
        ('search[shirt]', 'Results', ['click[B000000001]'], {'page_type':'search_results'}),
        ('click[B000000001]', 'Blue shirt\nPrice: $15.00\n[button] Description [button_]', ['click[Description]'], {'page_type':'item_page','product_id':'B000000001'}),
        ('click[Description]', 'Cotton, machine washable.', ['click[< Prev]'], {'page_type':'item_sub_page','product_id':'B000000001','detail_kind':'description'}),
        ('click[< Prev]', 'Blue shirt\nPrice: $15.00', ['click[Back to Search]'], {'page_type':'item_page','product_id':'B000000001'}),
        ('click[Back to Search]', 'Results', ['click[B000000001]'], {'page_type':'search_results'}),
        ('click[B000000001]', 'You have clicked blue.\nBlue shirt\nPrice: $15.00', ['click[Buy Now]'], {'page_type':'item_page','product_id':'B000000001'}),
    ]


def test_visible_memory_retains_old_details_and_replays_without_goal_attributes():
    from webshop_phase12.visible_state import VisibleMemory
    def replay():
        memory=VisibleMemory('Find a blue shirt',50)
        memory.observe('Search',['search[<your query>]'],{'page_type':'index'})
        for action,obs,actions,page in trace():
            memory.transition(action,True,obs,actions,{**page,'reward':1.,'gold_attributes':['secret']})
        return memory
    memory=replay(); state=memory.state()
    assert len(state['action_history'])==6
    assert len(state['recent_feedback'])==2
    evidence=state['product_evidence']['B000000001']
    assert evidence['details']['description'][0]['text']=='Cotton, machine washable.'
    assert evidence['selected_options'][-1]['text']=='blue'
    assert digest(state)==digest(replay().state())
    assert 'secret' not in json.dumps(state) and 'reward' not in state
    assert VisibleMemory('New goal',50).state()['product_evidence']=={}


def test_click_is_not_evidence_of_successful_navigation_or_variant_selection():
    from webshop_phase12.visible_state import VisibleMemory
    memory=VisibleMemory('Buy a shirt',50)
    memory.observe('Results',['click[B000000001]'],{'page_type':'search_results'})
    memory.transition('click[B000000001]',False,'Results',['click[B000000001]'],{'page_type':'search_results'})
    assert memory.state()['product_evidence']=={}
    memory.transition('click[blue]',False,'Shirt\nPrice: $10.00',[],{'page_type':'item_page','product_id':'B000000001'})
    assert memory.state()['product_evidence']['B000000001']['selected_options']==[]


def test_router_prompt_and_control_share_all_visible_facts():
    from webshop_phase12.visible_state import VisibleMemory
    from webshop_phase12.llm_router import build_router_prompt, choice_labels
    from webshop_phase12.prompts import build_state_prompt, remove_guidance
    memory=VisibleMemory('Buy cotton shirt',50)
    memory.observe('Cotton shirt\nPrice: $10.00',['click[Buy Now]'],{'page_type':'item_page','product_id':'B000000001'})
    state=memory.state();bank=WebshopBank()
    prompt=build_router_prompt(bank,state)
    assert len(choice_labels())==54
    assert all(sid in prompt for sid in bank.skill_ids)
    assert all(bank.get(sid).payload in prompt for sid in bank.skill_ids)
    payload=bank.get('gen_001').payload
    execution=build_state_prompt(state,payload)
    control=remove_guidance(execution,payload)
    assert 'Price: $10.00' in control and 'click[Buy Now]' in control
    assert state['task_description'] in prompt and state['task_description'] in control


def test_llm_router_deduplicates_batch_and_reuses_frozen_decisions(tmp_path):
    from webshop_phase12.llm_router import FrozenLLMSkillRouter, choice_labels
    from webshop_phase12.visible_state import VisibleMemory
    class Backend:
        identity={'model':'test-only-model','weights':'fixed'}
        def __init__(self):self.calls=[]
        def select_many(self,prompts):
            self.calls.append(prompts)
            return [{'label':'A','input_tokens':123,'output_tokens':1,'latency_ms':2} for _ in prompts]
    backend=Backend(); router=FrozenLLMSkillRouter(tmp_path/'router.sqlite3',backend)
    state=VisibleMemory('Buy a shirt',50).state()
    bundles=router.route_many([state,state])
    assert len(backend.calls)==1 and len(backend.calls[0])==1
    assert bundles[0]['selected_skill_id']=='gen_001'
    assert bundles[0]['candidate_skill_ids']==list(WebshopBank().skill_ids)
    assert bundles[0]['skill_router_api']['cache_hit'] is False
    assert bundles[1]['skill_router_api']['cache_hit'] is True
    assert router.route_many([state])[0]['skill_router_api']['cache_hit'] is True
    assert len(backend.calls)==1
    # Full catalog is a fixed prefix; only the factual state suffix changes.
    state2={**state,'task_description':'Buy shoes'}
    router.route_many([state2])
    assert backend.calls[0][0].split('SHOPPING GOAL:')[0]==backend.calls[1][0].split('SHOPPING GOAL:')[0]


def test_router_rejects_unlisted_choice_and_does_not_retry_failed_input(tmp_path):
    from webshop_phase12.llm_router import FrozenLLMSkillRouter
    from webshop_phase12.visible_state import VisibleMemory
    class Backend:
        identity={'model':'bad-test-model'}
        def __init__(self):self.calls=0
        def select_many(self,prompts):
            self.calls+=1
            return [{'label':'invented','input_tokens':2,'output_tokens':1,'latency_ms':1}]
    backend=Backend();router=FrozenLLMSkillRouter(tmp_path/'bad.sqlite3',backend)
    state=VisibleMemory('Buy a shirt',50).state()
    with pytest.raises(ValueError):router.route_many([state])
    with pytest.raises(Exception,match='Prior failed/incomplete'):router.route_many([state])
    assert backend.calls==1


def test_pinned_tokenizer_supports_exactly_54_distinct_single_token_labels():
    from transformers import AutoTokenizer
    from webshop_phase12.llm_router import choice_labels
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,local_files_only=True)
    tokens=[tokenizer.encode(label,add_special_tokens=False) for label in choice_labels()]
    assert all(len(ids)==1 for ids in tokens)
    assert len({ids[0] for ids in tokens})==54


def test_full_visible_packet_fits_with_previous_details_and_oversized_menu():
    from pathlib import Path
    from transformers import AutoTokenizer
    from webshop_phase12.visible_state import VisibleMemory
    from webshop_phase12.llm_router import build_router_prompt
    from webshop_phase12.prompts import build_state_prompt
    fixture=json.loads((Path(__file__).parent/'fixtures/oversized_menu_state.json').read_text())
    memory=VisibleMemory(fixture['task_description'],50)
    memory.observe('Search',[],{'page_type':'index'})
    memory.transition('click[Description]',True,'Previously observed factual detail.',[],
        {'page_type':'item_sub_page','product_id':'B000000001','detail_kind':'description'})
    memory.transition('click[item]',True,fixture['current_observation'],fixture['admissible_actions'],
        {'page_type':'item_page','product_id':'B000000001'})
    state=memory.state();tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,local_files_only=True);bank=WebshopBank()
    for prompt,limit in ((build_state_prompt(state,bank.get('gen_001').payload),16384),(build_router_prompt(bank,state),32768)):
        chat=tokenizer.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
        assert len(tokenizer.encode(chat,add_special_tokens=False))<=limit
        assert 'Previously observed factual detail.' in prompt
        assert all(action in prompt for action in fixture['admissible_actions'])


def test_bulk_cache_publish_is_atomic_and_integrity_checked(tmp_path):
    from agent_system.memory.router_cache import RouterCache,RouterCacheError
    cache=RouterCache(tmp_path/'bulk.sqlite3',{'kind':'test'},max_api_calls=4)
    keys=[digest('first'),digest('second')]
    attempts=cache.reserve_many_local([(key,{'state':i}) for i,key in enumerate(keys)])
    records=[{'cache_key':key,'protocol_hash':cache.protocol_hash,'choice':i} for i,key in enumerate(keys)]
    with pytest.raises(RouterCacheError):
        cache.finish_many([(attempts[keys[0]],records[0]),(99999,records[1])])
    assert cache.lookup(keys[0]) is None
    cache.finish_many([(attempts[key],record) for key,record in zip(keys,records)])
    assert cache.lookup_many(keys)==dict(zip(keys,records))
    cache.note_hits(keys+keys)
    assert cache.stats()['cache_hits']==4


def test_gpu_layout_uses_any_idle_cards_and_never_claims_an_occupied_one():
    from webshop_phase12.coordinate_llm import choose_layout
    rows={i:{'memory_mib':30000,'utilization':50} for i in range(8)}
    assert choose_layout(rows) is None
    for i in (1,6):rows[i]={'memory_mib':16,'utilization':0}
    layout=choose_layout(rows)
    assert layout['train_gpus']==[1,6] and layout['router_gpu']==6 and layout['shared_router']
    rows[4]={'memory_mib':16,'utilization':0}
    layout=choose_layout(rows)
    assert layout['train_gpus']==[1,4] and layout['router_gpu']==6 and not layout['shared_router']
    rows[2]={'memory_mib':16,'utilization':0}
    rows[5]={'memory_mib':16,'utilization':0}
    layout=choose_layout(rows)
    assert layout['train_gpus']==[1,2,4,5] and layout['router_gpu']==6
    for i in range(8):rows[i]={'memory_mib':16,'utilization':0}
    layout=choose_layout(rows)
    assert layout['train_gpus']==[0,1,2,3] and layout['router_gpus']==[4,5,6,7]


class CPUSelectionBackend:
    identity={'model':'test-only-routing-IPC'}
    def __init__(self):self.sleeping=False
    def sleep(self):self.sleeping=True
    def wake(self):self.sleeping=False
    def select_many(self,prompts):
        if self.sleeping:raise RuntimeError('Inference while sleeping')
        return [{'label':prompt[0],'input_tokens':len(prompt),'output_tokens':1,'latency_ms':0} for prompt in prompts]


def test_router_replica_pool_restores_order_after_parallel_dispatch():
    from webshop_phase12.llm_service import ReplicaPool
    # CPU backend replaces model loading only; real child processes and IPC
    # exercise distribution, identity validation, result order and cleanup.
    backend=ReplicaPool([0,1],factory=CPUSelectionBackend)
    try:
        prompts=['A short','B'+' long'*30,'C middle'*5,'D tiny']
        assert [row['label'] for row in backend.select_many(prompts)]==['A','B','C','D']
        backend.sleep();backend.wake()
        assert [row['label'] for row in backend.select_many(prompts)]==['A','B','C','D']
    finally:backend.close()
