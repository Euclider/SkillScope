from pathlib import Path
import pytest

from webshop_phase12.assets import WebshopBank, make_schedule
from webshop_phase12.prompts import build_prompt, remove_guidance, project_action


ROOT = Path(__file__).resolve().parents[2]


def test_complete_frozen_bank_and_payload_only_control():
    bank = WebshopBank(ROOT / 'memory_data/webshop/claude_style_skills.json')
    assert len(bank.skill_ids) == len(set(bank.skill_ids)) == 54
    skill = bank.get('gen_001')
    prompt = build_prompt('Find a blue shirt under $30', 'Results', ['click[Buy Now]'], [], skill.payload)
    control = remove_guidance(prompt, skill.payload)
    assert skill.payload not in control
    assert 'Find a blue shirt under $30' in control
    assert 'click[Buy Now]' in control
    assert bank.get('gen_001').payload == skill.payload
    assert len(bank.skill_ids) == 54
    with pytest.raises(ValueError):
        remove_guidance(prompt, 'different payload')


def test_schedule_preserves_native_splits_and_unique_train_tasks():
    plan = make_schedule(2500)
    assert plan['eval_ids'] == list(range(500))
    assert plan['dev_ids'] == list(range(500, 1500))
    assert plan['tasks_per_update'] == 128
    for seed in (404, 505):
        ids = plan['seeds'][str(seed)]
        assert len(ids) == len(set(ids)) == 640
        assert set(ids) <= set(range(1500, 2500))
        assert not set(ids) & set(plan['eval_ids'])
    assert make_schedule(2500) == plan


def test_schedule_adapts_to_available_training_inventory():
    plan = make_schedule(1600)
    assert plan['tasks_per_update'] == 20
    assert len(plan['seeds']['404']) == 100
    with pytest.raises(ValueError):
        make_schedule(1505)


def test_action_only_projection_preserves_webshop_command():
    assert project_action('<action>click[Buy Now]</action>') == ('click[buy now]', True)
    assert project_action('<action>search[blue shirt]</action>') == ('search[blue shirt]', True)
    assert project_action('some prose') == ('invalid', False)
    assert project_action('<think>reason</think><action>search[shirt]</action>') == ('invalid', False)


def test_anchor_replay_restores_same_visible_and_mutable_state():
    import random
    from webshop_phase12.envs import ShopWorld
    class FakeEnv:
        session_prefix = 'fake_'
        def reset(self,session):
            self.state=[]
            return 'start',None
        def get_available_actions(self):
            return {'has_search_bar':True,'clickables':[]}
        def step(self,action):
            self.state.append(action)
            return '|'.join(self.state),0.,False,None
    world=ShopWorld.__new__(ShopWorld)
    world.envs=[FakeEnv()]
    world.server=type('Server',(),{'goals':[{} for _ in range(10)],'user_sessions':{}})()
    world.prefixes=[[]];world.task_ids=[None];world.done=[False];world.last_obs=['']
    world.random_states=[random.Random(0).getstate()]
    first,_=world.replay(0,3,['search[shirt]','click[item]'])
    state=list(world.envs[0].state)
    world.step_one(0,'click[red]')
    second,_=world.replay(0,3,['search[shirt]','click[item]'])
    assert first==second
    assert world.envs[0].state==state
    assert world.prefixes[0]==['search[shirt]','click[item]']
