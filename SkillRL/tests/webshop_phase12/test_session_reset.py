import random
import sys

from webshop_phase12.envs import SHOP, ShopWorld


def test_reset_discards_purchase_state_only_for_current_session(monkeypatch):
    monkeypatch.syspath_prepend(str(SHOP))
    from web_agent_site.envs.web_agent_text_env import SimServer, WebAgentTextEnv

    # Use the real simulator/browser with a tiny goal inventory, without
    # loading the product catalog. Purchase metadata is the observed fixture.
    server = SimServer.__new__(SimServer)
    server.base_url = 'http://127.0.0.1:3000'
    server.goals = [{'instruction_text': 'Find a blue shirt', 'weight': 1.}]
    server.cum_weights = [0., 1.]
    server.user_sessions = {}
    server.assigned_instruction_text = None
    env = WebAgentTextEnv(observation_mode='text', server=server,
                          seed=0, session_prefix='reset_test_')
    world = ShopWorld.__new__(ShopWorld)
    world.server, world.envs = server, [env]
    world.prefixes, world.task_ids = [[]], [None]
    world.done, world.last_obs = [False], ['']
    world.random_states = [random.Random(0).getstate()]
    observation, _ = world.reset_one(0, 0)
    original_digest = world.state_digest(0)
    server.user_sessions[env.session].update(
        done=True, reward=1., verbose_info={'purchase': 'previous arm'})
    server.user_sessions['other_session'] = {'done': True, 'reward': .5}
    restored, _ = world.replay(0, 0, [])

    assert restored == observation
    assert world.state_digest(0) == original_digest
    assert server.user_sessions[env.session]['done'] is False
    assert 'reward' not in server.user_sessions[env.session]
    assert server.user_sessions['other_session'] == {'done': True, 'reward': .5}
