import random
from types import SimpleNamespace


def world(session, product='B0001', option='blue'):
    from webshop_phase12.envs import ShopWorld
    instance = ShopWorld.__new__(ShopWorld)
    instance.envs = [SimpleNamespace(session=session, browser=SimpleNamespace(
        current_url=f'http://127.0.0.1:3000/item_page/{session}/{product}/query/1'))]
    instance.server = SimpleNamespace(user_sessions={session: {'done': False, 'options': {'color': option}}})
    instance.random_states = [random.Random(1500).getstate()]
    return instance


def test_transport_session_id_does_not_change_semantic_environment_digest():
    first = world('p12_100_0_1500')
    second = world('p12_200_0_1500')
    assert first.state_digest(0) == second.state_digest(0)
    assert first.state_digest(0) != world('p12_200_0_1500', product='B0002').state_digest(0)
    assert first.state_digest(0) != world('p12_200_0_1500', option='red').state_digest(0)
    second.random_states = [random.Random(1501).getstate()]
    assert first.state_digest(0) != second.state_digest(0)
