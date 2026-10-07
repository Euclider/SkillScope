import pytest


def test_router_cost_uses_actual_complete_update_time_and_rejects_high_overhead():
    from webshop_phase12.router_cost import update_router_cost
    log = 'timing_s/update_actor:1200.0 - timing_s/step:1600.0'
    cost = update_router_cost({'router_seconds': 56., 'step_calls': 8}, log)
    assert cost['fraction'] == .035 and cost['metric'] == 'router_share_of_complete_RL_update'
    assert cost['threshold'] == .10 and cost['passed']
    with pytest.raises(ValueError):
        update_router_cost({'router_seconds': 200., 'step_calls': 8}, log)
    with pytest.raises(ValueError):
        update_router_cost({'router_seconds': 10., 'step_calls': 8}, 'no complete update timing')
