import json

import pytest

from agent_system.memory.router_cost_guard import CostGuard
from agent_system.memory.router_cache import RouterBudgetExceeded, RouterCacheError


def guard(tmp_path, cap=.1):
    path = tmp_path / 'money.json'
    path.write_text(json.dumps({'schema_version':'skillrl.router.cost_cap.v1','cap_rmb':cap,
        'rmb_per_usd_accounting':8, 'input_usd_per_million':.75, 'output_usd_per_million':4.5}))
    return CostGuard(path)


def test_inflight_prevents_overspend_and_unknown_usage_stays_reserved(tmp_path):
    money = guard(tmp_path)
    money.reserve('one', {'hello':'world'}, 128)
    with pytest.raises(RouterBudgetExceeded):
        money.reserve('two', {}, 128)
    money.settle('one', {'prompt_tokens':None, 'completion_tokens':None})
    assert money.stats()['unreconciled_calls'] == 1
    with pytest.raises(RouterBudgetExceeded):
        money.reserve('two', {}, 128)


def test_settlement_releases_unused_reservation_without_discounts(tmp_path):
    money = guard(tmp_path)
    money.reserve('one', {}, 128)
    money.settle('one', {'prompt_tokens':100, 'completion_tokens':10})
    assert money.stats()['usd_charged_or_reserved'] == pytest.approx(.00012)
    money.reserve('two', {}, 128)
    with pytest.raises(RouterCacheError):
        money.settle('one', {'prompt_tokens':0, 'completion_tokens':0})


def test_no_implicit_cap_extension(tmp_path):
    guard(tmp_path)
    with pytest.raises(RouterCacheError, match='Changed'):
        guard(tmp_path, cap=100)
