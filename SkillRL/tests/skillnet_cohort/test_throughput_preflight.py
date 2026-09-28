from types import SimpleNamespace as NS

import pytest

from skillnet_cohort.common import read_json
from skillnet_cohort.throughput_preflight import TimedPolicy, select_games


def test_probe_selection_is_balanced_and_input_order_independent():
    games = [{'game_id': f'{task}-{i}', 'task_type': task}
             for task in ('a', 'b', 'c') for i in range(5)]
    picked = select_games(games)
    assert picked == select_games(list(reversed(games)))
    assert [g['task_type'] for g in picked] == ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b']
    assert len({g['game_id'] for g in picked}) == 8


def test_probe_cannot_silently_reduce_count():
    with pytest.raises(ValueError, match='Insufficient'):
        select_games([{'game_id': 'a', 'task_type': 'a'}])


def test_timing_wrapper_preserves_request_and_response(tmp_path):
    seen = []
    def generate(*args, **kwargs):
        seen.append((args, kwargs))
        return 'open fridge', 100, 3
    policy = TimedPolicy(NS(tokenizer='tokenizer', generate=generate), tmp_path)
    assert policy.generate('prompt', seed=714040) == ('open fridge', 100, 3)
    assert seen == [(('prompt',), {'seed': 714040})]
    assert read_json(tmp_path / '0000.json')['completion_tokens'] == 3
    assert len(policy.calls) == 1 and policy.tokenizer == 'tokenizer'
