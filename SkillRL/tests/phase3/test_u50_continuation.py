from pathlib import Path

import pytest

from phase3.common import ProtocolError


def authority(tmp_path):
    return {'schema': 'phase3.skillrl_u50.v1', 'run_root': str(tmp_path),
            'branch': 'skillrl_failure', 'start_update': 20, 'stop_update': 50,
            'preparation': str(tmp_path / 'assets/manifest.json')}


def test_extension_only_allows_skillrl_registered_windows(tmp_path):
    from scripts.continue_skillrl_u50 import validate_extension
    record = authority(tmp_path)
    run = tmp_path / 'runs/skillrl_failure'
    for start in (20, 25, 30, 35, 40, 45):
        validate_extension(record, run, 'skillrl_failure', start, record['preparation'])
    for start in (0, 15, 21, 50, 150):
        with pytest.raises(ProtocolError):
            validate_extension(record, run, 'skillrl_failure', start, record['preparation'])
    with pytest.raises(ProtocolError):
        validate_extension(record, run, 'readout_d', 20, record['preparation'])
    with pytest.raises(ProtocolError):
        validate_extension(record, tmp_path / 'foreign', 'skillrl_failure', 20, record['preparation'])


def test_training_route_preserves_every_argument_and_leaves_scoring_untouched(tmp_path):
    from scripts.continue_skillrl_u50 import training_route
    record = authority(tmp_path)
    args = ['phase3.training', '--preparation', record['preparation'], '--root',
            str(tmp_path / 'runs/skillrl_failure'), '--branch', 'skillrl_failure',
            '--bank-path', str(tmp_path / 'bank.json'), '--bank-sha256', 'a'*64,
            '--start', '20', '--execute']
    routed = training_route(args, record)
    assert routed[routed.index('--train') + 1:] == args[1:]
    assert training_route(['phase3.predict', '--batch', 'same.pt'], record) is None
    args[args.index('--start') + 1] = '50'
    with pytest.raises(ProtocolError):
        training_route(args, record)
