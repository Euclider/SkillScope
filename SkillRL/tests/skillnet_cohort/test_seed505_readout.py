from pathlib import Path

import pytest

from skillnet_cohort.seed505_readout import COHORT, gpu_pair, scope


def test_seed505_gpu_pairs_cover_exactly_eight_cards():
    assert [gpu_pair(i) for i in range(8)] == [
        '0,1', '2,3', '4,5', '6,7',
        '0,1', '2,3', '4,5', '6,7',
    ]
    with pytest.raises(ValueError):
        gpu_pair(8)


def test_readout_requires_new_scoped_seed505_output():
    assert scope(COHORT/'realized-and-scores-s505-v19') == (
        COHORT/'realized-and-scores-s505-v19')
    with pytest.raises(PermissionError):
        scope(COHORT/'realized-reward-s404-v2')
    with pytest.raises(PermissionError):
        scope(Path('/tmp/realized-and-scores-s505-v1'))
