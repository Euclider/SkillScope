import pytest

from skillnet_cohort.common import read_json
from skillnet_cohort.seed505_precision import (
    COHORT, GOLD, NEW_GOLD, OUTPUT, SOURCE, expanded_config, jobs_with_ids,
    scope, validate_config,
)


def test_seed505_gold_bases_are_disjoint_and_keep_old_repeats():
    assert GOLD[:2] == (63011, 63021)
    assert len(GOLD) == 8
    assert NEW_GOLD == (505, 505100, 505200, 505300, 505400, 505500)
    for i, seed in enumerate(NEW_GOLD):
        assert all(abs(seed - other) >= 50 for other in GOLD if other != seed)


def test_seed505_config_changes_only_gold_and_owned_paths():
    old = read_json(SOURCE/'protocol.json')
    expanded = expanded_config(old, OUTPUT)
    validate_config(old, expanded, OUTPUT)
    assert expanded['runtime']['max_api_calls'] == 0
    assert expanded['evaluation']['gold_seeds'] == list(GOLD)
    assert len(jobs_with_ids(expanded, 0)) == 10908
    assert len(jobs_with_ids(expanded, 5)) == 10908
    changed = read_json(SOURCE/'protocol.json')
    changed['evaluation']['max_steps'] = 49
    with pytest.raises(ValueError):
        validate_config(old, changed, OUTPUT)


def test_seed505_precision_rejects_other_outputs():
    assert scope(COHORT/'utility-precision-s505-v19') == COHORT/'utility-precision-s505-v19'
    with pytest.raises(PermissionError):
        scope(COHORT/'utility-precision-s404-v1')
