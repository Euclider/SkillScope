import pandas as pd
import pytest

from skillnet_cohort.seed_queue import can_start, next_estimate
from skillnet_cohort.reports import ranking_diagnostics
from skillnet_cohort.runtime import disk_gate
from skillnet_cohort.common import write_new_json


def test_seeds_share_budget_and_recalibration_uses_walltime_only():
    assert next_estimate(50000, []) == 50000
    assert next_estimate(50000, [10000]) == 12500
    assert next_estimate(50000, [10000, 20000]) == 25000
    assert can_start(108000, 25000, 10800, now=70000)
    assert not can_start(108000, 25000, 10800, now=80000)


def test_shared_disk_budget_includes_prior_seeds(tmp_path):
    write_new_json(tmp_path / 'queue_launch.json', {'seeds': [404, 505, 606]})
    write_new_json(tmp_path / 'seed-404/evidence.json', {'keep': 'x' * 10000})
    root = tmp_path / 'seed-505'
    write_new_json(root / 'resource_limits.json', {'cohort_storage_root': str(tmp_path)})
    with pytest.raises(OSError):
        disk_gate(root, 0, minimum_free_bytes=1, maximum_run_bytes=5000)
    assert (tmp_path / 'seed-404/evidence.json').exists()


def test_ap_uses_locked_oriented_scores_and_same_supported_pool():
    scores = pd.DataFrame({'context_id': ['all'] * 3, 'phase': ['all'] * 3, 'skill_id': ['a', 'b', 'c'],
        'delta_utility': [-.2, .1, -.4], 'D_contribution': [2, 1, 100], 'random_expected': [0, 0, 0]})
    support = {'candidate_pools': [{'context_id': 'all', 'phase': 'all', 'shared_skill_ids': ['a', 'b']}]}
    result = ranking_diagnostics(scores, support, {'scores': {'D_contribution': 1}, 'event_thresholds': [0, .5]})
    d = result[(result.score == 'D_contribution') & (result.threshold == 0)].iloc[0]
    assert d.candidates == 2 and d.declines == 1 and d.average_precision == 1
    assert result[result.threshold == .5].average_precision.isna().all()
