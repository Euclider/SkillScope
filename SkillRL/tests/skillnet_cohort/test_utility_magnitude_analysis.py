import numpy as np
import pandas as pd
import pytest

from skillnet_cohort.utility_magnitude_analysis import (
    check_direction_reproduction, magnitude_arrays,
)


def test_magnitude_target_is_absolute_utility_change_not_signed_decline():
    scores = np.array([[0., 1., 2., 3.]])
    risk = np.array([[-.10, 0., .04, .20]])
    result = magnitude_arrays(scores, risk)
    assert result['top_quartile_mean_absolute_delta'][0, 0] == pytest.approx(.20)
    assert result['top_quartile_lift'][0, 0] == pytest.approx(.20 / .085)
    # Both improvement (-risk) and decline (+risk) can be large changes.
    opposite = magnitude_arrays(scores, -risk)
    for key in result:
        np.testing.assert_allclose(result[key], opposite[key], equal_nan=True)


def test_constant_magnitude_label_is_undefined_not_perfect():
    result = magnitude_arrays(np.array([[1., 2., 3.]]), np.zeros((1, 3)))
    assert np.isnan(result['spearman'][0, 0])
    assert np.isnan(result['average_precision'][0, 0])
    assert np.isnan(result['auroc_large_change_vs_rest'][0, 0])
    assert np.isnan(result['top_quartile_lift'][0, 0])


def test_direction_reproduction_rejects_changed_value():
    row = {'control': 'placebo', 'context_id': 'all_alfworld', 'phase': 'all',
           'threshold': 0., 'score': 'D', 'average_precision': .5,
           'auroc_decline_vs_rest': .5, 'auroc_decline_vs_increase': .5,
           'spearman': 0., 'kendall': 0.}
    prior = pd.DataFrame([row])
    assert check_direction_reproduction(prior.copy(), prior) == 1
    changed = prior.copy()
    changed.loc[0, 'spearman'] = .1
    with pytest.raises(ValueError, match='did not reproduce'):
        check_direction_reproduction(changed, prior)
