import numpy as np
import pandas as pd
import pytest

from phase2.observation_audit import EXTRA_SIGNALS, compare
from phase2.signed_analysis import SIGNALS


def test_direction_and_magnitude_metrics_have_explicit_different_labels():
    frame=pd.DataFrame({name:[3.,2.,1.,0.] for name in dict(SIGNALS,**EXTRA_SIGNALS)})
    frame["global_update"]=[34,34,35,35]
    frame["delta_utility"]=[-.10,-.02,.03,0.]
    result=compare(frame).set_index("signal")
    assert result.loc["D_contribution","ap_any_decline"]==1.
    assert result.loc["D_contribution","ap_decline_gt_5pp"]==1.
    assert result.loc["D_contribution","nonzero_points"]==3
    assert result.loc["D_contribution","units"]==4
    assert result.loc["D_contribution","negative_points"]==2
    assert result.loc["D_contribution","auroc_down_vs_up"]==1.
    # P is not flipped opportunistically to improve test ranking.
    assert result.loc["P_int","risk_orientation"]=="-value"
    assert result.loc["P_int","auroc_down_vs_up"]==0.


def test_constant_score_is_not_given_spurious_direction_correlation():
    frame=pd.DataFrame({name:[1.,1.,1.] for name in dict(SIGNALS,**EXTRA_SIGNALS)})
    frame["global_update"]=[31,32,33];frame["delta_utility"]=[-.1,.1,0.]
    result=compare(frame)
    assert result.rho_risk_decline.isna().all()
    assert result.auroc_down_vs_up.eq(.5).all()


def test_one_direction_only_cannot_estimate_conditional_direction_auroc():
    frame=pd.DataFrame({name:[1.,2.,3.] for name in dict(SIGNALS,**EXTRA_SIGNALS)})
    frame["global_update"]=[31]*3;frame["delta_utility"]=[.1,.2,0.]
    result=compare(frame)
    assert result.auroc_down_vs_up.isna().all()
    assert result.ap_any_decline.isna().all()
