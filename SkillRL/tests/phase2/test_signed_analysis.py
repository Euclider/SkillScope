import pandas as pd
import pytest

from phase2.signed_analysis import direction_counts,summarize


def frame():
    return pd.DataFrame({"global_update":[31,32,33],"supported":[True]*3,
        "delta_utility":[.1,-.2,0.],"ci_low":[0.,-.3,0.],"ci_high":[.2,-.1,0.],
        "P_int":[.2,.1,-.1],"D_contribution":[.1,.2,.3],"delta_norm":[1.,2.,3.]})


def test_sign_accuracy_is_not_positive_rank_correlation():
    q=frame();out=direction_counts(q)
    assert out["nonzero_points"]==2
    assert out["zero_points"]==1
    assert out["conditional_sign_accuracy"]==.5
    assert out["always_positive_conditional_accuracy"]==.5
    assert out["nonzero_point_coverage"]==pytest.approx(2/3)
    assert out["ci_excludes_zero_points"]==1
    assert out["ci_excludes_zero_sign_accuracy"]==0.


def test_zero_predictions_abstain_not_successful_zero_direction_labels():
    q=frame();q["zero"]=0.
    out=direction_counts(q,"zero")
    assert out["units"]==3
    assert out["prediction_coverage"]==0.
    assert out["conditional_sign_accuracy"] is None


def test_heldout_and_unsupported_not_mixed():
    q=frame();q.loc[1,"supported"]=False
    out=summarize(q,[33])
    assert out["direction_counts"][0]["units"]==2
    assert out["direction_counts"][1]["units"]==1
    assert out["direction_counts"][1]["conditional_sign_accuracy"] is None
    assert all(row["units"]==1 for row in out["associations"] if row["scope"]=="heldout_updates")
