"""Descriptive signed-utility audit; never selects features or changes forecasts."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SIGNALS = {
    "P_int": "signed reward-directed interaction",
    "D_contribution": "gated opposition (nonnegative; decline risk)",
    "D_ungated_contribution": "ungated opposition sensitivity",
    "C_upd": "update fidelity, not a utility sign",
    "u_original_norm": "unsigned skill-conditioned shift",
    "u_control_norm": "unsigned control shift",
    "delta_norm": "unsigned interaction shift",
    "delta_centered_norm": "centered interaction magnitude",
    "forward_kl_original": "unsigned distribution shift baseline",
    "js_original": "unsigned distribution shift baseline",
    **{f"activation_l{layer}_norm": "unsigned activation interaction" for layer in (8, 16, 24, 32)},
    "raw_parameter_delta_l2": "update-level magnitude (shared by all skills)",
    "old_margin": "independent pre-update evidence baseline",
    "old_margin_se": "pre-update evidence uncertainty",
    "train_success": "existing training rollout baseline",
    "advantage": "existing training advantage baseline",
}


def correlation(x, y):
    q=pd.DataFrame({"x":x,"y":y}).replace([np.inf,-np.inf],np.nan).dropna()
    if len(q)<3 or q.x.nunique()<2 or q.y.nunique()<2:return None
    return float(spearmanr(q.x,q.y).statistic)


def direction_counts(frame, prediction="P_int"):
    """Nonzero point-estimate accuracy is conditional, with coverage explicit.

    Zero predictions abstain; zero observed changes remain in continuous MAE.
    CI-excluding-zero coverage is a diagnostic, not a replacement for the
    existing +/-5pp primary direction labels.
    """
    q=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=["delta_utility",prediction])
    y=q.delta_utility.to_numpy();p=q[prediction].to_numpy()
    nonzero=np.abs(y)>1e-12
    predicted=np.abs(p)>1e-12
    evaluated=nonzero & predicted
    agree=np.sign(p)==np.sign(y)
    reliable=((q.ci_low>0)|(q.ci_high<0)).to_numpy()
    count=int(evaluated.sum())
    return {"units":len(q),"updates":int(q.global_update.nunique()),
            "positive_points":int((y>1e-12).sum()),"negative_points":int((y < -1e-12).sum()),
            "zero_points":int((~nonzero).sum()),"nonzero_points":int(nonzero.sum()),
            "nonzero_point_coverage":float(nonzero.mean()) if len(q) else None,
            "prediction_coverage":float(predicted.mean()) if len(q) else None,
            "evaluated_nonzero_points":count,"sign_agree":int((agree&evaluated).sum()),
            "conditional_sign_accuracy":float(agree[evaluated].mean()) if count else None,
            "always_positive_conditional_accuracy":float((y[nonzero]>0).mean()) if nonzero.any() else None,
            "ci_excludes_zero_points":int(reliable.sum()),
            "ci_excludes_zero_sign_accuracy":float(agree[reliable&predicted].mean()) if (reliable&predicted).any() else None}


def summarize(frame, test_updates):
    association=[];directions=[];by_update=[]
    for scope,q in (("all_updates_exploratory",frame),
                    ("heldout_updates",frame[frame.global_update.isin(test_updates)])):
        q=q[q.supported]
        if q.empty:continue
        directions.append({"scope":scope,**direction_counts(q)})
        for name,role in SIGNALS.items():
            if name not in q:continue
            good=q.dropna(subset=[name,"delta_utility"])
            association.append({"scope":scope,"signal":name,"role":role,"units":len(good),
                "updates":int(good.global_update.nunique()),
                "rho_signed_delta":correlation(good[name],good.delta_utility),
                "rho_absolute_delta":correlation(good[name],good.delta_utility.abs())})
    for update,q in frame[frame.supported].groupby("global_update"):
        by_update.append({"update":int(update),"P_vs_signed_delta":correlation(q.P_int,q.delta_utility),
                          "D_vs_decline":correlation(q.D_contribution,-q.delta_utility),**direction_counts(q)})
    return {"associations":association,"direction_counts":directions,"by_update":by_update,
            "interpretation":"descriptive only; shared update/skill and adjacent-endpoint dependence; no feature selection or independence claim"}
