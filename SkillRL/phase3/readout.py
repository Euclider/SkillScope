"""Fixed-sign, label-free selectors and compact online token aggregation.

These selectors consume endpoint projections, not a sum of five per-update
scores. Raw C/P/D are retained together. No selector decides an edit or uses
post-edit/held-out outcomes. GPU forward adapters are deliberately separate.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from types import MappingProxyType

from .common import ProtocolError, digest, finite, positive_int, require, sha256


SCORES = MappingProxyType({
    "reward_sign_balance": ("D_sign_balance", 1),
    "centered_magnitude": ("M_delta_centered", 1),
    "legacy_gated_d": ("D_contribution", 1),
    "negative_p": ("P_int", -1),
    "positive_c": ("C_upd_centered", 1),
})
NUMERICAL_VERSION = "fp64_zero_sum_readout_v1"
AGGREGATION = "game_equal_token_decision_trajectory_game"


@dataclass(frozen=True)
class WindowIdentity:
    branch_id: str
    bank_sha256: str
    old_policy_sha256: str
    new_policy_sha256: str
    start: int
    end: int

    def __post_init__(self):
        require(isinstance(self.branch_id, str) and bool(self.branch_id), "Missing branch identity")
        for value in (self.bank_sha256, self.old_policy_sha256, self.new_policy_sha256):
            sha256(value)
        positive_int(self.start, "start", zero=True)
        require(type(self.end) is int and self.start % 5 == 0 and self.end == self.start + 5
                and self.end <= 150, "Readout requires a registered five-update window")


def select(bundle, *, expected: WindowIdentity, active_versions, selector, k=5,
           eligible_skill_ids=None):
    """Return all registered rankings on the same naturally invoked pool, plus top-k.

    Missing current versions abstain. A stale row is a provenance error, not
    missing support. Ties use stable skill IDs, never utility or outcomes.
    Supported zero D is retained: no new skill-level D threshold is introduced.
    """
    require(selector in SCORES, "Unknown readout selector")
    positive_int(k, "candidate budget")
    require(k <= 5, "The confirmed editor candidate budget is at most five")
    require(set(bundle) == {"schema_version", "identity", "numerical_version", "aggregation", "control", "context_id", "phase",
                           "direction_batch_update", "target_gold_read", "rows"}, "Unexpected readout fields")
    require(bundle["schema_version"] == "skillrl.phase3.readout.v2", "Wrong readout schema")
    require(bundle["numerical_version"] == NUMERICAL_VERSION and bundle["aggregation"] == AGGREGATION,
            "Wrong numerical/aggregation protocol")
    require(bundle["identity"] == asdict(expected), "Foreign branch/bank/policy/window readout")
    require(bundle["control"] == "placebo" and bundle["phase"] == "all"
            and bundle["context_id"] == "all_alfworld", "Unexpected control/context/phase")
    require(bundle["direction_batch_update"] == expected.start + 1, "Wrong start-batch direction")
    require(bundle["target_gold_read"] is False, "Readout must be locked without target gold")
    require(bool(active_versions), "Missing active bank versions")
    if eligible_skill_ids is not None:
        eligible_skill_ids = set(eligible_skill_ids)
        require(eligible_skill_ids <= set(active_versions), "Unknown endpoint-evidence skill")
    for skill, version in active_versions.items():
        require(isinstance(skill, str) and bool(skill), "Invalid skill identity")
        sha256(version)
    require(isinstance(bundle["rows"], list), "Readout rows must be a list")
    seen, supported, excluded = set(), [], []
    naturally_supported_count = 0
    fields = {"skill_id", "skill_version_sha256", "supported", "unsupported_reason", "token_count",
              "nonzero_advantage_decisions", "nonzero_advantage_games", "nonzero_advantage_trajectories",
              "C_upd", "C_upd_centered", "P_int", "D_contribution", "D_sign_balance",
              "M_delta_centered", "gate_coverage"}
    for row in bundle["rows"]:
        require(isinstance(row, dict) and set(row) == fields, "Unexpected readout row fields")
        skill = row["skill_id"]
        require(skill in active_versions and skill not in seen, "Unknown/duplicate skill row")
        seen.add(skill)
        require(row["skill_version_sha256"] == active_versions[skill], "Stale skill content version")
        require(type(row["supported"]) is bool, "Support must be explicit")
        counts = [positive_int(row[name], name, zero=True) for name in (
            "nonzero_advantage_decisions", "nonzero_advantage_games", "nonzero_advantage_trajectories")]
        tokens = positive_int(row["token_count"], "token count", zero=True)
        reason = row["unsupported_reason"]
        require(reason is None or isinstance(reason, str), "Invalid support reason")
        require(row["supported"] == (tokens > 0), "Support flag must reflect natural token support")
        if not row["supported"]:
            excluded.append({"skill_id": skill, "reason": reason or "insufficient_current_version_support"})
            continue
        naturally_supported_count += 1
        require(reason is None, "Supported row cannot carry an unsupported reason")
        values = {name: finite(row[name], name) for name in (
            "C_upd", "C_upd_centered", "P_int", "D_contribution", "D_sign_balance",
            "M_delta_centered", "gate_coverage")}
        require(-1.0000001 <= values["C_upd"] <= 1.0000001
                and -1.0000001 <= values["C_upd_centered"] <= 1.0000001
                and values["D_contribution"] >= 0 and -1.0000001 <= values["D_sign_balance"] <= 1.0000001
                and values["M_delta_centered"] >= 0 and 0 <= values["gate_coverage"] <= 1,
                "Readout outside valid range")
        if eligible_skill_ids is not None and skill not in eligible_skill_ids:
            excluded.append({"skill_id": skill, "reason": "not_invoked_in_old_policy_batch_failures"})
            continue
        supported.append({"skill_id": skill, "skill_version_sha256": row["skill_version_sha256"],
                          "raw": values, "scores": {name: sign * values[field] for name, (field, sign) in SCORES.items()},
                          "support": dict(zip(("decisions", "games", "trajectories"), counts))})
    for skill in sorted(set(active_versions) - seen):
        excluded.append({"skill_id": skill, "reason": "no_current_version_readout"})
    rankings = {name: sorted(supported, key=lambda row: (-row["scores"][name], row["skill_id"]))
                for name in SCORES}
    chosen = rankings[selector][:k]
    return {"schema_version": "skillrl.phase3.selection.v3", "identity": asdict(expected),
            "source_sha256": digest(bundle), "selector": selector, "candidate_budget": k,
            "selected": chosen, "rankings": rankings, "excluded": sorted(excluded, key=lambda row: row["skill_id"]),
            "supported_count": naturally_supported_count,
            "eligible_supported_count": len(supported),
            "candidate_pool_rule": ("natural_readout_support" if eligible_skill_ids is None else
                                    "natural_readout_support_intersect_old_policy_batch_failed_invocations"),
            "active_count": len(active_versions),
            "abstain": not chosen, "tie_rule": "score_desc_then_skill_id_asc",
            "all_chosen_scores_tied": bool(chosen) and len({row["scores"][selector] for row in rankings[selector]}) == 1,
            "target_gold_read": False}


class CompactReadout:
    """Aggregate exact token_signals without retaining vocabulary/activation data.

    Caller streams ORIGINAL/PLACEBO old/new distributions from the SAME start
    batch. Full-vocabulary distributions are discarded after each add(). The
    compact per-token scalars reproduce Phase2's game-equal hierarchy exactly.
    """
    def __init__(self, identity: WindowIdentity, active_versions, *, tau_delta=1e-8):
        self.identity = identity
        self.versions = dict(active_versions)
        for version in self.versions.values():
            sha256(version)
        self.tau_delta = finite(tau_delta, "tau_delta")
        require(self.tau_delta >= 1e-8, "Direction noise floor cannot be reduced")
        self.states = defaultdict(lambda: {"tokens": [], "decisions": set(), "games": set(),
                                          "trajectories": set()})
        self.seen = set()

    def add(self, *, skill_id, skill_version_sha256, decision_id, game_id, trajectory_id,
            old_original, new_original, old_placebo, new_placebo, actions, advantages):
        require(skill_id in self.versions and skill_version_sha256 == self.versions[skill_id],
                "Unknown/stale streamed skill")
        require(all(isinstance(value, str) and value for value in (decision_id, game_id, trajectory_id)),
                "Missing natural decision/game/trajectory identity")
        require(decision_id not in self.seen, "Duplicate decision (including distributed padding)")
        from phase2.stable_direction import token_signals
        import torch
        signals = token_signals(old_original, new_original, old_placebo, new_placebo,
                                actions, advantages, tau_delta=self.tau_delta, tau_c=0., epsilon=1e-12)
        require(torch.allclose(signals["P_int"], signals["P_int_centered"], atol=1e-9, rtol=1e-11),
                "Stable projection centering identity failed")
        self.seen.add(decision_id)
        state = self.states[skill_id]
        for index in range(len(actions)):
            projection = float(signals["P_int"][index])
            state["tokens"].append({"decision_id": decision_id, "game_id": game_id,
                "trajectory_id": trajectory_id, "C_upd": float(signals["C_upd"][index]),
                "C_upd_centered": float(signals["C_upd_centered"][index]), "P_int": projection,
                "D_contribution": float(signals["D_contribution"][index]),
                "D_sign_balance": (-1. if projection > 0 else 1. if projection < 0 else 0.)
                    if bool(signals["direction_valid"][index]) else 0.,
                "M_delta_centered": float(signals["delta_centered_norm"][index]),
                "gate_coverage": float(signals["gate"][index])})
        if bool(signals["direction_valid"].any()):
            state["decisions"].add(decision_id)
            state["games"].add(game_id)
            state["trajectories"].add(trajectory_id)

    def bundle(self):
        import numpy as np
        import pandas as pd
        from skillnet_cohort.reward_variants import aggregation_weights
        rows = []
        for skill, version in sorted(self.versions.items()):
            state = self.states[skill]
            n = len(state["tokens"])
            counts = [len(state[name]) for name in ("decisions", "games", "trajectories")]
            supported = n > 0
            values = {}
            if supported:
                frame = pd.DataFrame(state["tokens"])
                weights = aggregation_weights(frame, "game")
                for key in ("C_upd", "C_upd_centered", "P_int", "D_contribution", "D_sign_balance",
                            "M_delta_centered", "gate_coverage"):
                    values[key] = float(np.dot(weights, frame[key].to_numpy(dtype=np.float64)))
            rows.append({"skill_id": skill, "skill_version_sha256": version,
                         "supported": supported, "unsupported_reason": None if supported else "no_natural_current_version_tokens",
                         "token_count": n, "nonzero_advantage_decisions": counts[0],
                         "nonzero_advantage_games": counts[1], "nonzero_advantage_trajectories": counts[2],
                         **{key: values.get(key) for key in ("C_upd", "C_upd_centered", "P_int", "D_contribution",
                             "D_sign_balance", "M_delta_centered", "gate_coverage")}})
        return {"schema_version": "skillrl.phase3.readout.v2", "identity": asdict(self.identity),
                "numerical_version": NUMERICAL_VERSION, "aggregation": AGGREGATION,
                "control": "placebo", "context_id": "all_alfworld", "phase": "all",
                "direction_batch_update": self.identity.start + 1, "target_gold_read": False, "rows": rows}
