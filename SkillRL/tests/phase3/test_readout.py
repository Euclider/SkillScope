import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from phase3.common import ProtocolError, digest, strict_json, write_new
from phase3.readout import CompactReadout, SCORES, WindowIdentity, select


def identity():
    return WindowIdentity("readout_d", "a" * 64, "b" * 64, "c" * 64, 0, 5)


def row(skill, *, c=.2, p=-.5, d=.1, balance=.1, magnitude=2., supported=True, version="d" * 64):
    return {"skill_id": skill, "skill_version_sha256": version, "supported": supported,
            "unsupported_reason": None if supported else "no_natural_current_version_tokens",
            "token_count": 100 if supported else 0, "nonzero_advantage_decisions": 20,
            "nonzero_advantage_games": 4, "nonzero_advantage_trajectories": 8,
            "C_upd": c if supported else None, "C_upd_centered": c if supported else None,
            "P_int": p if supported else None, "D_contribution": d if supported else None,
            "D_sign_balance": balance if supported else None, "M_delta_centered": magnitude if supported else None,
            "gate_coverage": .5 if supported else None}


def bundle(rows=None):
    return {"schema_version": "skillrl.phase3.readout.v2", "identity": asdict(identity()),
            "numerical_version": "fp64_zero_sum_readout_v1",
            "aggregation": "game_equal_token_decision_trajectory_game",
            "control": "placebo", "context_id": "all_alfworld", "phase": "all",
            "direction_batch_update": 1, "target_gold_read": False,
            "rows": rows if rows is not None else [row("s1"), row("s2"), row("s3")]}


def run(value=None, *, versions=None, **kwargs):
    value = bundle() if value is None else value
    if versions is None:
        versions = {item["skill_id"]: item["skill_version_sha256"] for item in value["rows"]}
    return select(value, expected=identity(), active_versions=versions,
                  selector=kwargs.pop("selector", "reward_sign_balance"), **kwargs)


def test_fixed_signs_produce_distinct_reward_projection_and_alignment_leaders():
    value = bundle([row("s1", c=.1, p=1., d=3., balance=.9), row("s2", c=.2, p=-4., d=2., balance=.5),
                    row("s3", c=.9, p=0., d=1., balance=.2)])
    before = copy.deepcopy(value)
    for name, chosen in (("reward_sign_balance", "s1"), ("negative_p", "s2"), ("positive_c", "s3")):
        result = run(value, selector=name, k=1)
        assert result["selected"][0]["skill_id"] == chosen
        assert set(result["rankings"]) == set(SCORES)
        assert result["source_sha256"] == digest(value)
    assert value == before


def test_centered_magnitude_and_historical_negative_part_are_independent_selectors():
    value = bundle([row('s1', d=.1, magnitude=4.), row('s2', d=3., magnitude=.2)])
    assert run(value, selector='centered_magnitude', k=1)['selected'][0]['skill_id'] == 's1'
    assert run(value, selector='legacy_gated_d', k=1)['selected'][0]['skill_id'] == 's2'


def test_supported_zero_reward_score_is_not_removed_by_an_unregistered_threshold():
    result = run(bundle([row("s1", balance=0.), row("s2", balance=0.)]))
    assert len(result["selected"]) == 2
    assert result["all_chosen_scores_tied"]
    assert not result["abstain"]


def test_ties_are_gold_blind_and_independent_of_input_order():
    value = bundle([row("s3"), row("s1"), row("s2")])
    result = run(value, k=2)
    assert [item["skill_id"] for item in result["selected"]] == ["s1", "s2"]
    value["rows"].reverse()
    assert run(value, k=2)["selected"] == result["selected"]


@pytest.mark.parametrize("field,value", [
    ("branch_id", "readout_p"), ("bank_sha256", "e" * 64),
    ("old_policy_sha256", "e" * 64), ("new_policy_sha256", "e" * 64),
    ("start", 5), ("end", 10),
])
def test_foreign_provenance_fails(field, value):
    data = bundle()
    data["identity"][field] = value
    with pytest.raises(ProtocolError, match="Foreign"):
        run(data)


@pytest.mark.parametrize("field,value", [
    ("target_gold_read", True), ("control", "null"), ("phase", "early"),
    ("context_id", "clean"), ("direction_batch_update", 5),
])
def test_legacy_or_wrong_window_data_cannot_be_relabelled(field, value):
    data = bundle()
    data[field] = value
    with pytest.raises(ProtocolError):
        run(data)


@pytest.mark.parametrize("field,value", [("target_success", True), ("delta_utility", -.9)])
def test_target_labels_are_not_accepted_as_selector_inputs(field, value):
    data = bundle()
    data["rows"][0][field] = value
    with pytest.raises(ProtocolError, match="fields"):
        run(data)


def test_modified_versions_cannot_inherit_old_scores():
    with pytest.raises(ProtocolError, match="Stale"):
        run(bundle([row("s1")]), versions={"s1": "e" * 64})


def test_new_skill_without_natural_history_abstains_instead_of_zero_fill():
    result = run(bundle([row("s1", supported=False)]), versions={"s1": "d" * 64, "new": "e" * 64})
    assert result["selected"] == []
    assert result["abstain"]
    assert result["supported_count"] == 0
    assert len(result["excluded"]) == 2
    assert all(not items for items in result["rankings"].values())


@pytest.mark.parametrize("field,value", [("nonzero_advantage_decisions", 1),
                                       ("nonzero_advantage_games", 1),
                                       ("nonzero_advantage_trajectories", 1)])
def test_low_natural_support_is_reported_not_filtered(field, value):
    data = bundle([row("s1")]); data["rows"][0][field] = value
    assert run(data)["supported_count"] == 1


def test_support_flag_must_match_token_count():
    data = bundle([row("s1")]); data["rows"][0]["token_count"] = 0
    with pytest.raises(ProtocolError, match="Support flag"):
        run(data)


def test_invoked_skill_with_zero_advantage_stays_in_pool_with_counts_exposed():
    data = bundle([row('s1', balance=0.)])
    data['rows'][0].update(nonzero_advantage_decisions=0,
                           nonzero_advantage_games=0, nonzero_advantage_trajectories=0)
    selection = run(data)
    assert selection['supported_count'] == 1
    assert selection['selected'][0]['support'] == {'decisions':0, 'games':0, 'trajectories':0}


@pytest.mark.parametrize("field,value", [("C_upd", float("nan")), ("C_upd_centered", 2.),
                                       ("D_contribution", -1.), ("D_sign_balance", 2.),
                                       ("P_int", float("inf")), ("P_int", None),
                                       ("M_delta_centered", -1.), ("gate_coverage", 1.1)])
def test_invalid_supported_scores_fail_for_all_selectors(field, value):
    data = bundle()
    data["rows"][0][field] = value
    for selector in SCORES:
        with pytest.raises(ProtocolError):
            run(data, selector=selector)


@pytest.mark.parametrize("k", [0, 6, True, 1.5])
def test_candidate_budget_is_explicit_and_bounded(k):
    with pytest.raises(ProtocolError):
        run(k=k)


def test_five_candidates_are_visible_but_edit_budget_is_separate():
    rows = [row(f's{i}', balance=float(6-i)/10) for i in range(1, 6)]
    versions = {item['skill_id']: item['skill_version_sha256'] for item in rows}
    selection = run(bundle(rows), versions=versions)
    assert selection['candidate_budget'] == 5
    assert len(selection['selected']) == 5


def test_old_policy_batch_failure_invocation_filters_readout_candidates_before_top_five():
    rows = [row(f's{i}', balance=float(7-i)/10) for i in range(1, 7)]
    versions = {item['skill_id']: item['skill_version_sha256'] for item in rows}
    selection = run(bundle(rows), versions=versions,
                    eligible_skill_ids={'s2', 's3', 's4', 's5', 's6'})
    assert [item['skill_id'] for item in selection['selected']] == ['s2', 's3', 's4', 's5', 's6']
    assert selection['supported_count'] == 6 and selection['eligible_supported_count'] == 5
    assert {'skill_id': 's1', 'reason': 'not_invoked_in_old_policy_batch_failures'} in selection['excluded']


@pytest.mark.parametrize("selector", ["P", "negative_c", "gated_d", "best_on_unseen"])
def test_unregistered_selector_cannot_be_chosen(selector):
    with pytest.raises(ProtocolError):
        run(selector=selector)


def test_duplicate_and_unknown_rows_fail():
    with pytest.raises(ProtocolError, match="duplicate"):
        run(bundle([row("s1"), row("s1")]))
    with pytest.raises(ProtocolError, match="Unknown"):
        run(bundle([row("s1")]), versions={"different": "d" * 64})


@pytest.mark.parametrize("start,end", [(1, 6), (0, 10), (150, 155), (-5, 0)])
def test_window_schedule_is_fixed(start, end):
    with pytest.raises(ProtocolError):
        replace(identity(), start=start, end=end)


def test_online_aggregation_matches_existing_token_equations_and_support():
    import numpy as np
    import pandas as pd
    from phase2.stable_direction import token_signals
    from skillnet_cohort.reward_variants import aggregation_weights

    old = torch.tensor([[.3, .4, .3], [.2, .5, .3]], dtype=torch.float64).log()
    new = torch.tensor([[.4, .3, .3], [.1, .6, .3]], dtype=torch.float64).log()
    old_control = torch.tensor([[.2, .5, .3], [.3, .4, .3]], dtype=torch.float64).log()
    new_control = torch.tensor([[.6, .2, .2], [.2, .5, .3]], dtype=torch.float64).log()
    actions = torch.tensor([0, 1])
    accumulator = CompactReadout(identity(), {"s1": "d" * 64, "new": "e" * 64})
    expected = []
    for i in range(22):
        advantage = torch.tensor([1., 1.] if i < 20 else [0., 0.], dtype=torch.float64)
        accumulator.add(skill_id="s1", skill_version_sha256="d" * 64,
                        decision_id=f"decision-{i}", game_id=f"game-{i % 4}", trajectory_id=f"traj-{i % 8}",
                        old_original=old, new_original=new, old_placebo=old_control, new_placebo=new_control,
                        actions=actions, advantages=advantage)
        signals = token_signals(old, new, old_control, new_control, actions, advantage)
        for j in range(len(actions)):
            p = float(signals['P_int'][j])
            expected.append({'decision_id':f'decision-{i}', 'game_id':f'game-{i % 4}',
                'trajectory_id':f'traj-{i % 8}', 'C_upd':float(signals['C_upd'][j]),
                'C_upd_centered':float(signals['C_upd_centered'][j]), 'P_int':p,
                'D_contribution':float(signals['D_contribution'][j]),
                'D_sign_balance':-np.sign(p) if bool(signals['direction_valid'][j]) else 0.,
                'M_delta_centered':float(signals['delta_centered_norm'][j]),
                'gate_coverage':float(signals['gate'][j])})
    result = accumulator.bundle()
    measured = next(item for item in result["rows"] if item["skill_id"] == "s1")
    frame = pd.DataFrame(expected)
    weights = aggregation_weights(frame, 'game')
    for key in ("C_upd", "C_upd_centered", "P_int", "D_contribution", "D_sign_balance", "M_delta_centered", "gate_coverage"):
        assert measured[key] == pytest.approx(weights @ frame[key].to_numpy(float))
    assert measured["nonzero_advantage_decisions"] == 20
    assert measured["nonzero_advantage_games"] == 4
    assert measured["nonzero_advantage_trajectories"] == 8
    assert measured["supported"]
    assert run(result, versions=accumulator.versions)["supported_count"] == 1
    assert not any(isinstance(value, torch.Tensor) for state in accumulator.states.values() for value in state.values())
    # A padded duplicate must not inflate support or change aggregate means.
    with pytest.raises(ProtocolError, match="Duplicate"):
        accumulator.add(skill_id="s1", skill_version_sha256="d" * 64,
                        decision_id="decision-0", game_id="game-0", trajectory_id="traj-0",
                        old_original=old, new_original=new, old_placebo=old_control, new_placebo=new_control,
                        actions=actions, advantages=advantage)


def test_write_new_is_idempotent_but_does_not_overwrite(tmp_path):
    target = tmp_path / "prediction.json"
    write_new(target, {"score": 1})
    write_new(target, {"score": 1})
    with pytest.raises(FileExistsError):
        write_new(target, {"score": 2})
    assert strict_json(target.read_text()) == {"score": 1}
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ProtocolError):
        write_new(link, {"score": 1})


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_strict_json_rejects_ambiguous_values(text):
    with pytest.raises(ProtocolError):
        strict_json(text)


@pytest.mark.parametrize('version,editor_model', [('v3', 'o3'), ('v4', 'gpt-5.5')])
def test_proposed_six_arm_setting_keeps_execution_and_credentials_out(version, editor_model):
    repo = Path(__file__).resolve().parents[2]
    settings = json.loads((repo / f"configs/phase3_setting_embedding_{version}.json").read_text())
    assert {item["selector"] for item in settings["arms"].values()} == {"failure_driven", *SCORES}
    assert list(settings['arms']) == ['readout_d', 'skillrl_failure', 'readout_magnitude',
                                      'readout_gated_d', 'readout_p', 'readout_c']
    assert settings['training']['optimizer_horizon_updates'] == 150
    assert settings['training']['first_execution_stop_update'] == 20
    assert settings['readout']['skill_level_edit_threshold'] is None
    assert settings["editor"]["model"] == editor_model
    assert settings["router"]["model"] == "Qwen3-Embedding-0.6B"
    assert settings["editor"]["base_url"] == "https://api.zhizengzeng.com/v1"
    assert settings["editor"]["key_env"] == "SKILLRL_PHASE3_EDITOR_API_KEY"
    assert "sk-" not in json.dumps(settings)
    assert settings["readout"]["primary_score"] == "D_sign_balance::game::reward"
    assert settings["execution"]["approved"] is True
    assert settings["execution"]["phase3_vllm_eight_card_preflight_passed"] is True
