import pandas as pd
import pytest

from phase2 import utilities


def test_semantic_change_uses_gold_only_and_weights_games_equally(monkeypatch, tmp_path):
    rows = []
    # Three anchors from game A must not outweigh one from game B.
    for game, anchors in (("A", range(3)), ("B", range(1))):
        for anchor in anchors:
            for update in (30, 31):
                for purpose in ("evidence", "gold"):
                    for arm in ("original", "placebo", "null"):
                        success = (purpose == "gold" and game == "A" and update == 31 and arm == "original")
                        # Evidence deliberately points in the opposite direction.
                        success |= purpose == "evidence" and update == 30 and arm == "original"
                        rows.append(dict(update=update, purpose=purpose, skill_id="cle_003",
                                         anchor_id=f"{game}-{anchor}", game_id=game,
                                         phase="middle", trigger_step=7, arm=arm, success=success))
    monkeypatch.setattr(utilities, "read_evaluations", lambda root: pd.DataFrame(rows))
    units, _, margins = utilities.units(tmp_path)
    row = units[(units.control == "placebo") & (units.phase == "all")].iloc[0]
    assert row.delta_utility == pytest.approx(.5)
    assert row.original_new == pytest.approx(.5)
    assert row.delta_control == 0
    assert row.delta_utility == row.delta_original-row.delta_control
    assert row.game_count == 2
    assert row.anchor_count == 4
    assert set(margins.purpose) == {"evidence", "gold"}


def test_duplicate_gold_arm_is_rejected():
    row = dict(update=30, purpose="gold", skill_id="s", anchor_id="a", game_id="g",
               phase="initial", trigger_step=0, arm="original", success=True)
    with pytest.raises(ValueError, match="Duplicate"):
        utilities.margins(pd.DataFrame([row, row]))


def test_no_on_batch_support_is_missing_not_zero_risk():
    from phase2.aggregate import unsupported_row
    row = unsupported_row(31, "placebo", "gen_002", "all", 1e-8, [])
    assert row["supported"] is False
    assert row["nonzero_advantage_games"] == 0
    assert pd.isna(row["D_contribution"])
    assert pd.isna(row["P_int"])
