from phase1.metrics import classify_transition


def test_strict_harmful_flip_uses_ci_bounds():
    assert classify_transition(0.1, 0.4, -0.5, -0.1) == "harmful_sign_flip"


def test_point_flip_with_crossing_ci_is_ambiguous():
    assert classify_transition(-0.1, 0.4, -0.5, 0.1) == "ambiguous"

