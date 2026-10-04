import pytest
from phase3.common import ProtocolError, write_new


def test_continuation_rejects_replay_and_other_arms():
    from scripts.resume_two_arm_u50 import validate_window
    validate_window('skillrl_failure', 25)
    validate_window('readout_d', 20)
    for branch, start in [('skillrl_failure', 20), ('readout_d', 15),
                          ('readout_p', 20), ('readout_d', 50), ('readout_d', 21)]:
        with pytest.raises(ProtocolError):
            validate_window(branch, start)


def test_reward_waits_for_both_skillrl_u50_evaluation_splits(tmp_path):
    from scripts.resume_two_arm_u50 import milestone_ready
    path = tmp_path / 'runs/skillrl_failure/milestones/u0050'
    assert not milestone_ready(tmp_path, 'skillrl_failure')
    write_new(path / 'complete.json', {'branch': 'skillrl_failure', 'endpoint': 50})
    write_new(path / 'valid_seen/complete.json', {'complete': True})
    assert not milestone_ready(tmp_path, 'skillrl_failure')
    write_new(path / 'valid_unseen/complete.json', {'complete': True})
    assert milestone_ready(tmp_path, 'skillrl_failure')
