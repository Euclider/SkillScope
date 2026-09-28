from phase3.common import write_new
from phase3.report import summarize


def test_mixed_editor_protocol_is_explicit_in_running_summary(tmp_path):
    write_new(tmp_path / 'plan.json', {
        'branch': 'readout_d', 'selector': 'reward_sign_balance',
        'router_backend': 'external_llm'})
    common = {'accepted': False, 'proposed': False, 'proposal_rejected': False,
              'rollback_count': 0, 'outcome': 'abstain_no_supported_candidates_or_evidence'}
    write_new(tmp_path / 'events/u0005/complete.json', {
        **common, 'editor_protocol': 'terminal_failed_current_policy_v11'})
    write_new(tmp_path / 'events/u0010/complete.json', {
        **common, 'editor_protocol': 'same_old_policy_batch_as_readout_v13'})
    result = summarize(tmp_path)
    assert result['mixed_editor_protocols'] is True
    assert result['editor_protocol_counts'] == {
        'terminal_failed_current_policy_v11': 1,
        'same_old_policy_batch_as_readout_v13': 1}
    assert 'do not present it as a uniform-protocol comparison' in result['interpretation']
