"""CPU regression on retained real readout shards; no target labels or rollouts.

Run from SkillRL with its configured Python. All derived files go to a NEW
recovery-v4/offline-check directory, not into the experiment window.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from skillnet_cohort.common import file_hash, read_json, write_new_bytes, write_new_json
from skillnet_cohort.evaluate import verify_prediction
from phase2 import aggregate
from phase2.window_forecast import lock_prediction


def main():
    torch.set_num_threads(1)
    cohort = Path('artifacts/phase12/skillnet37-independent-s404-505-606-v4').resolve()
    audit = cohort/'recovery-v4'
    original = cohort/'seed-404/windows/u0000-u0005-valid_unseen'
    root = audit/'offline-check'
    root.mkdir(exist_ok=False)
    for name in ('protocol.json', 'manifest.json'):
        write_new_bytes(root/name, (original/name).read_bytes())
    for name in ('models', 'batches', 'old_logprobs', 'optimizer_steps'):
        (root/name).symlink_to((original/name).resolve(), target_is_directory=True)
    for path in (original/'audits').glob('*.json'):
        write_new_bytes(root/'audits'/path.name, path.read_bytes())
    (root/'evaluations').mkdir()
    (root/'evaluations/u0000').symlink_to(original/'evaluations/u0000', target_is_directory=True)
    out = root/'window_signals/u0000-u0005'
    out.mkdir(parents=True)
    retained = []
    for path in sorted((original/'window_signals/u0000-u0005').iterdir()):
        if path.is_file():
            (out/path.name).symlink_to(path)
            retained.append({'path': str(path), 'sha256': file_hash(path)})
    raw = read_json(audit/'cpu-initial-parameter-audit.json')
    assert raw['status'] == 'PASS' and raw['precision'] == 'FP32_native_master_parameters'
    # This function was already exercised on the complete, registered 4B
    # parameters in the separately archived actual-model CPU audit. Reuse its
    # result here, avoiding a redundant ~17GB CPU model reconstruction.
    with patch('skillnet_cohort.parameter_delta.registered_initial_delta', return_value=raw), patch.object(
            sys, 'argv', ['aggregate', '--root', str(root), '--update', '5', '--start-update', '0', '--shards', '8']):
        aggregate.main()
    prediction = lock_prediction(root, 0, 5)
    config = read_json(root/'protocol.json')
    verification = verify_prediction(prediction, config['runtime']['preparation'], 5, root/'models/u0005')
    commit, frozen = read_json(out/'committed.json'), read_json(prediction)
    assert commit['live_start_batch_rows'] == 5512 and not commit['gold_read']
    assert frozen['target_gold_read'] is False and not frozen['models']
    assert frozen['protocol_sha256'] == file_hash(original/'protocol.json')
    assert not (original/'window_signals/u0000-u0005/committed.json').exists()
    assert not (original/'evaluations/u0005').exists()
    for item in retained:
        assert file_hash(item['path']) == item['sha256']
    write_new_json(audit/'offline-continuation-audit.json', {'status': 'PASS',
        'source_window': str(original), 'sandbox': str(root), 'original_window_unmodified': True,
        'source_protocol_sha256': file_hash(original/'protocol.json'),
        'actual_start_batch_rows': commit['live_start_batch_rows'],
        'actual_training_decisions': commit['unique_decisions'],
        'parameter_audit_sha256': file_hash(audit/'cpu-initial-parameter-audit.json'),
        'cpu_parameter_reconstruction_repeated_in_this_check': False,
        'prospective_lock': verification, 'target_gold_read': False,
        'environment_rollouts': 0, 'optimizer_updates': 0,
        'scope': 'aggregate + forecast + prediction/checkpoint gate on retained real shards'})
    print('PASS: real-shard aggregate / forecast / checkpoint binding; original window unchanged', flush=True)


if __name__ == '__main__':
    main()
