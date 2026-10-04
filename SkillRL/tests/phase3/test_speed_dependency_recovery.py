"""Catch missing transitive imports in the isolated training deployment."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_isolated_speed_candidate_imports_complete_training_entrypoint():
    candidate = os.environ.get('PHASE3_SPEED_TEST_CANDIDATE')
    if not candidate:
        pytest.skip('Requires the local isolated speed deployment')
    code = '''
from pathlib import Path
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from gigpo import core_gigpo
from phase3 import training
assert callable(run_ppo) and callable(training.execute)
assert Path(core_gigpo.__file__).resolve().is_relative_to(Path.cwd())
print('COMPLETE_TRAINING_IMPORT_OK')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=Path(candidate),
        env={**os.environ, 'PYTHONPATH': candidate, 'CUDA_VISIBLE_DEVICES': '',
             'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
        text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'COMPLETE_TRAINING_IMPORT_OK' in result.stdout


def test_retry_log_only_redirects_confirmed_pretraining_import_failure(tmp_path):
    from scripts.resume_phase3_speed_import import recovery_log
    from skillnet_cohort.common import file_hash
    from phase3.common import ProtocolError
    run = tmp_path / 'runs/skillrl_failure'
    log = run / 'logs/train-u0015-u0020.log'
    log.parent.mkdir(parents=True)
    log.write_text("ModuleNotFoundError: No module named 'gigpo'\n")
    output = tmp_path / 'recovery'
    args = ['phase3.training', '--root', run, '--branch', 'skillrl_failure', '--start', 15, '--execute']
    options = dict(run_root=run, output=output, expected_failure_sha256=file_hash(log))
    assert recovery_log(args, log, **options) == output / 'train-u0015-u0020-recovery.log'
    other = run / 'logs/predict-u0020.log'
    assert recovery_log(['phase3.predict'], other, **options) == other
    with pytest.raises(ProtocolError):
        recovery_log(args[:-2] + [10, '--execute'], log, **options)
    (run / 'metrics').mkdir()
    (run / 'metrics/u0016.json').write_text('{}')
    with pytest.raises(ProtocolError):
        recovery_log(args, log, **options)
