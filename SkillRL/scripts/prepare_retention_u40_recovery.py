"""Seal the U40 recovery without depending on future retired model directories."""
from pathlib import Path

from scripts import resume_two_arm_u50_retention as launch
from phase3.common import require, strict_json, write_new
from skillnet_cohort.common import file_hash


def main():
    previous = launch.ROOT / 'recovery-two-arm-u50-editor-u35-v2'
    old = strict_json((previous / 'authorization.json').read_text())
    require(strict_json((previous / 'stopped.json').read_text())['returncode'] == 1,
            'Previous queue not stopped')
    prior_launcher = launch.REPO / 'scripts/resume_two_arm_u50_editor.py'
    backup = launch.OUTPUT / 'prior-resume-two-arm-u50-editor.py'
    require(file_hash(backup) == old['sources'][str(prior_launcher)],
            'Original recovery launcher backup differs')
    changed = [name for name, sha in old['sources'].items() if file_hash(name) != sha]
    require(changed == [str(prior_launcher)], 'Unexpected source changes')
    run = launch.ROOT / 'runs/skillrl_failure'
    missing = run / 'models/u0035/phase2_export.json'
    absent = [name for name in old['evidence'] if not Path(name).exists()]
    require(absent == [str(missing)], 'Unexpected missing evidence')
    for name, sha in old['evidence'].items():
        if name != str(missing):
            require(file_hash(name) == sha, 'Prior evidence changed: ' + name)
    receipt = run / 'events/u0040/retention_complete.json'
    registration = {'receipt': str(receipt), 'receipt_sha256': file_hash(receipt)}
    launch.validate_retired_export(str(missing), registration)
    sources = {name: file_hash(name) for name in old['sources']}
    for path in (Path(launch.__file__).resolve(), Path(__file__).resolve()):
        sources[str(path)] = file_hash(path)
    evidence = dict(old['evidence'])
    # Persistent receipts/metrics only: do not register metadata inside model
    # directories that the unchanged rolling-retention policy will remove.
    for path in (previous / 'authorization.json', previous / 'stopped.json',
                 previous / 'skillrl_failure-runner.log', backup,
                 run / 'logs/train-u0040-u0045.log', run / 'metrics/u0040.json',
                 run / 'predictions/u0035-u0040/complete.json',
                 run / 'events/u0040/complete.json',
                 run / 'events/u0040/identity.json', receipt):
        evidence[str(path)] = file_hash(path)
    record = dict(old)
    record.pop('failed_editor_request', None)
    record.pop('failed_editor_result_sha256', None)
    record.update(schema='phase3.two_arm_u50.retention_u40_recovery.v1',
                  utc=launch.now(), starts=launch.STARTS, sources=sources,
                  evidence=evidence, retired_evidence={str(missing): registration},
                  completed_results_replayed=False, editor_retry_authorized=False,
                  no_scientific_protocol_change=True)
    write_new(launch.AUTHORITY, record)
    launch.authority()
    print('U40_RETENTION_RECOVERY_AUTHORITY_SEALED', launch.AUTHORITY)


if __name__ == '__main__':
    main()
