"""Register only the API diagnostic/retry repair; retain accepted RL sources."""
from pathlib import Path
import sqlite3
from scripts import resume_two_arm_u50_editor as launch
from phase3.common import require, strict_json, write_new, digest
from skillnet_cohort.common import file_hash


def main():
    previous = launch.ROOT/'recovery-two-arm-u50-router-handoff-v2'
    old = strict_json((previous/'authorization.json').read_text())
    require(strict_json((previous/'stopped.json').read_text())['returncode'] == 1, 'Previous queue not failed')
    changed = []
    for name, sha in old['sources'].items():
        if file_hash(name) != sha:
            changed.append(name)
    require(changed == [str(launch.REPO/'phase3/api.py')], 'Unexpected orchestration source changes')
    run = launch.ROOT/'runs/skillrl_failure'
    key = '425c059be1df8bc54a365fc6a7d754bfd17db486f43a8f397302bb93a1ce9f7c'
    with sqlite3.connect(f'file:{run}/editor.sqlite3?mode=ro', uri=True) as db:
        req, result = db.execute('SELECT request,result FROM attempts WHERE key=?',(key,)).fetchone()
    request, record = strict_json(req), strict_json(result)
    require(digest(request) == key and request['identity']['event_id'] == 'u0035'
            and record['failure']['type'] == 'ProtocolError' and record['status'] == 'failed', 'Wrong failed request')
    sources = {name:file_hash(name) for name in old['sources']}
    for p in (Path(launch.__file__).resolve(),Path(__file__).resolve()):
        sources[str(p)] = file_hash(p)
    evidence = dict(old['evidence'])
    for p in (previous/'authorization.json', previous/'stopped.json',
              previous/'skillrl_failure-runner.log', run/'metrics/u0035.json',
              run/'predictions/u0030-u0035/complete.json',run/'models/u0035/phase2_export.json'):
        evidence[str(p)] = file_hash(p)
    record = {k:v for k,v in old.items() if k not in
              ('failed_artifacts','archive_targets','router_retry','replay_scope','completed_results_replayed')}
    record.update(schema='phase3.two_arm_u50.editor_u35_recovery.v1',utc=launch.now(),
        starts=launch.STARTS, sources=sources, evidence=evidence,
        previous_api_sha256=old['sources'][str(launch.REPO/'phase3/api.py')],
        failed_editor_request=key, failed_editor_result_sha256=digest(strict_json(result)),
        completed_results_replayed=False, editor_prompt_and_constraints_unchanged=True)
    write_new(launch.AUTHORITY, record)
    launch.authority()
    print('U35_EDITOR_RECOVERY_AUTHORITY_SEALED',launch.AUTHORITY)


if __name__ == '__main__':
    main()
