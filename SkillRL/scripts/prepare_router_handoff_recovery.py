"""Seal the explicitly authorized October 2 router-OOM recovery inputs."""
from pathlib import Path
import sqlite3
from scripts import resume_two_arm_u50_router as launch
from phase3.common import require, strict_json, write_new
from skillnet_cohort.common import file_hash


def main():
    old = launch.ROOT / 'recovery-two-arm-u50-finite-exp-v2'
    r = strict_json((old/'authorization.json').read_text())
    require(strict_json((old/'stopped.json').read_text())['returncode'] == 1, 'Not the failed queue')
    run = launch.ROOT/'runs/skillrl_failure'
    require((run/'speed-audits/u0026-complete.json').is_file(), 'Missing successful U26 audit')
    targets = ['direction_batches/u0026.pt', 'direction_batches/u0026.json', 'episodes/u0026',
               'metrics/u0026.json', 'segments/u0025-u0030.json', 'speed-profiles/u0025-u0030.json']
    targets += [str(p.relative_to(run)) for p in sorted((run/'speed-audits').glob('u0026-*.json'))]
    inventory = {}
    for relative in targets:
        p = run/relative
        require(p.exists(), 'Missing recovery artifact: ' + relative)
        for item in sorted(p.rglob('*')) if p.is_dir() else [p]:
            if item.is_file():
                inventory[str(item.relative_to(run))] = file_hash(item)
    ledger = run/'router-local.sqlite3'
    with sqlite3.connect(f'file:{ledger}?mode=ro', uri=True) as db:
        failed = [(key,bank) for key,bank,result in db.execute('SELECT key,bank,result FROM local_attempts')
                  if result and strict_json(result)['status'] == 'failed']
    require(len(failed) == 15 and len({bank for _,bank in failed}) == 1, 'Failed query set changed')
    candidate = Path('/mnt/workspace/users/wangyifan/phase3-speed-742f1aa-BlbHzX/candidate-router-handoff-v2')
    receipt = candidate.parent/'router-handoff-acceptance-v2/receipt.json'
    require(strict_json(receipt.read_text())['passed'], 'Memory/router parity not accepted')
    for name, sha in r['sources'].items():
        require(file_hash(name) == sha, 'Prior orchestration source changed: ' + name)
    sources = dict(r['sources'])
    for p in (Path(launch.__file__).resolve(), Path(__file__).resolve(),
              launch.REPO/'scripts/check_router_memory_handoff_v2.py',
              launch.REPO/'verl/utils/fsdp_utils.py', launch.REPO/'verl/workers/fsdp_workers.py'):
        sources[str(p)] = file_hash(p)
    evidence = dict(r['evidence'])
    for p in (receipt, old/'stopped.json', old/'train-skillrl-u0025-u0030-recovery.log'):
        evidence[str(p)] = file_hash(p)
    r.update(schema='phase3.two_arm_u50.router_handoff.v2', utc=launch.now(),
        candidate=str(candidate), numeric_receipt=str(receipt), sources=sources, evidence=evidence,
        archive_targets=targets, failed_artifacts=inventory,
        router_retry={'ledger':str(ledger), 'bank_cache':str(run/'router-local.banks'/f'{failed[0][1]}.sqlite3'),
                      'keys':sorted(key for key,_ in failed)},
        completed_results_replayed=True, replay_scope='Only uncheckpointed U26; original evidence archived',
        save_freq=1, checkpoint_retention='two rolling native saves plus window-start until sealed endpoint',
        no_scientific_protocol_change=True)
    write_new(launch.AUTHORITY, r)
    launch.authority()
    print('ROUTER_HANDOFF_AUTHORITY_SEALED', launch.AUTHORITY)


if __name__ == '__main__':
    main()
