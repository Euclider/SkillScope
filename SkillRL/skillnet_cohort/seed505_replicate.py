"""One-shot handoff from sealed seed505 readouts to eight-repeat utility labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import REPO, file_hash, read_json, write_new_json
from .seed505_readout import COHORT, OUTPUT as READOUT
from .seed505_precision import OUTPUT as UTILITY, prepare as prepare_utility, launch as launch_utility

OUTPUT = COHORT/'seed505-eight-repeat-coordinator-v1'
PROTOCOL = REPO/'2026-09-24-seed505-eight-repeat-replication-protocol.md'
MODULE = 'skillnet_cohort.seed505_replicate'


def utc():
    return datetime.now(timezone.utc).isoformat()


def alive(pid, module):
    proc = Path('/proc')/str(pid)
    try:
        state = (proc/'stat').read_text().split(') ')[1][0]
        cmdline = (proc/'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
    except FileNotFoundError:
        return False
    return state != 'Z' and module in cmdline


def wait_stage(root, module, pid, stage):
    count = 0
    while True:
        if (root/'complete.json').exists():
            value = read_json(root/'complete.json')
            if value.get('status') != 'complete':
                raise ValueError(stage+' completion marker invalid')
            return value
        if (root/'stopped.json').exists():
            raise RuntimeError(stage+' stopped; original evidence preserved')
        if not alive(pid, module):
            raise RuntimeError(stage+' exited without sealed completion')
        if count % 5 == 0:
            write_new_json(OUTPUT/'heartbeats'/f'{stage}-{count:06d}.json',
                {'utc': utc(), 'stage': stage, 'pid': pid, 'readout_complete':
                 (READOUT/'complete.json').exists(), 'utility_complete':
                 (UTILITY/'complete.json').exists()})
        count += 1
        time.sleep(60)


def run():
    if not (OUTPUT/'launch.json').exists() or (OUTPUT/'run-intent.json').exists():
        raise FileExistsError('Only a fresh explicit coordinator attempt is allowed')
    launch = read_json(READOUT/'launch.json')
    if launch['plan_sha256'] != file_hash(READOUT/'plan.json'):
        raise ValueError('Readout launch binding changed')
    write_new_json(OUTPUT/'run-intent.json', {'pid': os.getpid(), 'started_utc': utc(),
        'readout_pid': launch['pid'], 'readout_plan_sha256': launch['plan_sha256'],
        'protocol_sha256': file_hash(PROTOCOL), 'automatic_retry': False})
    stage = 'readout'
    try:
        complete = wait_stage(READOUT, 'skillnet_cohort.seed505_readout',
            launch['pid'], stage)
        if complete.get('readout_columns') != 285 or complete.get('new_api_calls') != 0:
            raise ValueError('Readout result differs from frozen plan')
        from .seed505_readout import binding as check_readout
        check_readout(READOUT, models=True)
        if file_hash(PROTOCOL) != read_json(OUTPUT/'run-intent.json')['protocol_sha256']:
            raise ValueError('Replication protocol changed')
        from .first_calls_defer import gpu_users
        stage = 'gpu_handoff'
        while gpu_users():
            time.sleep(30)
        stage = 'utility_prepare'
        if UTILITY.exists():
            raise FileExistsError('Utility attempt already exists; never overwrite or auto-retry')
        prepare_utility(UTILITY)
        stage = 'utility_launch'
        launch_utility(UTILITY)
        downstream = read_json(UTILITY/'launch.json')
        write_new_json(OUTPUT/'utility-handoff.json', {'utc': utc(),
            'readout_provenance_sha256': file_hash(READOUT/'provenance.json'),
            'utility_plan_sha256': downstream['plan_sha256'],
            'utility_pid': downstream['pid'], 'score_columns': 285})
        stage = 'utility'
        result = wait_stage(UTILITY, 'skillnet_cohort.seed505_precision',
            downstream['pid'], stage)
        if result.get('gold_repeats') != 8 or result.get('readout_columns') != 285:
            raise ValueError('Utility completion differs from frozen protocol')
        write_new_json(OUTPUT/'complete.json', {'status': 'complete', 'utc': utc(),
            'seed': 505, 'readout_provenance_sha256': file_hash(READOUT/'provenance.json'),
            'utility_provenance_sha256': file_hash(UTILITY/'provenance.json'),
            'all_285_direction_and_magnitude_reported': True})
        print(json.dumps({'status': 'COMPLETE', 'report': str(UTILITY/'reports/phase2-results.md')}),
              flush=True)
    except BaseException as error:
        write_new_json(OUTPUT/'stopped.json', {'stage': stage, 'error': repr(error),
            'utc': utc(), 'automatic_retry': False})
        raise


def launch():
    if OUTPUT.exists():
        raise FileExistsError('Only a fresh one-shot coordinator launch is allowed')
    readout = read_json(READOUT/'launch.json')
    if not alive(readout['pid'], 'skillnet_cohort.seed505_readout') and not (
            READOUT/'complete.json').exists():
        raise RuntimeError('Readout is neither active nor complete')
    OUTPUT.mkdir(parents=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(REPO)+os.pathsep+env.get('PYTHONPATH', ''))
    with (OUTPUT/'workflow.log').open('xb') as stream:
        proc = subprocess.Popen([sys.executable, '-u', '-B', '-m', MODULE, 'run'],
            cwd=REPO, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write_new_json(OUTPUT/'launch.json', {'pid': proc.pid, 'started_utc': utc(),
        'readout_pid': readout['pid'], 'protocol_sha256': file_hash(PROTOCOL),
        'automatic_retry': False})
    print(json.dumps({'status': 'LAUNCHED_NOT_COMPLETE', 'pid': proc.pid,
        'output': str(OUTPUT)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'run'))
    mode = parser.parse_args().mode
    {'launch': launch, 'run': run}[mode]()


if __name__ == '__main__':
    main()
