"""Finish the existing gate process, then replace its stopped controller safely."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from phase3.common import require, strict_json
from .runtime import ROOT, validate


def process(pid):
    path=Path('/proc')/str(pid)
    try:
        # Split after comm, whose contents can contain spaces or parentheses.
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        return {'state':fields[0],'start':fields[19],
                'argv':(path/'cmdline').read_bytes().split(b'\0')[:-1]}
    except FileNotFoundError:
        return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--controller',type=int,required=True)
    p.add_argument('--controller-start',required=True)
    p.add_argument('--gate',type=int,required=True)
    p.add_argument('--gate-start',required=True)
    args=p.parse_args()
    root=args.root.resolve()
    validate(root)
    controller=process(args.controller)
    require(controller is not None and controller['start']==args.controller_start
            and controller['state']=='T' and str(root).encode() in controller['argv']
            and b'phase3.logicbench_loop' in controller['argv'],'Unexpected old controller')
    print('Waiting for the existing U5 editor/gate process; no API replay.',flush=True)
    while True:
        gate=process(args.gate)
        if gate is None or gate['state']=='Z': break
        require(gate['start']==args.gate_start and str(root).encode() in gate['argv'], 'Gate PID reused')
        time.sleep(10)
    seal=root/'windows/u0000-u0005/complete.json'
    if not seal.is_file():
        current=process(args.controller)
        if current and current['start']==args.controller_start:
            os.kill(args.controller,signal.SIGCONT)  # Let the old controller record the failure and exit.
        raise RuntimeError('U5 did not seal; no training or API retry authorized by this handoff')
    require(strict_json(seal.read_text())['end']==5,'Wrong handoff boundary')
    validate(root)
    require(not (root/'direction_batches/u0006.pt').exists(),'U6 already started')
    gpu_ids=strict_json((root/'launch.json').read_text())['gpu_ids']
    require(len(gpu_ids)==4 and len(set(gpu_ids))==4 and all(type(i) is int for i in gpu_ids),
            'Invalid frozen GPU placement')
    executable=ROOT/'scripts/run_logicbench_phase3_fast.sh'
    command=[str(executable),'--setting',str(root/'setting.json'),'--root',str(root),
             '--gpus',','.join(map(str,gpu_ids)),'--execute']
    current=process(args.controller)
    require(current is not None and current['start']==args.controller_start and current['state']=='T',
            'Controller changed during gate')
    # Only the stopped supervisor is removed. The gate has exited and its output
    # is sealed; all native checkpoints remain under the existing retention rules.
    os.kill(args.controller,signal.SIGKILL)
    for _ in range(100):
        current=process(args.controller)
        if current is None or current['state']=='Z': break
        time.sleep(.1)
    require(current is None or current['state']=='Z','Old controller did not exit')
    # The stopped controller had not yet performed its post-seal retention step.
    # Do that before startup's next-window disk reservation check.
    from phase3.logicbench_loop import cleanup
    from skillnet_cohort.common import exclusive_writer
    with exclusive_writer(root):
        cleanup(root,5)
    (root/'performance/handoff-complete.json').write_text(json.dumps({
        'previous_controller':args.controller,'new_controller':os.getpid(),
        'boundary':5,'u0_u5_reused':True,'timestamp':time.time()},indent=2)+'\n')
    print('U5 sealed; starting optimized U6–U50 runtime.',flush=True)
    os.execv(str(executable),command)


if __name__=='__main__': main()
