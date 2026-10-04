"""Register one explicitly authorized ALFWorld queue; no training/GPUs/API."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from xml.etree import ElementTree

from phase3.common import require, write_new
from skillnet_cohort.common import file_hash
from .launch import CODE, dependencies, fingerprint, hardware


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cpu-tests', type=Path, required=True)
    p.add_argument('--prescreen', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    require(root == Path('/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v7-20260927'),
            'This migration is authorized only for the current ALFWorld queue')
    suite = ElementTree.parse(args.cpu_tests).getroot()
    suites = [suite] if suite.tag == 'testsuite' else list(suite)
    require(sum(int(s.get('tests', '0')) for s in suites) >= 158 and all(
        int(s.get('failures', '0')) == int(s.get('errors', '0')) == 0 for s in suites), 'CPU acceptance incomplete')
    screen = json.loads(args.prescreen.read_text())
    require(screen['rows'] and all(r['max_logprob_error'] <= 1e-5 and r['max_entropy_error'] <= 1e-5
                                  for r in screen['rows']), 'Forward prescreen incomplete')
    branch = root / 'runs/skillrl_failure'
    native = branch / 'checkpoints/global_step_15'
    from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
    require(validate_full_checkpoint(native)['world_size'] == 8, 'Expected native eight-rank U15')
    batch = branch / 'direction_batches/u0011.pt'
    config = branch / 'segments/u0010-u0015.json'
    original = Path('/mnt/workspace/users/wangyifan/skill-RL/SkillRL')
    stable = [args.cpu_tests, args.prescreen, original / 'phase3/training.py', original / 'phase3/speed_dispatch.py',
              root / 'assets-launch-v1/manifest.json', root / 'assets-launch-v1/runtime.json']
    stable += [original / 'configs' / name for name in
               ('phase3_vllm_v1.json', 'phase3_vllm_memory_v2.json', 'phase3_vllm_memory_v3.json')]
    probe = [batch, config] + sorted(path for path in native.rglob('*') if path.is_file())
    initial = Path('/mnt/workspace/users/wangyifan/model/Qwen3.5-4B')
    probe += sorted(initial.glob('*.safetensors')) + [initial / 'config.json']
    value = dict(schema='alfworld.phase3.rl_speed.request.v1', created_utc=datetime.now(timezone.utc).isoformat(),
        upstream_commit='742f1aa99840b267d644634fe8a4b0fd44b1683f', candidate=str(CODE), run_root=str(root),
        acceptance_root=str(CODE.parent / 'eight-gpu-acceptance-v1'),
        batch=str(batch), checkpoint=str(native), baseline_config=str(config),
        start_updates={'skillrl_failure':15, 'readout_magnitude':0, 'readout_gated_d':0, 'readout_p':0, 'readout_c':0},
        source_hashes=fingerprint(), dependencies=dependencies(), hardware=hardware(),
        inputs={str(path.resolve()): file_hash(path) for path in stable},
        probe_inputs={str(path.resolve()): file_hash(path) for path in probe},
        authorization='User requested local acceptance then next RL boundary activation; no running worker interrupted',
        status='registered_pending_eight_gpu_acceptance; not an acceptance receipt',
        upstream_sha256_verification='124 entries verified in extracted pristine archive',
        numerical_limits={'logprob_entropy_max_abs':1e-5, 'global_preclip_gradient_relative_l2':.03,
                          'global_preclip_gradient_cosine_min':.9995, 'allocated_headroom_gib':2.})
    target = root / 'rl-speed-next-window-v1.json'
    write_new(target, value)
    print('REGISTERED_PENDING_LOCAL_EIGHT_GPU_ACCEPTANCE', target, flush=True)


if __name__ == '__main__':
    main()
