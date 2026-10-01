from pathlib import Path
from importlib.metadata import version

from omegaconf import OmegaConf
from phase3.common import digest, require, strict_json
from skillnet_cohort.common import file_hash

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = 'performance/rl-speed-v1.json'
OPTIONS = {
    'actor_rollout_ref.actor.response_logits_only': True,
    'actor_rollout_ref.actor.trim_common_padding': False,
    'actor_rollout_ref.actor.accumulate_no_sync': True,
    'actor_rollout_ref.ref.response_logits_only': True,
    'actor_rollout_ref.ref.trim_common_padding': False,
    'actor_rollout_ref.ref.fsdp_config.cpu_offload': False,
    'actor_rollout_ref.ref.fsdp_config.param_offload': False,
    'actor_rollout_ref.ref.fsdp_config.wrap_policy.disable': True,
}


def apply_options(cfg, *, start):
    if start >= 5:
        for key, value in OPTIONS.items():
            OmegaConf.update(cfg, key, value, force_add=True)
    return cfg


def child_args(args):
    result = list(args)
    if result[0] == 'phase3.logicbench_loop':
        result[0] = 'logicbench_rl_speed.launch'
    return result


def validate(root):
    root = Path(root).resolve()
    receipt = strict_json((root / RECEIPT).read_text())
    require(receipt['root'] == str(root) and receipt['start_update'] == 5
            and receipt['options'] == OPTIONS, 'Changed performance protocol')
    require(receipt['setting_sha256'] == digest(strict_json((root / 'setting.json').read_text())),
            'Performance protocol setting mismatch')
    require(receipt['timeout_receipt_sha256'] == file_hash(root / 'recovery/editor-timeout-v1.json'),
            'Changed editor recovery')
    require(receipt['boundary_seal_sha256'] == file_hash(root / 'windows/u0000-u0005/complete.json'),
            'Changed U5 boundary evidence')
    for name, checksum in receipt['sources'].items():
        require(file_hash(ROOT / name) == checksum, f'Changed performance source: {name}')
    for name, checksum in receipt['validation'].items():
        require(file_hash(ROOT / name) == checksum, f'Changed performance evidence: {name}')
    for name, expected in receipt['runtime_packages'].items():
        require(version(name)==expected, f'Changed performance runtime package: {name}')
    # Existing checks retain the frozen Phase3 and editor recovery contracts.
    from logicbench_phase3_recovery.runtime import _load, verify_implementation
    previous = _load(root)
    current = {name: file_hash(ROOT / name) for name in previous['authorized_implementation']}
    verify_implementation(root, current)
    return receipt
