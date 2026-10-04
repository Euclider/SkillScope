"""Pure configuration and acceptance rules; never changes the RL recipe."""
import math


def probe_indices(mask, advantage, count):
    lengths = mask.sum(-1).tolist()
    signs = advantage.sum(-1).sign().tolist()
    buckets = {}
    for i, (length, sign) in enumerate(zip(lengths, signs)):
        if length:
            buckets.setdefault((int(sign), int(length)), []).append(i)
    selected = []
    while len(selected) < count:
        before = len(selected)
        for key in sorted(buckets):
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop(0))
        if len(selected) == before:
            raise ValueError('Insufficient distinct nonempty response rows')
    return selected


def acceptable(report):
    ranks = report.get('ranks', [])
    if len(ranks) != 8:
        return False
    for row in ranks:
        fields = ('max_logprob_error', 'max_entropy_error', 'grad_relative_l2',
                  'grad_cosine', 'peak_gib', 'total_gib', 'seconds')
        if not all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in fields):
            return False
        if not (row.get('passed') and row.get('finite') and row.get('world_size') == 8
                and row.get('rows_per_rank') == 16 and row.get('optimizer_boundaries') == 1
                and row['max_logprob_error'] <= 1e-5 and row['max_entropy_error'] <= 1e-5
                and row['grad_relative_l2'] <= .03 and row['grad_cosine'] >= .9995
                and row['peak_gib'] <= row['total_gib'] - 2):
            return False
    return True


def choose_options(reports):
    if not acceptable(reports.get('suffix', {})):
        raise ValueError('Local eight-GPU suffix probe has not passed')
    return dict(response_logits_only=True,
                reference_gpu_root=acceptable(reports.get('reference_gpu_root', {})),
                accumulate_no_sync=acceptable(reports.get('no_sync', {})))


def apply_options(cfg, options):
    from omegaconf import OmegaConf
    changes = {
        'actor_rollout_ref.actor.response_logits_only': options['response_logits_only'],
        'actor_rollout_ref.ref.response_logits_only': options['response_logits_only'],
        'actor_rollout_ref.actor.accumulate_no_sync': options['accumulate_no_sync'],
        'actor_rollout_ref.actor.trim_common_padding': False,
        'actor_rollout_ref.ref.trim_common_padding': False,
    }
    if options['reference_gpu_root']:
        changes.update({
            'actor_rollout_ref.ref.fsdp_config.cpu_offload': False,
            'actor_rollout_ref.ref.fsdp_config.param_offload': False,
            'actor_rollout_ref.ref.fsdp_config.wrap_policy.disable': True,
        })
    for name, value in changes.items():
        OmegaConf.update(cfg, name, value, force_add=True)
    return cfg
