"""Outcome-blind, user-confirmed small-cohort scope and admission calculations."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from .common import file_hash, read_json


PROFILE = Path(__file__).resolve().parents[1] / 'configs/phase12_30h_budget_v1.json'
OFFLOAD_PROFILE = PROFILE.with_name('phase12_30h_budget_v2.json')
VLLM_SCOPE_PROFILE = PROFILE.with_name('phase12_single_window_v3.json')
INDEPENDENT_PROFILE = PROFILE.with_name('phase12_independent_windows_v4.json')


def load_budget(path):
    profile = read_json(path)
    if profile not in tuple(read_json(p) for p in (PROFILE, OFFLOAD_PROFILE, VLLM_SCOPE_PROFILE, INDEPENDENT_PROFILE)):
        raise ValueError('Unregistered day-budget profile; confirm a new version instead of silently adapting scope')
    if (profile['schema_version'] != 'skillnet.phase12.budget_profile.v1'
            or profile['execution_authorized'] or not profile['scientific_scope_confirmed']):
        raise ValueError('A scope profile must not grant experiment execution authority')
    return profile


def apply_budget(spec, path):
    profile = load_budget(path)
    if spec['seed'] not in profile.get('registered_seeds', [404]) or spec['training']['gpus'] != 8 or spec['router_backend'] != 'skillrl_embedding_state_batch':
        raise ValueError('The confirmed budget profile binds registered seeds, eight GPUs and the batched router')
    spec = deepcopy(spec)
    spec['budget_profile'] = {'path': str(Path(path).resolve()), 'sha256': file_hash(path), 'settings': profile}
    spec['training'].update(iterations=profile['iterations'], rollout_microbatch_per_gpu=profile['rollout_microbatch_per_gpu'])
    if 'actor_optimizer_offload' in profile:
        spec['training']['actor_optimizer_offload'] = profile['actor_optimizer_offload']
    else:
        spec['training'].pop('actor_optimizer_offload', None)
    spec['readout']['windows'] = [{'start': 0, 'end': 5, 'role': 'test'}]
    spec['readout']['horizon'] = profile['window_horizon']
    if profile.get('capture_scope'):
        spec['readout']['capture_scope'] = profile['capture_scope']
        spec['readout']['capture'] = 'window-start actual batch/advantages/live OLD FP32 probabilities; endpoint replay on identical tokens'
        spec['readout']['additional_fixed_ranking_scores'] = deepcopy(profile.get('additional_fixed_ranking_scores', {}))
    else:
        spec['readout'].pop('capture_scope', None)
        spec['readout'].pop('additional_fixed_ranking_scores', None)
    for key in ('utility_splits', 'anchor_source_seeds', 'evidence_seeds', 'gold_seeds',
                'maximum_gold_skills', 'maximum_anchors_per_skill', 'minimum_anchor_occurrences',
                'minimum_anchor_games', 'gold_skill_selection', 'max_steps', 'max_new_tokens'):
        spec['evaluation'][key] = deepcopy(profile[key])
    spec['evaluation']['parallel_workers'] = profile['parallel_evaluation_workers']
    spec['execution']['hard_limit_seconds'] = profile['hard_limit_seconds']
    spec['execution']['finish_reserve_seconds'] = profile['finish_reserve_seconds']
    return spec


def validate_budget_binding(spec):
    binding = spec.get('budget_profile')
    if binding is None:
        return
    if file_hash(binding['path']) != binding['sha256'] or load_budget(binding['path']) != binding['settings']:
        raise ValueError('Frozen day-budget profile changed after preparation')
    expected = apply_budget(spec, binding['path'])
    if expected != spec:
        raise ValueError('Preparation differs from its frozen day-budget profile')


def select_gold_skills(eligible, settings):
    eligible = sorted(set(eligible))
    cap = settings.get('maximum_gold_skills')
    if cap is None:
        return eligible
    if settings.get('gold_skill_selection') != 'sha256_daybudget_s404_natural_support_v1' or type(cap) is not int or cap <= 0:
        raise ValueError('Unknown outcome-blind gold sampling rule')
    return sorted(eligible, key=lambda sid: (hashlib.sha256(('daybudget-s404:' + sid).encode()).hexdigest(), sid))[:cap]


def workload(spec, games):
    windows = len(spec['readout']['windows'])
    ev, tr = spec['evaluation'], spec['training']
    splits = ev.get('utility_splits', ev['splits'])
    skills = min(37, ev.get('maximum_gold_skills', 37))
    performance = sum(games['splits'][s]['count'] for s in ev['splits']) * len(ev['performance_seeds']) * (windows + 1)
    anchors = sum(games['splits'][s]['count'] for s in splits) * len(ev['anchor_source_seeds']) * windows
    utility = len(splits) * skills * ev['maximum_anchors_per_skill'] * 2 * 3 * (
        len(ev['evidence_seeds']) + len(ev['gold_seeds'])) * windows
    train = tr['iterations'] * tr['games_per_iteration'] * tr['group_size']
    monitor_upper = (tr['iterations'] // tr['monitor_every'] + 1) * tr['monitor_episodes']
    return {'training_trajectories': train, 'monitor_trajectories_upper': monitor_upper,
            'performance_episodes': performance, 'anchor_source_episodes': anchors,
            'utility_continuations_upper': utility,
            'local_router_calls_upper': (train + monitor_upper + performance + anchors + utility) * ev['max_steps'],
            'full_vocab_old_new_bytes_per_response_token': 248320 * 4 * 2,
            'requires_measured_runtime_and_storage_admission': bool(spec.get('budget_profile'))}


def validate_admission(permit, spec, preparation, *, require_full_budget=True):
    """A deadline alone is not evidence that a complete run fits its budget."""
    import time
    binding = permit.get('budget_admission', {})
    if not binding.get('path') or file_hash(binding['path']) != binding.get('sha256'):
        raise PermissionError('Missing or changed measured day-budget admission evidence')
    evidence = read_json(binding['path'])
    profile = spec['budget_profile']['settings']
    if (evidence.get('status') != 'PASS'
            or evidence.get('preparation_sha256') != file_hash(preparation)
            or evidence.get('router_profile_sha256') != spec['router_profile_sha256']
            or not all(evidence.get(k) is True for k in (
                'batch_router_validated', 'native_eight_gpu_microbatch_validated',
                'exact_capture_validated', 'sharded_evaluation_validated'))):
        raise PermissionError('Day-budget runtime/recording evidence is incomplete or belongs to another protocol')
    estimate = evidence.get('projected_total_upper_seconds')
    storage = evidence.get('projected_peak_run_bytes')
    unlimited = permit.get('unlimited_wallclock') is True
    if type(estimate) not in (int, float) or not math.isfinite(estimate) or not 0 < estimate or (not unlimited and estimate > profile['hard_limit_seconds'] - profile['finish_reserve_seconds']):
        raise PermissionError('Measured whole-pipeline time estimate exceeds the registered admission budget')
    if type(storage) is not int or not 0 < storage <= permit['storage']['maximum_run_bytes']:
        raise PermissionError('Projected complete exact recording does not fit the storage cap')
    start, end = permit.get('budget_started_unix'), permit.get('budget_deadline_unix')
    if unlimited:
        plan_path = permit.get('shared_queue_plan')
        if not plan_path or file_hash(plan_path) != permit.get('shared_queue_plan_sha256'):
            raise PermissionError('Unlimited wallclock requires the explicitly amended queue plan')
        plan = read_json(plan_path)
        override = plan.get('runtime_limit_override', {})
        if (plan.get('approved') is not True or 'hard_limit_seconds' not in plan or plan['hard_limit_seconds'] is not None
                or override.get('unlimited_wallclock_user_authorized') is not True
                or override.get('storage_limits_unchanged') is not True
                or not any(job['preparation_sha256'] == file_hash(preparation)
                           and job['run_root'] == permit.get('run_root') for job in plan['jobs'])
                or any(permit['storage'].get(k) != v for k, v in plan['storage'].items())
                or type(start) not in (int, float) or not math.isfinite(start) or start > time.time() or end is not None):
            raise PermissionError('Time-limit removal must preserve seed identity and storage protections')
        return
    if (type(start) not in (int, float) or type(end) not in (int, float)
            or not 0 < end - start <= profile['hard_limit_seconds']
            or not start <= time.time() < end
            or (require_full_budget and end - time.time() < estimate)):
        raise PermissionError('Missing, expired or insufficient absolute wallclock budget')
