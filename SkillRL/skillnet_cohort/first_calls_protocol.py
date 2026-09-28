"""Append-only coverage amendment preparation; leaves active RL sources untouched."""
import argparse
import copy
from pathlib import Path
import time

from .common import REPO, file_hash, read_json, write_new_bytes, write_new_json
from .first_calls_support import build

EXTRA_SOURCES = ('skillnet_cohort/first_calls_support.py', 'skillnet_cohort/first_calls_protocol.py',
    'skillnet_cohort/first_calls_measure.py', 'skillnet_cohort/first_calls_reuse.py',
    'skillnet_cohort/first_calls_report.py', 'skillnet_cohort/first_calls_run.py',
    'skillnet_cohort/first_calls_defer.py')


def prepare(queue_plan, output, *, seed=404, handoff_processes=None, test_report=None, policy_registration=None):
    from .assets import model_inventory
    from .runtime import runtime_settings
    from .seed_queue import verify_sources
    from phase2.protocol import validate_extended
    queue_plan, output = Path(queue_plan).resolve(), Path(output).resolve()
    previous = read_json(queue_plan); verify_sources(previous)
    if seed not in (404, 505, 606):
        raise ValueError('Only the three already registered RL seeds are authorized')
    cohort = Path(previous['root']); source = cohort/f'seed-{seed}'
    expected_output = cohort/'all-first-calls-v1'
    if seed != 404:
        expected_output = expected_output/f'followup-s{seed}'
    if output != expected_output or (output/'plan.json').exists():
        raise FileExistsError('Use a fresh versioned all-first-calls amendment only once')
    if read_json(source/'complete.json')['status'] != 'complete':
        raise ValueError('This explicit amendment requires a completed retained seed')
    if seed == 404 and handoff_processes is None:
        raise ValueError('Register the live seed505 RL tree before arming seed404')
    if seed != 404 and (not policy_registration or handoff_processes):
        raise ValueError('Followup rules must be bound to the original coverage amendment')
    import xml.etree.ElementTree as ET
    report_path = Path(test_report).resolve()
    suites = list(ET.parse(report_path).getroot().iter('testsuite'))
    if not suites or any(int(s.attrib.get(k, 0)) for s in suites for k in ('failures', 'errors')) or sum(int(s.attrib.get('tests', 0)) for s in suites) < 20:
        raise ValueError('A passing CPU regression report is required before preparation')
    old_job = previous['jobs'][previous['seeds'].index(seed)]
    root = output/f'seed-{seed}'; preparation = Path(old_job['preparation'])
    legacy = source/'windows/u0000-u0005-valid_unseen'; old_config = read_json(legacy/'protocol.json')
    support = build(preparation, source/'evaluations/u0000-valid_unseen-anchors', source, root/'support')
    # Verify endpoints against the U0 source binding and original native export.
    model0 = model_inventory(source/'models/u0000'); model5 = model_inventory(source/'models/u0005')
    if model0 != support['source_checkpoint_identity']:
        raise ValueError('Retained U0 differs from the natural-source checkpoint')
    exported = read_json(source/'models/u0005/phase2_export.json')
    if exported['native_weight_parity']['all_parameters_bitwise_equal'] is not True:
        raise ValueError('Retained U5 lacks native parity')
    spec = read_json(preparation.parent/'spec.json')
    jobs = support['anchor_count']*3*(len(spec['evaluation']['evidence_seeds'])+len(spec['evaluation']['gold_seeds']))
    local_calls = support['anchor_count']*2*3*(len(spec['evaluation']['evidence_seeds'])+len(spec['evaluation']['gold_seeds']))*spec['evaluation']['max_steps']
    # Four MiB per continuation plus eight GiB for readout/router/logs.
    # A disk guard remains authoritative; this is not a recording-size guarantee.
    recording_bound = jobs*2*4*2**20 + 8*2**30
    old_files = []
    for path in sorted((source/'reports').glob('*')):
        if path.is_file():
            backup = output/f'archived-reports/seed-{seed}'/path.name
            write_new_bytes(backup, path.read_bytes())
            old_files.append({'path': str(path), 'sha256': file_hash(path), 'archive': str(backup)})
    runtime_hashes = dict(previous['source_sha256'])
    runtime_hashes.update({name: file_hash(REPO/name) for name in EXTRA_SOURCES})
    amendment = {'version': 'all_u0_first_calls_v1', 'approved': True,
        'user_authority': '取消评估数量筛选；每条轨迹每个skill只取第一次调用；等seed505跑完RL后先重算seed404并替换分析报告，后续seed同协议',
        'prior_target_labels_available': True, 'scientific_status': 'retrospective_coverage_amendment',
        'ranking_formula_and_orientations_unchanged': True, 'selection_uses_target_outcomes': False,
        'all_invocations': False, 'first_invocation_per_trajectory_skill': True,
        'minimum_source_calls': None, 'minimum_source_games': None,
        'maximum_skills': None, 'maximum_anchors_per_skill': None,
        'readout_count_thresholds_removed': True, 'missing_scores_are_not_zero': True,
        'effect': 'target payload intervention at this anchor and later target invocations; unchanged prefix',
        'point_weighting': 'equal games, equal first-call source trajectories within game, equal continuation repeats',
        'interval': 'paired game/continuation bootstrap, unchanged paired arm and endpoint contrasts',
        'training_readout_uses_unchanged_actual_batch_decisions': True,
        'single_game_interval': 'undefined, point estimate retained', 'legacy_window': str(legacy)}
    if policy_registration:
        registered = read_json(policy_registration)
        if registered['followup_seeds'] != [505, 606] or registered['amendment']['version'] != amendment['version']:
            raise ValueError('Followups must use the identical earlier frozen coverage rule')
        for key, value in registered['amendment'].items():
            if key != 'legacy_window' and amendment.get(key) != value:
                raise ValueError('Do not tune followup rules after seeing earlier outcomes')
    plan = {'schema_version': 'skillnet.all_first_calls_queue.v1', 'approved': True, 'root': str(output),
        'cohort_root': str(cohort), 'hard_limit_seconds': None, 'source_sha256': runtime_hashes,
        'runtime_limit_override': {'unlimited_wallclock_user_authorized': True, 'storage_limits_unchanged': True},
        'storage': previous['storage'], 'legacy_queue_plan': str(queue_plan),
        'legacy_queue_plan_sha256': file_hash(queue_plan), 'amendment': amendment,
        'jobs': [{'seed': seed, 'preparation': str(preparation), 'preparation_sha256': file_hash(preparation),
            'run_root': str(root), 'training_root': str(source)}],
        'legacy_seal': {'path': str(legacy/'sealed.json'), 'sha256': file_hash(legacy/'sealed.json')},
        'engineering_tests': {'path': str(report_path), 'sha256': file_hash(report_path), 'status': 'PASS'},
        'handoff_processes': handoff_processes, 'followup_seeds': [505, 606] if seed == 404 else [],
        'policy_registration': str(policy_registration) if policy_registration else None,
        'policy_registration_sha256': file_hash(policy_registration) if policy_registration else None,
        'legacy_reports': old_files, 'model_inventory': {'u0000': model0, 'u0005': model5},
        'expected_anchor_count': support['anchor_count'], 'expected_skills': support['observed_skill_count'],
        'continuations_per_endpoint': jobs, 'projected_recording_upper_bytes': recording_bound,
        'wait_for_seed505_RL': seed == 404, 'interrupt_training': False,
        'defer_legacy_supervisors_only_after_training_child_exit': True,
        'resume_legacy_supervisors_on_completion_or_error': True,
        'report_publication': 'archive first, replace only after all new evaluations and checks pass',
        'automatic_retry': False}
    write_new_json(output/'plan.json', plan)
    old_admission = read_json(old_job['admission'])
    admission = {**old_admission, 'projected_peak_run_bytes': recording_bound,
        'coverage_amendment': amendment, 'timing_is_not_completion_guarantee': True,
        'scope': 'no RL/export/performance resampling; missing readout and all-first-call continuations only'}
    write_new_json(output/'admission.json', admission)
    permit = {'approved': True, 'operations': ['evaluation', 'readout'],
        'preparation_sha256': file_hash(preparation), 'run_root': str(root), 'gpu_ids': list(range(8)),
        'router_cache_path': str(root/'router.sqlite3'), 'router_max_local_calls': local_calls, 'router_max_api_calls': 0,
        'storage': {**previous['storage'], 'cohort_storage_root': str(cohort)},
        'budget_admission': {'path': str(output/'admission.json'), 'sha256': file_hash(output/'admission.json')},
        'budget_started_unix': time.time(), 'budget_deadline_unix': None, 'unlimited_wallclock': True,
        'shared_queue_plan': str(output/'plan.json'), 'shared_queue_plan_sha256': file_hash(output/'plan.json')}
    write_new_json(output/'permit.json', permit)
    config = copy.deepcopy(old_config)
    config.update(root=str(root), coverage_amendment=amendment)
    config['signals'].update(minimum_nonzero_advantage_decisions=0, minimum_training_games=0,
        minimum_training_trajectories=0, support_policy='all_computable_on_batch_scores_no_count_filter')
    config['ranking']['shared_candidate_pool'] = 'All observed skills with computable finite scores and paired utility; counts descriptive only'
    config['ranking']['bootstrap_status'] = amendment['interval']
    config['evaluation'].update(anchor_sets=support['anchor_sets'], anchor_count=support['anchor_count'],
        anchor_scope='all_first_invocations_per_trajectory_skill', statistical_cluster='game_id')
    config['runtime'] = {**runtime_settings(spec, permit['router_cache_path'], 0, max_local_calls=local_calls),
        'preparation': str(preparation), 'authorization_path': str(output/'permit.json'), 'data_root': spec['data_root']}
    for name in ('models', 'batches', 'old_logprobs', 'optimizer_steps'):
        (root/name).symlink_to(source/name, target_is_directory=True)
    write_new_json(root/'resource_limits.json', permit['storage'])
    write_new_json(root/'protocol.json', config)
    write_new_json(root/'manifest.json', {'registered_protocol_sha256': file_hash(root/'protocol.json'),
        'skill_bank_sha256': spec['bank_manifest_sha256'], 'support_manifest_sha256': file_hash(root/'support/manifest.json'),
        'amendment_plan_sha256': file_hash(output/'plan.json')})
    validate_extended(config, REPO)
    signal = legacy/'window_signals/u0000-u0005'
    files = sorted([p for p in signal.iterdir() if '-shard-' in p.name or p.name.startswith('shard-')])
    write_new_json(root/'readout-reuse.json', {'signal_directory': str(signal),
        'training_batch_sha256': support['training_batch_sha256'],
        'files': [{'path': str(p), 'sha256': file_hash(p)} for p in files],
        'legacy_protocol_sha256': file_hash(legacy/'protocol.json'), 'policy_paths_unchanged': True})
    write_new_bytes(root/'signals/calibration.json', (legacy/'signals/calibration.json').read_bytes())
    print({'status': 'PREPARED_NOT_STARTED', 'skills': support['observed_skill_count'],
        'all_first_call_anchors': support['anchor_count'], 'total_continuations': jobs*2,
        'recording_upper_GiB': recording_bound/2**30, 'plan': str(output/'plan.json')}, flush=True)
    return plan


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--test-report', type=Path, required=True)
    p.add_argument('--queue-pid', type=int, required=True)
    p.add_argument('--supervisor-pid', type=int, required=True)
    p.add_argument('--training-pid', type=int, required=True)
    a = p.parse_args()
    from .first_calls_defer import bind_handoff
    bindings = bind_handoff(a.queue_pid, a.supervisor_pid, a.training_pid, read_json(a.queue_plan)['root'])
    prepare(a.queue_plan, a.output, handoff_processes=bindings, test_report=a.test_report)
