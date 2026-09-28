"""All observed skills and all trajectory/skill first calls, without count cutoffs.

This module is deliberately separate from the source-frozen running RL queue.
No environment reset, model forward, outcome-based selection or source edit.
"""
from collections import Counter, defaultdict
from pathlib import Path
import json

from phase1.archive import stable_hash
from .common import REPO, digest, file_hash, load_preparation, read_json, write_new_bytes, write_new_json

PHASES = ('all', 'initial', 'early', 'middle', 'late')


def state_phase(step):
    return 'initial' if step == 0 else 'early' if step < 5 else 'middle' if step < 15 else 'late'


def call_anchor(trajectory, trajectory_path, index, step):
    """Only the earliest target invocation is an evaluation anchor."""
    steps = trajectory['steps']; trigger = steps[step]; prefix = steps[:step]
    skill = trigger['selected_skill_id']
    if not skill or trigger['step_index'] != step or not 0 <= step < index['max_steps']:
        raise ValueError('Not a natural, in-horizon invocation')
    if any(s.get('selected_skill_id') == skill for s in prefix):
        raise ValueError('Later target invocations are descriptive calls, not anchors')
    history = [{'observation': s.get('observation', ''), 'action': s.get('projected_action', '')} for s in prefix]
    state = {'game_id': trajectory['game_id'], 'trigger_step': step,
        'task_description': trajectory.get('task_description', ''), 'observation': trigger.get('observation'),
        'admissible_actions': trigger.get('admissible_actions', []), 'history': history}
    identity = {'source_trajectory_id': trajectory['trajectory_id'], 'skill_id': skill, 'trigger_step': step}
    return {'schema_version': 'phase1.all_first_invocation_anchor.v1',
        'anchor_id': stable_hash(identity)[:24], 'state_id': stable_hash(state)[:24],
        'skill_id': skill, 'context_id': 'all_alfworld', 'split': index['split'],
        'game_id': trajectory['game_id'], 'environment_seed': trajectory.get('environment_seed', index['environment_seed']),
        'source_eval_seed': trajectory.get('eval_seed', index['eval_seed']),
        'source_checkpoint_id': trajectory.get('checkpoint_id', index['checkpoint_id']),
        'source_trajectory_id': trajectory['trajectory_id'], 'source_trajectory_path': str(trajectory_path),
        'task_description': trajectory.get('task_description', ''), 'trigger_step': step,
        'max_steps': index['max_steps'], 'remaining_steps': index['max_steps']-step,
        'prefix_actions': [s.get('projected_action', '') for s in prefix], 'prefix_history': history,
        'prefix_rewards': [s.get('reward', 0.) for s in prefix],
        'trigger_observation': trigger.get('observation'),
        'trigger_admissible_actions': trigger.get('admissible_actions', []),
        'trigger_prompt_text': trigger.get('prompt_text'),
        'trigger_router_state_flags': trigger.get('skill_router_state_flags', []),
        'trigger_router_scores': trigger.get('skill_router_scores', {}),
        'trigger_selected_skill_id': skill, 'state_hash': stable_hash(state),
        'invocation_ordinal': 1, 'phase': state_phase(step),
        'anchor_scope': 'first_invocation_per_trajectory_and_skill_in_registered_U0_unseen_source'}


def training_coverage(batch_file, skill_ids):
    import torch
    torch.set_num_threads(1)
    batch = torch.load(batch_file, map_location='cpu', weights_only=False)
    b = batch['tensors']; seen = set(); buckets = {}
    for skill in skill_ids:
        for phase in PHASES:
            buckets[skill, phase] = {'decisions': 0, 'loss_tokens': 0, 'nonzero_advantage_decisions': 0,
                'games': set(), 'trajectories': set(), 'nonzero_games': set(), 'nonzero_trajectories': set()}
    for row, meta in enumerate(batch['metadata']):
        if meta['decision_id'] in seen:
            continue
        seen.add(meta['decision_id'])
        skill = meta['info'].get('selected_skill_id')
        if skill not in skill_ids:
            if skill is not None:
                raise ValueError('Foreign training skill')
            continue
        mask = b['phase2_actual_loss_mask'][row].bool()
        adv = b['advantages'][row, mask]
        if not torch.isfinite(adv).all():
            raise ValueError('Nonfinite actual advantage')
        nonzero = bool((adv.abs() > 1e-12).any())
        for phase in ('all', state_phase(meta['environment_step'])):
            v = buckets[skill, phase]
            v['decisions'] += 1; v['loss_tokens'] += int(mask.sum())
            v['games'].add(meta['info']['extra.gamefile']); v['trajectories'].add(meta['trajectory_id'])
            if nonzero:
                v['nonzero_advantage_decisions'] += 1
                v['nonzero_games'].add(meta['info']['extra.gamefile']); v['nonzero_trajectories'].add(meta['trajectory_id'])
    rows = []
    for (skill, phase), v in sorted(buckets.items()):
        rows.append({'skill_id': skill, 'phase': phase,
            **{f'train_{k}': len(x) if isinstance(x, set) else x for k, x in v.items()},
            'readout_input_available': v['loss_tokens'] > 0,
            'reward_direction_observed': v['nonzero_advantage_decisions'] > 0,
            'legacy_count_thresholds_met': v['nonzero_advantage_decisions'] >= 20
                and len(v['nonzero_games']) >= 4 and len(v['nonzero_trajectories']) >= 8})
    return rows


def build(preparation, source, training_root, output):
    from agent_system.memory.frozen_skill_bank import load_skillnet37
    from .evaluate import collect_results, job_plan, result_path
    preparation, source, training_root, output = map(lambda x: Path(x).resolve(),
        (preparation, source, training_root, output))
    load_preparation(preparation)
    spec = read_json(preparation.parent/'spec.json'); plan = read_json(source/'plan.json')
    canonical = job_plan(preparation, training_root/'models/u0000', 0, 'valid_unseen', 'anchors')
    if any(plan.get(k) != v for k, v in canonical.items()):
        raise ValueError('Full registered U0 unseen source required')
    rows, missing = collect_results(plan, source)
    if missing or read_json(source/'completion.json')['plan_sha256'] != digest(plan):
        raise ValueError('Incomplete or changed U0 anchor-source traversal')
    bank = load_skillnet37(); by_skill = {s: [] for s in bank.skill_ids}
    source_files = []; calls = Counter(); call_games = defaultdict(set); call_trajectories = defaultdict(set)
    for row in sorted(rows, key=lambda r: r['job']['job_id']):
        t = row['result']; path = result_path(source, row['job']['job_id'])
        source_files.append({'path': str(path), 'sha256': file_hash(path)})
        if [s['step_index'] for s in t['steps']] != list(range(len(t['steps']))):
            raise ValueError('Incomplete or unordered U0 trajectory')
        index = {**row['job'], 'checkpoint_id': 'u0000', 'max_steps': spec['evaluation']['max_steps']}
        occurrences = Counter()
        for step, event in enumerate(t['steps']):
            skill = event.get('selected_skill_id')
            if skill is None:
                continue
            if skill not in by_skill:
                raise ValueError('Foreign U0 skill')
            occurrences[skill] += 1
            for phase in ('all', state_phase(step)):
                calls[skill, phase] += 1
                call_games[skill, phase].add(t['game_id'])
                call_trajectories[skill, phase].add(t['trajectory_id'])
            if occurrences[skill] == 1:
                by_skill[skill].append(call_anchor(t, path, index, step))
    training = training_coverage(training_root/'batches/u0001/training_batch.pt', set(bank.skill_ids))
    train = {(r['skill_id'], r['phase']): r for r in training}
    controls = read_json(preparation.parent/'placebos.json')['controls']
    coverage, sets = [], []
    for skill, anchors in sorted(by_skill.items()):
        anchors.sort(key=lambda a: (a['game_id'], a['source_trajectory_id'], a['trigger_step']))
        if anchors:
            stem = skill.replace(':', '--'); path = output/(stem+'.jsonl'); placebo = output/(stem+'.placebo.json')
            write_new_bytes(path, ''.join(json.dumps(a, ensure_ascii=False, sort_keys=True)+'\n' for a in anchors).encode())
            write_new_json(placebo, {**controls[skill], 'original_text': bank.get(skill).payload})
            sets.append({'skill_id': skill, 'context_id': 'all_alfworld', 'anchors_path': str(path),
                'anchors_sha256': file_hash(path), 'placebo_path': str(placebo), 'placebo_sha256': file_hash(placebo)})
        for phase in PHASES:
            subset = anchors if phase == 'all' else [a for a in anchors if a['phase'] == phase]
            coverage.append({**train[skill, phase], 'context_id': 'all_alfworld',
                'source_calls': calls[skill, phase], 'source_trajectories': len(call_trajectories[skill, phase]),
                'source_games': len(call_games[skill, phase]),
                'first_call_anchors': len(subset), 'later_call_anchors': 0,
                'later_calls_descriptive_only': calls[skill, phase]-len(subset),
                'selected_anchors': len(subset), 'utility_evaluable': bool(subset),
                'quantity_filter_applied': False,
                'missing_utility_reason': None if subset else 'no_first_invocation_in_registered_U0_source_phase',
                'missing_readout_reason': None if train[skill, phase]['readout_input_available'] else 'no_actual_U0_training_loss_token'})
    import pandas as pd
    write_new_bytes(output/'coverage.csv', pd.DataFrame(coverage).to_csv(index=False).encode())
    result = {'schema_version': 'skillnet.all_first_calls_support.v1', 'preparation': str(preparation),
        'preparation_sha256': file_hash(preparation), 'source_plan_sha256': file_hash(source/'plan.json'),
        'source_update': 0, 'split': 'valid_unseen', 'source_checkpoint_identity': plan['checkpoint_identity'],
        'source_files': source_files, 'source_episode_count': len(rows), 'all_candidate_count': 37,
        'observed_skill_count': len(sets), 'coverage': coverage, 'anchor_sets': sets,
        'anchor_count': sum(map(len, by_skill.values())), 'quantity_filter_applied': False,
        'natural_call_count': sum(calls[s, 'all'] for s in by_skill),
        'maximum_skills': None, 'maximum_anchors_per_skill': None,
        'selection': 'all first natural calls per source trajectory and skill; later calls counted but not anchored',
        'readout_batch_decisions_unchanged': True,
        'target_outcomes_used_for_selection': False, 'training_batch_sha256': file_hash(training_root/'batches/u0001/training_batch.pt')}
    write_new_json(output/'manifest.json', result)
    return result
