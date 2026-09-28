"""Appendix-ready compact summaries; incomplete runs are never labelled complete."""
from __future__ import annotations

import argparse
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .common import digest, require, strict_json, write_new
from .evolution import paired_effect
from .prepare import ARMS


def api_totals(path):
    if not path.exists():
        return {'attempts': 0, 'completed_responses': 0, 'failed': 0, 'unreconciled': 0,
                'usage': {}, 'latency_seconds': 0., 'provider_cost': None}
    db = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)
    try:
        results = [row[0] for row in db.execute('SELECT result FROM attempts')]
    finally:
        db.close()
    usage, seconds, statuses, missing = Counter(), 0., Counter(), Counter()
    for item in results:
        if item is None:
            statuses['unreconciled'] += 1
            continue
        value = strict_json(item)
        checksum = value.pop('record_sha256')
        require(checksum == digest(value), 'Changed API ledger result')
        statuses[value['status']] += 1
        seconds += value['accounting']['latency_seconds']
        for name, number in value['accounting']['usage'].items():
            if number is None:
                missing[name] += 1
            else:
                usage[name] += number
    return {'attempts': len(results), 'completed_responses': statuses['success'], 'failed': statuses['failed'],
            'unreconciled': statuses['unreconciled'], 'usage_known_subtotals': dict(usage),
            'usage_missing_response_counts': dict(missing), 'latency_seconds': seconds, 'provider_cost': None}


def success_summary(rows):
    require(rows, 'No evaluation rows')
    seen = set()
    by_task, by_game = defaultdict(list), defaultdict(list)
    prompt, completion, steps, invalid, skills = 0, 0, 0, 0, set()
    for row in rows:
        key = (row['game_id'], row['eval_seed'])
        require(key not in seen and type(row['success']) is bool, 'Duplicate/invalid evaluation')
        seen.add(key)
        by_task[row['task_type']].append(row['success'])
        by_game[row['game_id']].append(int(row['success']))
        for step in row['steps']:
            prompt += step['prompt_tokens']
            completion += step['completion_tokens']
            steps += 1
            invalid += not step['is_action_valid']
            if step.get('selected_skill_id'):
                skills.add(step['selected_skill_id'])
    return {'games': len(by_game), 'episodes': len(rows), 'success_rate': sum(sum(v)/len(v) for v in by_game.values()) / len(by_game),
            'six_task_breakdown': {task: {'episodes':len(v), 'success_rate':sum(v)/len(v)} for task,v in sorted(by_task.items())},
            'actor_prompt_tokens': prompt, 'actor_completion_tokens': completion, 'steps': steps,
            'invalid_actions': invalid, 'invalid_action_rate': invalid / steps if steps else None, 'unique_invoked_skills': len(skills)}


def actor_totals(rows):
    """Observed generation tokens, including unsuccessful/invalid-action episodes."""
    totals = Counter()
    for row in rows:
        totals['episodes'] += 1
        for step in row['steps']:
            totals['steps'] += 1
            for name in ('prompt_tokens', 'completion_tokens'):
                value = step[name]
                require(type(value) is int and value >= 0, 'Invalid actor token accounting')
                totals[name] += value
    return dict(totals)


def actor_accounting(root):
    stages = defaultdict(list)
    for path in sorted((root / 'episodes').glob('*/*/*.json')):
        row = strict_json(path.read_text())
        stages['training' if row['split'] == 'train' else 'seen_monitor'].append(row)
    for stage, paths in (
            ('paired_gate', (root / 'events').glob('*/evaluations/*/episodes/*.json')),
            ('milestone', (root / 'milestones').glob('u*/*/episodes/*.json')),
            ('final', (root / 'final').glob('*/episodes/*.json'))):
        for path in sorted(paths):
            row = strict_json(path.read_text())
            require(row['result_sha256'] == digest(row['result']), 'Changed evaluation accounting')
            stages[stage].append(row['result'])
    return {'by_stage': {key: actor_totals(rows) for key, rows in stages.items()},
            'total': actor_totals([row for rows in stages.values() for row in rows]),
            'scope': 'Persisted complete episodes only; unfinished rollout work is not inferred as zero.'}


def summarize(root):
    root = Path(root)
    plan = strict_json((root / 'plan.json').read_text())
    events = [strict_json(path.read_text()) for path in sorted((root / 'events').glob('*/complete.json'))]
    proposed, accepted = Counter(), Counter()
    for event in events:
        proposed.update(event.get('changes', {}).get('counts', {}))
        if event['accepted']:
            accepted.update(event['changes']['counts'])
    milestones = {}
    for directory in sorted((root / 'milestones').glob('u*')):
        if not (directory / 'complete.json').exists():
            continue
        marker = strict_json((directory / 'complete.json').read_text())
        splits = {}
        for split in ('valid_seen', 'valid_unseen'):
            records = [strict_json(path.read_text()) for path in sorted((directory / split / 'episodes').glob('*.json'))]
            for item in records:
                require(item['result_sha256'] == digest(item['result']), 'Changed milestone episode')
            require((directory / split / 'complete.json').is_file(), 'Incomplete milestone split')
            splits[split] = success_summary([item['result'] for item in records])
        milestones[directory.name] = {'record': marker, 'splits': splits}
    latest_milestone_splits = milestones[sorted(milestones)[-1]]['splits'] if milestones else {}
    final = latest_milestone_splits if (root / 'complete.json').is_file() else {}
    if not milestones:
        for split in ('valid_seen', 'valid_unseen'):
            directory = root / 'final' / split
            if (directory / 'complete.json').exists():
                records = [strict_json(path.read_text()) for path in sorted((directory / 'episodes').glob('*.json'))]
                for item in records:
                    require(item['result_sha256'] == digest(item['result']), 'Changed final episode')
                final[split] = success_summary([item['result'] for item in records])
    readouts = [strict_json(path.read_text()) for path in (root / 'predictions').glob('*/complete.json')]
    editor_protocols = Counter(event.get('editor_protocol', 'legacy_unversioned') for event in events)
    mixed_editor_protocols = len(editor_protocols) > 1
    shadows = [strict_json(path.read_text()) for path in sorted((root / 'events').glob('*/shadow_action_bias.json'))]
    require(all(row['edit_threshold'] is None and row['threshold_decision_applied'] is False
                and row['shadow_record_is_edit_gate'] is False for row in shadows), 'A shadow action bias was used as an edit gate')
    from .embedding_routing import local_totals
    router_cost = local_totals(root / 'router-local.sqlite3') if plan.get('router_backend') == 'skillrl_embedding_state' else api_totals(root / 'router.sqlite3')
    measured_readout = {'windows': len(readouts), 'forward_calls': sum(r['forward_calls'] for r in readouts),
                        'forward_input_tokens': sum(r['forward_input_tokens'] for r in readouts),
                        'wall_seconds': sum(r['wall_seconds'] for r in readouts)}
    zero_readout = {'windows': 0, 'forward_calls': 0, 'forward_input_tokens': 0, 'wall_seconds': 0.}
    failure_arm = plan['selector'] == 'failure_driven'
    return {'branch': plan['branch'], 'complete': (root / 'complete.json').is_file(),
        'completed_edit_opportunities': len(events), 'proposals': sum(e['proposed'] for e in events),
        'accepted_events': sum(e['accepted'] for e in events), 'candidate_rejections': sum(e['proposal_rejected'] for e in events),
        'rollbacks': sum(e['rollback_count'] for e in events), 'outcomes': dict(Counter(e['outcome'] for e in events)),
        'proposed_operation_counts': dict(proposed), 'accepted_operation_counts': dict(accepted),
        'event_records': events, 'milestones': milestones, 'latest_milestone': sorted(milestones)[-1] if milestones else None,
        'editor_protocol_counts': dict(editor_protocols), 'mixed_editor_protocols': mixed_editor_protocols,
        'latest_milestone_splits': latest_milestone_splits,
        'final': final, 'actor': actor_accounting(root), 'router': router_cost,
        'editor': api_totals(root / 'editor.sqlite3'),
        'readout': {**measured_readout, 'decision_use': not failure_arm,
                    'operational_selector_cost': zero_readout if failure_arm else measured_readout,
                    'passive_shadow_cost': measured_readout if failure_arm else zero_readout},
        'shadow_action_bias': {'windows': len(shadows), 'event_records': shadows,
                               'threshold': None, 'edit_gate_applied': False},
        'interpretation': ('One RL seed per arm; game/decoding repeats are not independent training seeds. '
                           + ('This arm mixes editor evidence protocols across windows; do not present it as a uniform-protocol comparison.'
                              if mixed_editor_protocols else ''))}


def paired_game_ci(before, after, repeats=2000):
    import numpy as np
    effect = paired_effect(before, after)
    deltas = np.asarray(effect['per_game_deltas'])
    rng = np.random.default_rng(404)
    means = np.asarray([rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(repeats)])
    return {**effect, 'paired_game_bootstrap_95_interval': np.quantile(means, [.025,.975]).tolist(),
            'bootstrap_repeats': repeats, 'uncertainty_scope': 'games, not RL training seeds'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON path; never overwrites a report')
    args = parser.parse_args()
    write_new(args.output, summarize(args.root))
    print(str(args.output))


if __name__ == '__main__':
    main()
