"""Audited reuse of matching O/P/N episodes; never overwrite source evidence."""
from pathlib import Path
import json

from phase1.archive import stable_hash
from phase2.protocol import anchor_sets, evaluation_identity, evaluation_jobs
from .common import REPO, file_hash, read_json, write_new_bytes, write_new_json

ANCHOR_SEMANTIC_FIELDS = ('anchor_id', 'state_hash', 'state_id', 'skill_id', 'game_id',
    'environment_seed', 'source_eval_seed', 'source_checkpoint_id', 'source_trajectory_id',
    'task_description', 'trigger_step', 'max_steps', 'remaining_steps', 'prefix_actions',
    'prefix_history', 'prefix_rewards', 'trigger_observation', 'trigger_admissible_actions',
    'trigger_prompt_text', 'trigger_selected_skill_id')
EVAL_FIELDS = ('arms', 'old_evidence_seeds', 'gold_seeds', 'temperature', 'top_p',
    'history_length', 'max_steps', 'max_new_tokens', 'max_prompt_tokens')


def validate_compatibility(config, old, root, legacy):
    if (config['run_id'] != old['run_id'] or config['rl_path_id'] != old['rl_path_id']
            or any(config['evaluation'][k] != old['evaluation'][k] for k in EVAL_FIELDS)):
        raise ValueError('Episode reuse changes policy/arm/seed/horizon semantics')
    for key in ('bank_manifest_sha256', 'router_profile_sha256', 'router_backend',
                'router_model_path', 'router_device', 'inference_profile', 'data_root', 'preparation'):
        if config['runtime'][key] != old['runtime'][key]:
            raise ValueError('Episode runtime changed: '+key)
    for update in (0, 5):
        if (root/'models'/f'u{update:04d}').resolve() != (legacy/'models'/f'u{update:04d}').resolve():
            raise ValueError('Reuse requires identical retained model paths')
    new_sets = {s['skill_id']: s for s in anchor_sets(config, REPO)}
    for source in anchor_sets(old, REPO):
        target = new_sets[source['skill_id']]
        if source['placebo'] != target['placebo']:
            raise ValueError('Changed target payload/control')
        by_id = {a['anchor_id']: a for a in target['anchors']}
        for a in source['anchors']:
            b = by_id[a['anchor_id']]
            if any(a.get(k) != b.get(k) for k in ANCHOR_SEMANTIC_FIELDS):
                raise ValueError('Changed replay state for a retained anchor')


def import_endpoint(root, update):
    root = Path(root).resolve(); config = read_json(root/'protocol.json')
    legacy = Path(config['coverage_amendment']['legacy_window'])
    old = read_json(legacy/'protocol.json')
    plan = read_json(root.parent/'plan.json')
    seal = Path(plan['legacy_seal']['path'])
    if seal != legacy/'sealed.json' or file_hash(seal) != plan['legacy_seal']['sha256']:
        raise ValueError('Episode reuse lacks the original sealed evidence binding')
    hashes = {str(legacy/r['path']): r['sha256'] for r in read_json(seal)['files']}
    def verified(path):
        if hashes.get(str(path)) != file_hash(path):
            raise ValueError('Sealed source changed: '+str(path))
    validate_compatibility(config, old, root, legacy)
    if update == 5:
        locked = read_json(root/'window_signals/u0000-u0005/prediction.json')
        if locked['coverage_amendment']['prior_target_labels_available'] is not True:
            raise ValueError('Disclose retrospective coverage amendment before reusing old target labels')
    directory = root/'evaluations'/f'u{update:04d}'
    if directory.exists():
        raise FileExistsError('Endpoint was already opened; no implicit import retry')
    source_dir = legacy/'evaluations'/f'u{update:04d}'
    source_rows = {}
    for path in sorted(source_dir.glob('shard-*.jsonl')):
        verified(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['trajectory_id'] in source_rows:
                raise ValueError('Duplicate retained continuation')
            source_rows[row['trajectory_id']] = row
    for rank in range(8):
        verified(source_dir/f'shard-{rank}-complete.json')
    markers = [read_json(source_dir/f'shard-{rank}-complete.json') for rank in range(8)]
    if any(m['protocol_sha256'] != file_hash(legacy/'protocol.json') or m['max_jobs'] is not None for m in markers):
        raise ValueError('Incomplete or changed old continuation endpoint')
    jobs = evaluation_jobs(config, REPO)
    expected = {stable_hash(evaluation_identity(config, update, job))[:24] for job in jobs}
    if not set(source_rows).issubset(expected):
        raise ValueError('Retained continuations are not a subset of all-first-call jobs')
    imported, by_rank = [], [[] for _ in range(8)]
    for position, job in enumerate(jobs):
        identity = evaluation_identity(config, update, job); tid = stable_hash(identity)[:24]
        if tid not in source_rows:
            continue
        row = source_rows[tid]; original = Path(row['trajectory_path']); result = read_json(original)
        verified(original)
        if (any(row.get(k) != v for k, v in identity.items()) or not result['steps']
                or not result['prefix_replay_verified'] or result['trajectory_id'] != tid
                or result['actual_continuation_seed'] != identity['continuation_seed']
                or any(result.get(k) != v for k, v in row.items())):
            raise ValueError('Retained trajectory identity/continuation is not complete')
        anchor = job[1]
        if any(result['original_anchor'].get(k) != anchor.get(k) for k in ANCHOR_SEMANTIC_FIELDS):
            raise ValueError('Retained trajectory replay anchor changed')
        target = directory/'trajectories'/identity['skill_id']/(tid+'.json')
        provenance = {'source_path': str(original), 'source_sha256': file_hash(original),
            'source_protocol_sha256': file_hash(legacy/'protocol.json'), 'episode_reexecuted': False}
        value = {**result, 'trajectory_path': str(target), 'original_anchor': anchor, 'reused_evidence': provenance}
        write_new_json(target, value)
        by_rank[position % 8].append({**row, 'trajectory_path': str(target), 'reused_evidence': provenance})
        imported.append({'trajectory_id': tid, **provenance})
    for rank, rows in enumerate(by_rank):
        if rows:
            write_new_bytes(directory/f'shard-{rank}.jsonl', ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows).encode())
    write_new_json(root/'reuse'/f'u{update:04d}.json', {'expected_total': len(jobs), 'reused': len(imported),
        'missing_to_evaluate': len(jobs)-len(imported), 'records': imported,
        'prior_target_labels_available': update == 5, 'semantics_equal': True})
    return len(imported)
