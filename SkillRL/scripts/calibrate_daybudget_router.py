"""Short, outcome-blind router timing on a readonly, stopped cohort's visible states.

No ALFWorld reset, policy generation, RL update or API call. All new evidence is
written to a new directory; the historical SQLite database is opened read-only.
"""
import argparse
import json
from pathlib import Path
import random
import sqlite3
import time

import numpy as np

from skillnet_cohort.common import file_hash, write_new_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-cache', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sample-count', type=int, default=32)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute or not 2 <= a.sample_count <= 64:
        p.error('Explicit --execute and a bounded 2..64 query sample are required')
    if a.output.exists():
        raise FileExistsError('Use a new calibration directory; never overwrite failed evidence')
    source_hash = file_hash(a.source_cache)
    with sqlite3.connect(a.source_cache.resolve().as_uri() + '?mode=ro', uri=True) as db:
        rows = [json.loads(r[0]) for r in db.execute('SELECT record FROM decisions ORDER BY key')]
    if len(rows) < a.sample_count + 1:
        raise ValueError('Not enough distinct natural visible states')
    from agent_system.memory.skillnet_runtime import create_batched_embedding_skillnet37_runtime
    import torch
    torch.set_num_threads(1)
    memory, router = create_batched_embedding_skillnet37_runtime(model_path=a.model, device='cpu',
        cache_path=a.output / 'router.sqlite3', max_local_calls=a.sample_count + 1)
    py, np_state, th = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    started = time.monotonic()
    router.route(memory.retrieve(''), **rows[0]['visible_input'])
    cold_seconds = time.monotonic() - started
    chosen = rows[1:a.sample_count + 1]
    started = time.monotonic()
    results = router.route_many([{'candidate_bundle': memory.retrieve(''), **r['visible_input']} for r in chosen])
    batch_seconds = time.monotonic() - started
    differences = []
    for before, after in zip(chosen, results):
        differences.append({'input_hash': before['input_hash'],
            'old_selected': before['selected_skill_id'], 'new_selected': after['selected_skill_id'],
            'selected_equal': before['selected_skill_id'] == after['selected_skill_id'],
            'max_abs_score_difference': max(abs(before['scores'][sid] - after['skill_router_scores'][sid])
                                            for sid in before['scores']),
            'old_observed_seconds': before['response']['latency_ms'] / 1000,
            'new_accounted_seconds': after['skill_router_api']['latency_ms'] / 1000})
    after_np = np.random.get_state()
    rng_equal = (py == random.getstate() and np.array_equal(np_state[1], after_np[1])
        and np_state[2:] == after_np[2:] and torch.equal(th, torch.get_rng_state()))
    before_replay = router.stats()
    replay = router.route_many([{'candidate_bundle': memory.retrieve(''), **r['visible_input']} for r in chosen])
    replay_ok = router.stats()['local_attempts'] == before_replay['local_attempts'] and all(
        r['skill_router_api']['cache_hit'] for r in replay)
    unchanged = file_hash(a.source_cache) == source_hash
    passed = rng_equal and replay_ok and unchanged and torch.get_num_threads() == 1 and all(
        d['selected_equal'] and d['max_abs_score_difference'] <= 1e-4 for d in differences)
    old_seconds = sum(d['old_observed_seconds'] for d in differences)
    report = {'status': 'PASS' if passed else 'REQUIRES_REVIEW', 'synthetic_or_replayed_visible_states_only': True,
        'not_an_alfworld_performance_result': True, 'sample_count': len(chosen),
        'selection_rule': 'source cache key order, first state cold warmup, next N states; no reward/label selection',
        'source_path': str(a.source_cache.resolve()), 'source_sha256': source_hash,
        'source_unchanged': unchanged, 'router_protocol_sha256': router.protocol_hash,
        'cold_load_index_and_one_query_seconds': cold_seconds, 'batch_query_wall_seconds': batch_seconds,
        'mean_wall_seconds_per_query': batch_seconds / len(chosen),
        'old_observed_serial_seconds_same_states': old_seconds,
        'observed_timing_ratio_not_guaranteed_pipeline_speedup': old_seconds / batch_seconds,
        'cpu_rng_restored': rng_equal, 'cpu_threads_restored': torch.get_num_threads() == 1,
        'cache_replay_zero_new_calls': replay_ok, 'differences': differences,
        'external_api_calls': 0, 'formal_rl_started': False,
        'remaining_gate': 'Full real rollout/optimizer/readout/storage and evaluation timing are not established by this component test'}
    write_new_json(a.output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'differences'}, indent=2), flush=True)
    router.close()
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
