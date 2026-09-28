"""Validate endpoint replay against the actual start batch, without target labels."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phase1.archive import sha256_file
from phase2.protocol import controls_by_skill, signal_directory, validate_extended
from skillnet_cohort.common import REPO, read_json


def validate_shards(root, start, end, shards=8):
    root = Path(root)
    config = read_json(root/'protocol.json')
    validate_extended(config, REPO)
    if (config.get('capture_scope') != 'window_start_old_only_v1'
            or config.get('window_direction') != 'start_batch_endpoint_projection'
            or not any(w['start'] == start and w['end'] == end for w in config['windows'])
            or shards != config['evaluation']['shards']):
        raise ValueError('Endpoint replay must follow the frozen window protocol')
    batch_root = root/'batches'/f'u{start+1:04d}'
    manifest = read_json(batch_root/'manifest.json')
    if sha256_file(batch_root/'training_batch.pt') != manifest['batch_sha256']:
        raise ValueError('Actual training batch hash changed')
    expected_old = {f'row-{r:06d}.pt' for r in range(manifest['row_count'])}
    if {p.name for p in (root/'old_logprobs'/f'u{start+1:04d}').glob('row-*.pt')} != expected_old:
        raise ValueError('Missing or extra start OLD rows')
    batch = torch.load(batch_root/'training_batch.pt', map_location='cpu', weights_only=False)
    b = batch['tensors']
    controls = controls_by_skill(config, REPO)
    seen, rows = set(), []
    for row, meta in enumerate(batch['metadata']):
        if meta['decision_id'] in seen:
            continue
        seen.add(meta['decision_id'])
        if meta['info'].get('selected_skill_id') in controls:
            rows.append((row, meta))
    out = signal_directory(root, end, start)
    counts = []
    for shard in range(shards):
        selected = rows[shard::shards]
        m = read_json(out/f'shard-{shard}.json')
        if (m.get('shard') != shard or m.get('update') != end or m.get('start_update') != start
                or m.get('direction_batch_update') != start+1 or m.get('max_decisions') is not None
                or m.get('readout_kind') != 'start_batch_endpoint_projection'
                or m.get('decisions') != len(selected)
                or sha256_file(out/f'tokens-shard-{shard}.parquet') != m['tokens_sha256']):
            raise ValueError('Changed, partial or wrong-window measurement shard')
        tokens = pd.read_parquet(out/f'tokens-shard-{shard}.parquet')
        decisions = pd.read_parquet(out/f'decisions-shard-{shard}.parquet')
        expected = {}
        for row, meta in selected:
            offsets = b['phase2_actual_loss_mask'][row].bool().nonzero().flatten().tolist()
            for control in ('placebo', 'null'):
                expected[(meta['decision_id'], control)] = (row, meta, offsets)
        dk = ['decision_id', 'control']
        if (decisions.duplicated(dk).any() or tokens.duplicated(dk+['response_token_offset']).any()
                or set(map(tuple, decisions[dk].to_numpy())) != set(expected)
                or set(map(tuple, tokens[dk].drop_duplicates().to_numpy())) != set(expected)
                or len(tokens) != m['tokens_with_controls']):
            raise ValueError('Missing, extra or duplicate readout decision/control/token')
        indexed = decisions.set_index(dk)
        for identity, frame in tokens.groupby(dk, sort=False):
            row, meta, offsets = expected[identity]
            frame = frame.sort_values('response_token_offset')
            if frame.response_token_offset.tolist() != offsets or indexed.loc[identity, 'token_count'] != len(offsets):
                raise ValueError('Readout does not cover the actual loss-mask positions')
            checks = {'global_update': end, 'start_update': start, 'direction_batch_update': start+1,
                'window_horizon': end-start, 'row_index': row, 'trajectory_id': meta['trajectory_id'],
                'game_id': meta['info']['extra.gamefile'], 'skill_id': meta['info']['selected_skill_id']}
            for key, value in checks.items():
                if not frame[key].eq(value).all() or indexed.loc[identity, key] != value:
                    raise ValueError('Readout metadata does not match the actual training batch: '+key)
            if (not np.array_equal(frame.action_token_id.to_numpy(), b['responses'][row, offsets].numpy())
                    or not np.array_equal(frame.advantage.to_numpy(), b['advantages'][row, offsets].double().numpy())):
                raise ValueError('Readout actions/advantages differ from the actual update batch')
        noise = np.asarray(m['repeat_forward_noise'], dtype=float)
        if not np.isfinite(noise).all() or (noise < 0).any():
            raise ValueError('Invalid label-free calibration noise')
        if selected:
            witness = torch.load(out/f'witness-shard-{shard}.pt', map_location='cpu', weights_only=False)
            row, meta = selected[0]
            required = {'old_original_live', 'end_original_replay', 'old_placebo', 'new_placebo', 'old_null', 'new_null'}
            if (not required.issubset(witness) or 'new_original_live' in witness
                    or witness['metadata']['decision_id'] != meta['decision_id']
                    or not torch.equal(witness['token_ids'], b['responses'][row, witness['positions']])
                    or not torch.equal(witness['advantage'], b['advantages'][row, witness['positions']])):
                raise ValueError('Wrong endpoint-replay witness')
        counts.append({'shard': shard, 'decisions': len(selected), 'tokens_with_controls': len(tokens)})
    return {'status': 'PASS', 'batch_sha256': manifest['batch_sha256'],
            'live_start_batch_rows': manifest['row_count'], 'eligible_decisions': len(rows),
            'shards': counts, 'new_live_rows_required': False,
            'endpoint_original_source': 'end_original_replay', 'target_gold_read': False}
