"""Coverage-only readout expansion; the actual training decisions are unchanged."""
import argparse
from pathlib import Path
import json

import pandas as pd
import torch

from .common import REPO, file_hash, read_json, require_authorization, write_new_bytes, write_new_json
from phase2.measure import counter_input, forward, load_model, phase
from phase2.direction import token_signals
from phase2.protocol import controls_by_skill, signal_directory, validate_extended


def score_decision(models, tokenizer, tensors, row, meta, control, temperature, repeat=False):
    from .lossless_tensor import load
    root, old_model, new_model = models
    b = tensors
    old = load(root/'old_logprobs/u0001'/f'row-{row:06d}.pt')
    use = b['phase2_actual_loss_mask'][row, old['token_positions']].bool()
    positions, ids = old['token_positions'][use], old['token_ids'][use]
    if not torch.equal(ids, b['responses'][row, positions]):
        raise ValueError('Actual optimizer action tokens changed')
    advantage = b['advantages'][row, positions]
    old_o = old['log_probs'][use].cuda(); length = b['responses'].shape[-1]
    offline_old, ho = forward(old_model, b['input_ids'][row], b['attention_mask'][row], b['position_ids'][row], length, positions, temperature)
    new_o, hn = forward(new_model, b['input_ids'][row], b['attention_mask'][row], b['position_ids'][row], length, positions, temperature)
    base = {'global_update': 5, 'start_update': 0, 'direction_batch_update': 1, 'window_horizon': 5,
        'row_index': row, 'decision_id': meta['decision_id'], 'trajectory_id': meta['trajectory_id'],
        'group_id': meta['group_id'], 'game_id': meta['info']['extra.gamefile'],
        'skill_id': meta['info']['selected_skill_id'], 'environment_step': meta['environment_step'],
        'phase': phase(meta['environment_step']), 'context_id': 'all_alfworld'}
    parity = {'old_live_offline_max_abs': float((old_o-offline_old).abs().max()),
        'new_live_offline_max_abs': float('nan'), 'new_chosen_max_abs': float('nan'),
        'old_chosen_max_abs': float((old_o-offline_old).gather(1, ids[:, None].cuda()).abs().max())}
    witness = {'metadata': base, 'positions': positions, 'token_ids': ids, 'advantage': advantage,
        'old_original_live': old_o.cpu(), 'end_original_replay': new_o.cpu()}
    tokens, decisions, noise = [], [], []
    for arm in ('placebo', 'null'):
        ci, cm, cp = counter_input(tokenizer, meta, b, row, arm, control['text'])
        old_c, hco = forward(old_model, ci, cm, cp, length, positions, temperature)
        new_c, hcn = forward(new_model, ci, cm, cp, length, positions, temperature)
        if repeat:
            repeated, _ = forward(old_model, ci, cm, cp, length, positions, temperature)
            noise.extend((repeated-old_c).norm(dim=-1).cpu().tolist())
        values = token_signals(old_o, new_o, old_c, new_c, ids.cuda(), advantage.cuda())
        same = token_signals(offline_old, new_o, old_c, new_c, ids.cuda(), advantage.cuda())
        for target, origin in (('P_int_matched_backend', 'P_int'), ('D_matched_backend', 'D_contribution'),
                ('C_upd_matched_backend', 'C_upd'), ('delta_norm_matched_backend', 'delta_norm')):
            values[target] = same[origin]
        for i, position in enumerate(positions.tolist()):
            tokens.append({**base, 'control': arm, 'response_token_offset': position,
                'action_token_id': int(ids[i]), **{k: v[i].item() for k, v in values.items()}})
        decisions.append({**base, 'control': arm, 'token_count': len(positions), 'advantage_mean': float(advantage.mean()),
            **parity, **{f'activation_l{layer}_norm': float(((hn[layer]-ho[layer])-(hcn[layer]-hco[layer])).norm(dim=-1).mean()) for layer in ho}})
        witness[f'old_{arm}'], witness[f'new_{arm}'] = old_c.cpu(), new_c.cpu()
    return pd.DataFrame(tokens), pd.DataFrame(decisions), witness, noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args(); root = args.root.resolve(); rank = args.shard
    if not 0 <= rank < 8:
        raise ValueError('Eight registered logical/GPU shards required')
    torch.set_num_threads(1)
    config = read_json(root/'protocol.json'); validate_extended(config, REPO)
    require_authorization(config['runtime']['authorization_path'], config['runtime']['preparation'], 'readout')
    if list((root/'evaluations/u0005').glob('shard-*.jsonl')):
        raise ValueError('Do not compute new readout after importing/opening this variant target gold')
    out = signal_directory(root, 5, 0); out.mkdir(parents=True, exist_ok=True)
    if (out/f'shard-{rank}.json').exists():
        raise FileExistsError('No automatic measurement retry')
    reuse = read_json(root/'readout-reuse.json')
    for item in reuse['files']:
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Retained readout source changed')
    batch = torch.load(root/'batches/u0001/training_batch.pt', map_location='cpu', weights_only=False)
    if file_hash(root/'batches/u0001/training_batch.pt') != reuse['training_batch_sha256']:
        raise ValueError('Readout reuse belongs to another actual batch')
    controls = controls_by_skill(config, REPO)
    rows, seen = [], set()
    for row, meta in enumerate(batch['metadata']):
        if meta['decision_id'] in seen:
            continue
        seen.add(meta['decision_id'])
        if meta['info'].get('selected_skill_id') in controls:
            rows.append((row, meta))
    selected = rows[rank::8]
    source = Path(reuse['signal_directory'])
    cached_t = pd.concat([pd.read_parquet(source/f'tokens-shard-{r}.parquet') for r in range(8)], ignore_index=True)
    cached_d = pd.concat([pd.read_parquet(source/f'decisions-shard-{r}.parquet') for r in range(8)], ignore_index=True)
    tokens_by_id = {key: value for key, value in cached_t.groupby('decision_id', sort=False)}
    decisions_by_id = {key: value for key, value in cached_d.groupby('decision_id', sort=False)}
    witnesses = {}
    for r in range(8):
        path = source/f'witness-shard-{r}.pt'
        if path.exists():
            witness = torch.load(path, map_location='cpu', weights_only=False)
            witnesses[witness['metadata']['decision_id']] = path
    frames, decision_frames, noise = [], [], []
    old_noise = [v for r in range(8) for v in read_json(source/f'shard-{r}.json')['repeat_forward_noise']]
    models = tokenizer = None; reused = computed = witness_only = 0
    for i, (row, meta) in enumerate(selected):
        key = meta['decision_id']; cached = key in tokens_by_id
        need_witness = i == 0 and key not in witnesses
        if not cached or need_witness:
            if models is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(root/'models/u0000', local_files_only=True)
                models = root, load_model(root/'models/u0000'), load_model(root/'models/u0005')
            fresh_t, fresh_d, witness, repeated_noise = score_decision(models, tokenizer, batch['tensors'], row, meta,
                controls[meta['info']['selected_skill_id']], float(batch['meta_info']['temperature']), repeat=i < 2)
            noise.extend(repeated_noise)
            if cached:
                # Only a missing partition witness may require a repeat forward;
                # keep the original exact scores and expose this count explicitly.
                witness_only += 1
                keys = ['decision_id', 'control', 'response_token_offset']
                left = tokens_by_id[key].sort_values(keys).reset_index(drop=True)
                right = fresh_t.sort_values(keys).reset_index(drop=True)
                if not left.equals(right[left.columns]):
                    raise ValueError('Repeated witness differs from retained exact decision signals')
            else:
                computed += 1
        if cached:
            frames.append(tokens_by_id[key]); decision_frames.append(decisions_by_id[key]); reused += 1
        else:
            frames.append(fresh_t); decision_frames.append(fresh_d)
        if i == 0:
            if key in witnesses:
                write_new_bytes(out/f'witness-shard-{rank}.pt', witnesses[key].read_bytes())
            else:
                from phase2.capture import save_tensor_file
                save_tensor_file(out/f'witness-shard-{rank}.pt', witness)
        print(f'shard{rank} {i+1}/{len(selected)} reused={reused} missing_scored={computed}', flush=True)
    tokens = pd.concat(frames, ignore_index=True) if frames else cached_t.iloc[:0].copy()
    decisions = pd.concat(decision_frames, ignore_index=True) if decision_frames else cached_d.iloc[:0].copy()
    write_new_bytes(out/f'tokens-shard-{rank}.parquet', tokens.to_parquet(index=False))
    write_new_bytes(out/f'decisions-shard-{rank}.parquet', decisions.to_parquet(index=False))
    write_new_json(out/f'shard-{rank}.json', {'update': 5, 'start_update': 0, 'direction_batch_update': 1,
        'shard': rank, 'decisions': len(selected), 'tokens_with_controls': len(tokens), 'max_decisions': None,
        'readout_kind': 'start_batch_endpoint_projection', 'tokens_sha256': file_hash(out/f'tokens-shard-{rank}.parquet'),
        'repeat_forward_noise': noise or old_noise, 'reused_exact_decisions': reused,
        'newly_scored_decisions': computed, 'witness_only_repeated_decisions': witness_only,
        'calibration_preserved_from_original_label_free_start_batch': True, 'target_gold_read': False})


if __name__ == '__main__':
    main()
