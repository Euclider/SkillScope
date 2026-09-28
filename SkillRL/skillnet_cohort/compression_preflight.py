"""Fixed, read-only input calibration of lossless full-vocabulary row storage.

No generation, environment rollout, optimizer step, checkpoint or paid API.
Natural cases teacher-force retokenized saved ALFWorld outputs on B0 with the
existing native readout forward. They are capacity samples, not live-logit parity
or performance evidence. Synthetic inputs additionally cover a post-update actor.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from .common import file_hash, read_json, write_new_bytes, write_new_json
from .lossless_tensor import bitwise_equal, encode, load, shuffled_profile


def natural_cases(root):
    cases = []
    for path in sorted(Path(root).glob('shards/*/results/*.json')):
        result = read_json(path)
        if result['job']['source_split'] != 'train':
            continue
        steps = result['result']['steps']
        for index in sorted({0, len(steps)//2, len(steps)-1}):
            cases.append((path, result['job']['job_id'], steps[index]))
    if len(cases) != 24:
        raise ValueError('Expected first/middle/last from each of eight fixed train trajectories')
    return cases


def measure(root, name, value, metadata):
    started = time.monotonic()
    encoded, audit = encode(value, shuffled_profile())
    encode_seconds = time.monotonic() - started
    path = root / 'rows' / (name + '.pt')
    started = time.monotonic()
    write_new_bytes(path, encoded)
    write_seconds = time.monotonic() - started
    started = time.monotonic()
    restored = load(path)
    if not bitwise_equal(value, restored):
        raise ValueError('On-disk compressed calibration did not preserve every tensor bit')
    row = {**metadata, **audit, 'path': str(path), 'tokens': len(value['log_probs']),
           'vocab_size': value['log_probs'].shape[-1], 'encode_and_verify_seconds': encode_seconds,
           'write_and_fsync_seconds': write_seconds, 'read_and_verify_seconds': time.monotonic()-started}
    write_new_json(root / 'records' / (name + '.json'), row)
    print(f"COMPRESSED {name} tokens={row['tokens']} ratio={len(encoded)/audit['plain_torch_save_bytes']:.4f}", flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'synthetic', 'natural', 'model'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        p.error('Explicit --execute required; no implicit GPU loading')
    root = a.output.resolve()
    if (root/'calibration-start.json').exists():
        raise FileExistsError('Do not restart a prior compression calibration')
    cases = natural_cases(a.natural)
    if read_json(a.synthetic/'verification-audit.json')['status'] != 'PASS':
        raise ValueError('Synthetic exact capture has not been audited')
    write_new_json(root/'calibration-start.json', {'compression': shuffled_profile(),
        'source_files': {str(path): file_hash(path) for path, _, _ in cases},
        'calibration_scope': 'engineering only, retokenized saved text; no original generation token IDs claim',
        'formal_rl_iterations': 0, 'optimizer_steps': 0, 'external_api_calls': 0})
    import torch
    from transformers import AutoTokenizer
    from phase2.measure import forward, load_model
    torch.set_num_threads(1)
    records = []
    for stage in ('old', 'new'):
        for rank in range(8):
            path = a.synthetic/f'{stage}_logprobs/u0001'/f'row-{16*rank:06d}.pt'
            value = torch.load(path, map_location='cpu', weights_only=False)
            records.append(measure(root, f'synthetic-{stage}-{rank}', value,
                {'kind': 'synthetic', 'stage': stage, 'source': str(path), 'source_sha256': file_hash(path)}))
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    started = time.monotonic()
    model = load_model(a.model)
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    for index, (source, job_id, step) in enumerate(cases):
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': step['prompt_text']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt = tokenizer.encode(rendered, add_special_tokens=False)
        response = tokenizer.encode(step['raw_model_output'], add_special_tokens=False) + [tokenizer.eos_token_id]
        if not 0 < len(prompt) <= 4096 or not 0 < len(response) <= 512:
            raise ValueError('Saved calibration case exceeds unchanged actor bounds')
        ids = torch.tensor(prompt + response, dtype=torch.long)
        positions, mask = torch.arange(len(ids)), torch.ones_like(ids)
        started = time.monotonic()
        lp, hidden = forward(model, ids, mask, positions, len(response), torch.arange(len(response)), 1.)
        torch.cuda.synchronize()
        forward_seconds = time.monotonic() - started
        lp = lp.cpu()
        if lp.dtype != torch.float32 or lp.shape != (len(response), 248320) or not torch.isfinite(lp).all():
            raise ValueError('Natural full-vocabulary forward has wrong shape/dtype or nonfinite values')
        tokens = torch.tensor(response, dtype=torch.long)
        chosen = lp.gather(1, tokens[:, None]).squeeze(-1)
        value = {'row_index': index, 'rank': 0, 'token_positions': torch.arange(len(tokens)),
                 'token_ids': tokens, 'log_probs': lp, 'chosen_log_probs': chosen,
                 'trainer_chosen_log_probs': chosen.clone()}
        records.append(measure(root, f'natural-{index:02d}', value,
            {'kind': 'natural_retokenized', 'job_id': job_id, 'source': str(source),
             'source_sha256': file_hash(source), 'step_index': step['step_index'],
             'prompt_tokens': len(prompt), 'forward_with_all_hidden_states_seconds': forward_seconds,
             'not_a_live_trainer_capture': True}))
        del hidden
    write_new_json(root/'complete.json', {'status': 'PASS', 'compression': shuffled_profile(),
        'synthetic_rows': 16, 'natural_rows': 24, 'records': records, 'model_load_seconds': load_seconds,
        'formal_rl_iterations': 0, 'external_api_calls': 0,
        'not_verified': ['future training token lengths', 'future compressed row sizes',
                         'natural live/replay numerical parity', 'whole-pipeline time/storage admission']})


if __name__ == '__main__':
    main()
