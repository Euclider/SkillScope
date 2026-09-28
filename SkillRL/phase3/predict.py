"""Matched offline endpoint readout; four distributions exist only in RAM."""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

from .bank import Bank
from .common import digest, require, strict_json, write_new
from .readout import CompactReadout, WindowIdentity


def endpoint_forward_precision(start):
    """Preserve the sealed first-window backend; match native BF16 on resumes."""
    require(type(start) is int and start >= 0 and start % 5 == 0, 'Invalid prediction window start')
    if start == 0:
        return 'float32', 'matched-offline-FP32-weights-BF16-autocast-SDPA-full-vocabulary-v2'
    return 'bfloat16', 'matched-offline-BF16-weights-FP32-logsoftmax-SDPA-full-vocabulary-v3'


def predict(*, bank, old_path, new_path, identity, batch_path, output, calibration_path, parity_atol):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from phase2.measure import counter_input
    from skillnet_cohort.assets import LocalTokenizer, build_controls
    from skillnet_cohort.common import file_hash, exclusive_writer

    output, batch_path = Path(output), Path(batch_path)
    require(identity.bank_sha256 == bank.manifest_sha256 and identity.branch_id == bank.branch_id, 'Foreign prediction bank')
    precision, backend = endpoint_forward_precision(identity.start)
    metadata = strict_json(batch_path.with_suffix('.json').read_text())
    require(metadata['sha256'] == file_hash(batch_path), 'Changed direction batch')
    source = {'identity': asdict(identity), 'batch_sha256': metadata['sha256'],
              'backend': backend,
              'numerical_version': 'fp64_zero_sum_readout_v1',
              'score': 'shadow_only' if identity.branch_id == 'skillrl_failure' else 'all_registered_readouts',
              'decision_use': identity.branch_id != 'skillrl_failure', 'parity_atol': parity_atol}
    with exclusive_writer(output):
        write_new(output / 'source.json', source)
        if (output / 'complete.json').exists():
            complete = strict_json((output / 'complete.json').read_text())
            bundle = strict_json((output / 'readout.json').read_text())
            require(complete['readout_sha256'] == digest(bundle) and bundle['identity'] == asdict(identity), 'Changed prediction')
            return bundle
        torch.set_num_threads(4)
        batch = torch.load(batch_path, weights_only=False, map_location='cpu')  # Only our hash-verified local file.
        require(batch['global_update'] == identity.start + 1 and batch['bank_sha256'] == bank.manifest_sha256,
                'Wrong window direction batch')
        tensors = batch['tensors']
        tokenizer = AutoTokenizer.from_pretrained(old_path, local_files_only=True)
        controls = build_controls(bank, LocalTokenizer(old_path))
        write_new(output / 'controls.json', controls)
        rows, seen = [], set()
        for row, meta in enumerate(batch['metadata']):
            if meta['decision_id'] not in seen:
                seen.add(meta['decision_id'])
                require(meta['info']['bank_sha256'] == bank.manifest_sha256, 'Foreign captured bank')
                if tensors['actual_loss_mask'][row].bool().any():
                    rows.append((row, meta))
        require(rows, 'No valid start-batch decisions')
        rows.sort(key=lambda pair: pair[1]['decision_id'])
        started, forward_tokens, forward_calls = time.monotonic(), 0, 0
        require(torch.cuda.device_count() >= 2, 'Matched FP32 endpoint scoring requires two free GPUs')
        # U0->U5 is already sealed under the FP32-weight/autocast backend.
        # On resumed windows, the live FSDP forward uses BF16 parameters;
        # FP32-weight scoring fails captured OLD parity. Cast logits to FP32
        # before log_softmax, as in the actor's chosen-token computation.
        model_dtype = torch.float32 if precision == 'float32' else torch.bfloat16
        old = AutoModelForCausalLM.from_pretrained(old_path, dtype=model_dtype,
                    attn_implementation='sdpa', local_files_only=True).to('cuda:0').eval()
        new = AutoModelForCausalLM.from_pretrained(new_path, dtype=model_dtype,
                    attn_implementation='sdpa', local_files_only=True).to('cuda:1').eval()
        length, temperature = tensors['responses'].shape[-1], batch['temperature']

        @torch.inference_mode()
        def forward(model, ids, mask, positions, valid):
            nonlocal forward_calls, forward_tokens
            device = next(model.parameters()).device
            pos = positions.unsqueeze(0).to(device)
            if pos.ndim == 3:
                pos = pos.transpose(0, 1)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                result = model(input_ids=ids.unsqueeze(0).to(device), attention_mask=mask.unsqueeze(0).to(device),
                    position_ids=pos, use_cache=False, output_hidden_states=False,
                    logits_to_keep=length + 1, return_dict=True)
            logits = result.logits[0, :-1].float() / temperature
            lp = logits.index_select(0, valid.to(device)).log_softmax(-1).cpu()
            forward_calls += 1
            forward_tokens += int(mask.sum())
            return lp

        # Outcome-blind repeated same-checkpoint noise calibration; frozen on
        # the first block and never chosen to improve downstream labels.
        calibration_path = Path(calibration_path)
        if calibration_path.exists():
            calibration = strict_json(calibration_path.read_text())
            require(calibration['branch_id'] == bank.branch_id, 'Foreign calibration')
        else:
            require(identity.start == 0, 'First-window calibration is missing')
            noise = []
            for row, _ in rows[:8]:
                valid = tensors['actual_loss_mask'][row].bool().nonzero().flatten()
                args = [tensors[key][row] for key in ('input_ids', 'attention_mask', 'position_ids')]
                first, second = forward(old, *args, valid), forward(old, *args, valid)
                noise.extend((first - second).square().sum(-1, dtype=torch.float64).sqrt().tolist())
            p95 = float(torch.tensor(noise, dtype=torch.float64).quantile(.95))
            calibration = {'branch_id': bank.branch_id, 'old_policy_sha256': identity.old_policy_sha256,
                'batch_sha256': metadata['sha256'], 'method': '8 decisions x two same-checkpoint forwards; L2 p95; fixed before edit/gate',
                'tau_delta': max(1e-8, 10 * p95), 'noise_p95': p95, 'target_outcomes_read': False}
            write_new(calibration_path, calibration)
        aggregate = CompactReadout(identity, bank.active_versions, tau_delta=calibration['tau_delta'])
        parity_max = 0.
        for index, (row, meta) in enumerate(rows):
            info = meta['info']
            sid = info['selected_skill_id']
            require(info['skill_version_sha256'] == bank.active_versions.get(sid), 'Stale direction skill')
            require(info['phase3_payload_text'] == bank.get(sid).payload, 'Changed injected payload')
            valid = tensors['actual_loss_mask'][row].bool().nonzero().flatten()
            args = [tensors[key][row] for key in ('input_ids', 'attention_mask', 'position_ids')]
            o0, o1 = forward(old, *args, valid), forward(new, *args, valid)
            tokens = tensors['responses'][row, valid]
            chosen = o0.gather(1, tokens[:, None]).squeeze(1)
            error = float((chosen - tensors['old_log_probs'][row, valid]).abs().max())
            parity_max = max(parity_max, error)
            require(error <= parity_atol,
                    f'Offline/live chosen-token parity exceeds the predeclared tolerance '
                    f'at decision {index + 1}/{len(rows)}: error={error:.8g}, atol={parity_atol:.8g}')
            adapted = {**meta, 'info': {**info, 'phase2_payload_text': info['phase3_payload_text']}}
            counter = counter_input(tokenizer, adapted, tensors, row, 'placebo', controls['controls'][sid]['text'])
            # Exact whole-prompt matching, not merely equal isolated payload lengths.
            require(int(counter[1][:-length].sum()) == int(tensors['attention_mask'][row, :-length].sum()), 'PLACEBO full-prompt token mismatch')
            p0, p1 = forward(old, *counter, valid), forward(new, *counter, valid)
            aggregate.add(skill_id=sid, skill_version_sha256=info['skill_version_sha256'], decision_id=meta['decision_id'],
                game_id=info['extra.gamefile'], trajectory_id=meta['trajectory_id'], old_original=o0, new_original=o1,
                old_placebo=p0, new_placebo=p1, actions=tokens, advantages=tensors['advantages'][row, valid])
            if index % 25 == 0:
                print(f'Phase3 prediction {index + 1}/{len(rows)}', flush=True)
        bundle = aggregate.bundle()
        write_new(output / 'readout.json', bundle)
        write_new(output / 'complete.json', {'readout_sha256': digest(bundle), 'source_sha256': digest(source),
            'decisions': len(rows), 'forward_calls': forward_calls, 'forward_input_tokens': forward_tokens,
            'wall_seconds': time.monotonic() - started, 'chosen_logprob_max_abs_error': parity_max,
            'calibration_sha256': digest(calibration), 'full_vocab_saved': False, 'utility_gold_computed': False})
        return bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'old-path', 'new-path', 'batch', 'output', 'calibration', 'identity'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--bank-sha256', required=True)
    p.add_argument('--parity-atol', type=float, required=True)
    a = p.parse_args()
    predict(bank=Bank.load(a.bank, a.bank_sha256), old_path=a.old_path, new_path=a.new_path,
            identity=WindowIdentity(**strict_json(a.identity.read_text())), batch_path=a.batch,
            output=a.output, calibration_path=a.calibration, parity_atol=a.parity_atol)


if __name__ == '__main__':
    main()
