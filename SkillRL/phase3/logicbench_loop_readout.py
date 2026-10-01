"""Compact question-equal diagnosis and v13 failed-invocation candidates."""
from pathlib import Path
import time

from .common import digest, require, strict_json, write_new
from .logicbench import validate_evidence

FIELDS = ('D_sign_balance', 'D_signed_gate', 'D_real', 'D_original', 'D_signed',
          'C_upd_centered', 'M_delta_centered', 'M_delta_raw', 'P_int')


def aggregate_tokens(records, bank):
    import pandas as pd
    import numpy as np
    require(records, 'No naturally observed loss tokens')
    frame = pd.DataFrame(records)
    require(np.isfinite(frame[list(FIELDS)].to_numpy(dtype=float)).all(), 'Non-finite compact readout')
    require(set(frame.skill_id) <= set(bank.skill_ids), 'Unknown token skill')
    means = (frame.groupby(['skill_id', 'question_id', 'trajectory_id'])[list(FIELDS)].mean()
             .groupby(['skill_id', 'question_id']).mean().groupby('skill_id').mean())
    out = []
    for skill in bank.skills:
        group = frame[frame.skill_id == skill.skill_id]
        out.append({'skill_id': skill.skill_id, 'skill_version_sha256': skill.version_sha256,
            'supported': len(group) > 0, 'token_count': len(group),
            'n_questions': int(group.question_id.nunique()), 'n_trajectories': int(group.trajectory_id.nunique()),
            **{name: float(means.loc[skill.skill_id, name]) if len(group) else None for name in FIELDS}})
    return out


def v13_payload(bank, evidence, scores, *, k=5):
    validate_evidence(bank, evidence)
    require(type(k) is int and 1 <= k <= 5, 'Invalid v13 candidate budget')
    require(len(scores) == len({r['skill_id'] for r in scores}), 'Duplicate readout skill')
    failed = [row for row in evidence if not row['success']]
    invoked = {row['selected_skill_id'] for row in failed}
    eligible = []
    for row in scores:
        require(row['skill_id'] in bank.skill_ids and row['skill_version_sha256'] == bank.get(row['skill_id']).version_sha256,
                'Stale readout version')
        if row['supported'] and row['skill_id'] in invoked:
            from .common import finite
            finite(row['D_sign_balance'], 'sign balance')
            eligible.append(row)
    priority = [r['skill_id'] for r in sorted(eligible, key=lambda r: (-r['D_sign_balance'], r['skill_id']))[:k]]
    if not priority:
        return None, []
    keys = ('evidence_id', 'question_id', 'context_id', 'sampling_policy_update', 'selected_skill_id',
            'skill_version_sha256', 'question', 'task_type', 'response', 'success')
    rows = [{key: r[key] for key in keys} for r in sorted(failed, key=lambda r: r['evidence_id'])
            if r['selected_skill_id'] in priority]
    return {'environment': 'LogicBench single-answer logical reasoning', 'mutation_budget': 3,
        'active_bank_size': len(bank), 'priority_targets': priority,
        'candidate_skills': [{'skill_id': s.skill_id, 'version_sha256': s.version_sha256,
                             'name': s.name, 'description': s.description, 'body': s.payload}
                            for s in (bank.get(sid) for sid in priority)],
        'evidence': rows}, priority


def predict(bank, old_path, new_path, batch_path, output, *, start, end, tau_delta=1e-8):
    """Reuse initial actual actions/advantages. No utility rollout and no vocab saved."""
    import torch
    from transformers import AutoTokenizer
    from phase1.logicbench_single_step import LogicBenchQuestion, build_prompt
    from logicbench_phase12.fixed_state import _load_model, score_recorded_tokens
    from phase2.stable_direction import token_signals
    from skillnet_cohort.common import file_hash
    from .run import model_identity
    batch_path, output = Path(batch_path), Path(output)
    metadata = strict_json(batch_path.with_suffix('.json').read_text())
    require(metadata['sha256'] == file_hash(batch_path), 'Changed direction batch')
    batch = torch.load(batch_path, map_location='cpu', weights_only=False)
    require(batch['bank_sha256'] == bank.manifest_sha256 and batch['global_update'] == start + 1
            and end == start + 5, 'Wrong readout window/bank')
    require(batch['schema_version'] == 'skillrl.phase3.logicbench.direction.v1' and batch['temperature'] == 1.,
            'Unsupported actual direction batch or training temperature')
    validate_evidence(bank, batch['evidence'])
    source = {'bank_sha256': bank.manifest_sha256, 'start': start, 'end': end,
              'batch_sha256': metadata['sha256'], 'old_model': str(old_path), 'new_model': str(new_path),
              'old_model_sha256': model_identity(old_path), 'new_model_sha256': model_identity(new_path),
              'control': 'remove_skill_segment', 'aggregation': 'token_answer_question_equal',
              'backend': 'same_hf_bfloat16_sdpa', 'tau_delta': tau_delta, 'eval_labels_used': False}
    write_new(output / 'source.json', source)
    if (output / 'complete.json').exists():
        record = strict_json((output / 'complete.json').read_text())
        rows = strict_json((output / 'readout.json').read_text())
        require(record['source_sha256'] == digest(source) and record['readout_sha256'] == digest(rows), 'Changed readout')
        return rows
    require(torch.cuda.device_count() >= 2, 'Readout needs two visible GPUs')
    tokenizer = AutoTokenizer.from_pretrained(old_path, local_files_only=True)
    old, new = _load_model(old_path, 'cuda:0'), _load_model(new_path, 'cuda:1')
    tensors, records, started = batch['tensors'], [], time.monotonic()
    forward_calls = forward_tokens = 0
    parity_max = 0.
    try:
        for i, item in enumerate(batch['evidence']):
            mask = tensors['actual_loss_mask'][i].bool()
            if not mask.any():
                continue
            actions = tensors['responses'][i, mask].tolist()
            prompt_ids = tensors['prompts'][i, tensors['prompt_mask'][i].bool()].tolist()
            question = LogicBenchQuestion(item['question_id'], item['question'], item['task_type'], '', '')
            text = tokenizer.apply_chat_template([{'role':'user', 'content':build_prompt(question, None)}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False)
            control = tokenizer(text, add_special_tokens=False)['input_ids']
            skill_old = score_recorded_tokens(old, prompt_ids, actions, device='cuda:0')
            skill_new = score_recorded_tokens(new, prompt_ids, actions, device='cuda:1').to('cuda:0')
            control_old = score_recorded_tokens(old, control, actions, device='cuda:0')
            control_new = score_recorded_tokens(new, control, actions, device='cuda:1').to('cuda:0')
            advantages = tensors['advantages'][i, mask].to('cuda:0')
            ids = torch.tensor(actions, device='cuda:0')
            signals = token_signals(skill_old, skill_new, control_old, control_new, ids, advantages,
                                   tau_delta=tau_delta, tau_c=0., epsilon=1e-12)
            chosen = skill_old.gather(1, ids[:, None]).flatten()
            parity_max = max(parity_max, float((chosen - tensors['old_log_probs'][i, mask].to('cuda:0')).abs().max()))
            u = skill_new.double() - skill_old.double()
            delta = u - (control_new.double() - control_old.double())
            uc, dc = u - u.mean(-1, keepdim=True), delta - delta.mean(-1, keepdim=True)
            real = -advantages.double() * u.gather(1, ids[:, None]).flatten() * (uc * dc).sum(-1) / (uc.square().sum(-1) + 1e-12)
            for j in range(len(actions)):
                p = float(signals['P_int'][j])
                records.append({'skill_id': item['selected_skill_id'], 'question_id': item['question_id'],
                    'trajectory_id': item['evidence_id'], 'decision_id': item['evidence_id'] + ':s0',
                    'D_sign_balance': -float(torch.sign(signals['P_int'][j])) if bool(signals['direction_valid'][j]) else 0.,
                    'D_signed_gate': -p * bool(signals['gate'][j]), 'D_original': float(signals['D_contribution'][j]),
                    'D_real': float(real[j]), 'D_signed': -p, 'P_int': p,
                    'C_upd_centered': float(signals['C_upd_centered'][j]),
                    'M_delta_centered': float(signals['delta_centered_norm'][j]), 'M_delta_raw': float(signals['delta_norm'][j])})
            forward_calls += 4
            forward_tokens += 2 * (len(prompt_ids) + len(control) + 2 * (len(actions) - 1))
            if (i + 1) % 32 == 0:
                print(f'LogicBench readout {i+1}/{len(batch["evidence"])}', flush=True)
        rows = aggregate_tokens(records, bank)
        write_new(output / 'readout.json', rows)
        write_new(output / 'complete.json', {'source_sha256':digest(source), 'readout_sha256':digest(rows),
            'forward_calls':forward_calls, 'forward_input_tokens':forward_tokens, 'seconds':time.monotonic()-started,
            'native_vs_hf_chosen_max_abs_error':parity_max, 'native_parity_is_diagnostic_only':True,
            'four_conditions_use_same_hf_backend':True, 'full_vocab_saved':False, 'utility_gold_computed':False})
        return rows
    finally:
        del old, new
        torch.cuda.empty_cache()
