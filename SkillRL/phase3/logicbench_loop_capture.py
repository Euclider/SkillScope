"""Bind actual single-answer GRPO evidence to its frozen window bank."""
import io
from collections import Counter
from pathlib import Path

from .bank import Bank
from .common import require, strict_json, write_new
from .logicbench_loop_data import DATA
from skillnet_cohort.common import file_hash, write_new_bytes


def archive_batch(batch, *, update, config, tokenizer):
    import pandas as pd
    import torch
    from phase1.logicbench_single_step import LogicBenchQuestion, build_prompt, extract_final_answer
    cfg = config.phase3
    bank = Bank.load(cfg.bank_path, cfg.bank_sha256)
    root = Path(cfg.root)
    first = update == cfg.segment_start + 1
    target = root / 'direction_batches' / f'u{update:04d}.pt'
    require(not first or not target.exists(), 'Refusing to overwrite captured direction batch')
    questions = {q['question_id']:q for q in strict_json((DATA/'train.json').read_text())}
    source = pd.read_parquet(config.data.train_files).to_dict('records')
    routed = {q['question_id']:q for q in source}
    tensors = batch.batch
    length = tensors['responses'].shape[-1]
    require(not config.actor_rollout_ref.rollout.multi_turn.enable, 'LogicBench requires single-answer loss')
    mask = tensors['attention_mask'][:, -length:].detach().cpu().bool()
    require(torch.equal(mask, tensors['response_mask'].detach().cpu().bool()), 'Response/loss mask mismatch')
    advantages = tensors['advantages'].detach().cpu()
    require(mask.shape == advantages.shape and torch.isfinite(advantages[mask]).all().item(), 'Invalid actual advantages')
    evidence, prompt_cache = [], {}
    prompt_mask = tensors['attention_mask'][:, :tensors['prompts'].shape[-1]].detach().cpu().bool()
    for i, raw in enumerate(batch.non_tensor_batch['phase2_metadata']):
        meta = strict_json(str(raw))
        qid = str(batch.non_tensor_batch['question_id'][i])
        sid = str(batch.non_tensor_batch['selected_skill_id'][i])
        q, route = questions[qid], routed[qid]
        skill = bank.get(sid)
        require(route['selected_skill_id'] == sid and route['bank_sha256'] == bank.manifest_sha256
                and route['skill_version_sha256'] == skill.version_sha256, 'Foreign training route/version')
        require(meta['global_update'] == update and meta['environment_step'] == 0
                and meta['info']['question_id'] == qid and meta['info']['selected_skill_id'] == sid,
                'Foreign training response identity')
        if qid not in prompt_cache:
            question = LogicBenchQuestion(qid,q['question'],q['task_type'],q['answer'],'')
            chat = tokenizer.apply_chat_template([{'role':'user','content':build_prompt(question,skill.payload)}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False)
            prompt_cache[qid] = tokenizer(chat,add_special_tokens=False)['input_ids']
        require(tensors['prompts'][i].detach().cpu()[prompt_mask[i]].tolist() == prompt_cache[qid],
                'Actual prompt differs from current-bank prompt')
        response = tokenizer.decode(tensors['responses'][i].detach().cpu()[mask[i]],skip_special_tokens=True)
        parsed = extract_final_answer(response,q['task_type'])
        success = parsed == q['answer']
        require(float(success) == float(batch.non_tensor_batch['episode_rewards'][i]) == float(meta['info']['reward']),
                'Actual answer/reward mismatch')
        evidence.append({'evidence_id':meta['trajectory_id'],'question_id':qid,'context_id':q['context_id'],
            'split':'train','sampling_policy_update':update-1,'selected_skill_id':sid,
            'skill_version_sha256':skill.version_sha256,'question':q['question'],'task_type':q['task_type'],
            'response':response,'success':success,'format_valid':parsed is not None,
            'prompt_tokens':int(prompt_mask[i].sum()),'response_tokens':int(mask[i].sum())})
    counts = Counter(e['question_id'] for e in evidence)
    require(len(counts) == cfg.questions_per_update and set(counts.values()) == {cfg.repeats}, 'Wrong GRPO groups')
    require(len({e['evidence_id'] for e in evidence}) == len(evidence), 'Duplicate training trajectory')
    write_new(root/'episodes'/f'u{update:04d}'/'logicbench.json', {
        'bank_sha256':bank.manifest_sha256,'global_update':update,'evidence':evidence})
    if not first:
        return
    copied = {k:tensors[k].detach().cpu().clone() for k in ('prompts','responses','advantages','old_log_probs')}
    copied.update(prompt_mask=prompt_mask,actual_loss_mask=mask)
    value = {'schema_version':'skillrl.phase3.logicbench.direction.v1','tensors':copied,'evidence':evidence,
        'bank_sha256':bank.manifest_sha256,'global_update':update,'branch_id':bank.branch_id,
        'temperature':float(batch.meta_info['temperature'])}
    buffer = io.BytesIO()
    torch.save(value,buffer)
    write_new_bytes(target,buffer.getvalue())
    write_new(target.with_suffix('.json'), {'sha256':file_hash(target),'row_count':len(evidence),
        'loss_tokens':int(mask.sum()),'bank_sha256':bank.manifest_sha256,
        'capture_point':'actual GRPO advantages before optimizer update','full_vocab_saved':False})
