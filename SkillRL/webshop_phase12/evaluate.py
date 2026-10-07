"""U0 natural-call anchors and paired U0/U5 skill/control continuations."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from logicbench_phase12.fixed_state import _load_model
from webshop_phase12.assets import BASE_MODEL, RUN_ROOT, WebshopBank, digest
from webshop_phase12.envs import ShopWorld
from webshop_phase12.prompts import action_list, build_state_prompt, project_action, policy_inputs
from webshop_phase12.visible_state import replay_visible
from webshop_phase12.router import RouterClient
from webshop_phase12.accelerated import BUCKET, ACTIVE, DENSE, load_forward_contract, prompt_multiple
from webshop_phase12.pairing import anchor_partition, merge_paired_rows


def model_receipt_hash():
    return hashlib.sha256((RUN_ROOT/'frozen-router-model.json').read_bytes()).hexdigest()


def finish(out, results, spec, seeds, contract, smoke):
    groups = defaultdict(list)
    for row in results:
        groups[row['skill_id']].append(row)
    utility = [{'skill_id': sid, 'n_eval_questions': len(rows),
                'n_eval_tasks': len({r['task_id'] for r in rows}),
                **{k: float(np.mean([r[k] for r in rows])) for k in ('m_old', 'm_new', 'delta_m')}}
               for sid, rows in sorted(groups.items())]
    (out/'skill_utility.json').write_text(json.dumps(utility, indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({
        'schema_version': 'skillscope.webshop_paired_utility.v1', 'reference_tasks': len(spec['eval_ids']),
        'eval_task_ids': spec['eval_ids'], 'anchors': len(results), 'continuation_seeds': list(seeds),
        'bank_frozen': True, 'same_prefix_replay_verified': True, 'same_visible_memory_replay_verified': True,
        'router_frozen': True, 'state_version': 'webshop-visible-evidence-v1',
        'old_model_receipt_sha256': model_receipt_hash(), 'bank_sha256': WebshopBank().content_sha256,
        'dense_prompt_width': 16384 if contract == DENSE else None,
        'same_prompt_padding_budget_as_training': True, 'prompt_budget': 16384, 'prompt_storage_width': 16384,
        'forward_contract': contract,
        'generation_padding': 'bucket256-shared-skill-control' if contract in (BUCKET, ACTIVE)
                              else ('dense-16384' if contract == DENSE else 'exact-valid-length'),
        'estimand': 'success_skill-success_control; delta_M=M_new-M_old',
        'control': 'target payload only, selected ID and candidate pool retained',
        'gold_used_by_readout': False, 'smoke': smoke}, indent=2)+'\n')


def run(seed_dir, new_model, prepared, *, smoke=False, stage='full', shard=0, shards=1, reuse_old_from=None):
    if not (seed_dir/'readout/prediction-locked.json').is_file():
        raise ValueError('Readout must be locked before collecting independent outcomes')
    contract = load_forward_contract(seed_dir)
    prediction = json.loads((seed_dir/'readout/prediction-locked.json').read_text())
    if prediction.get('forward_contract', DENSE) != contract:
        raise ValueError('Readout and paired evaluation forward contracts differ')
    spec = json.loads((prepared/'manifest.json').read_text())
    max_steps = 6 if smoke else 50
    seeds = (0, 1) if smoke else tuple(spec.get('continuation_seeds', range(16)))
    out = seed_dir/'paired_eval'
    if stage == 'merge':
        anchors = json.loads((out/'anchors.json').read_text())
        paths = [out/'shards'/f'shard-{i:04d}.jsonl' for i in range(shards)]
        for i in range(shards):
            receipt = json.loads((out/'shards'/f'shard-{i:04d}-complete.json').read_text())
            if receipt['forward_contract'] != contract or receipt['anchors_sha256'] != digest(anchors):
                raise ValueError('Shard protocol differs')
        results = merge_paired_rows(paths, anchors, seeds)
        with (out/'paired_anchors.jsonl').open('x') as stream:
            for row in results:
                stream.write(json.dumps(row)+'\n')
        finish(out, results, spec, seeds, contract, smoke)
        return
    if stage in ('full', 'reference'):
        out.mkdir(exist_ok=False)
    elif stage != 'continuations' or not out.is_dir():
        raise ValueError('Continuation stage requires completed reference anchors')
    bank = WebshopBank(); world = ShopWorld(1)
    router = RouterClient(f'eval-{seed_dir.name}-shard{shard}')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    old_cache = None
    if reuse_old_from is not None:
        origin = Path(reuse_old_from)
        manifest = json.loads((origin/'manifest.json').read_text())
        if (manifest['forward_contract'] != contract or manifest['continuation_seeds'] != list(seeds)
                or manifest['old_model_receipt_sha256'] != model_receipt_hash()
                or manifest['bank_sha256'] != bank.content_sha256
                or not all(manifest[k] for k in ('bank_frozen', 'router_frozen', 'same_prefix_replay_verified',
                                                'same_visible_memory_replay_verified'))
                or (origin/'anchors.json').read_bytes() != (out/'anchors.json').read_bytes()):
            raise ValueError('Reusable U0 outcomes have different anchors, bank, model or sampling')
        anchors = json.loads((out/'anchors.json').read_text())
        old_cache = merge_paired_rows([origin/'paired_anchors.jsonl'], anchors, seeds)
    models = {}
    if old_cache is None or stage in ('full', 'reference'):
        models['old'] = _load_model(BASE_MODEL, 'cuda')
    if stage != 'reference':
        models['new'] = _load_model(new_model, 'cuda')

    @torch.inference_mode()
    def generate(model, prompt, seed, reference_prompt=None):
        inputs = policy_inputs(tokenizer, prompt, device='cuda', compact=contract != DENSE,
                               prompt_multiple=prompt_multiple(contract), reference_prompt=reference_prompt)
        length = inputs['input_ids'].shape[-1]
        if length > 16384:
            raise ValueError('Paired continuation prompt exceeds registered budget')
        torch.manual_seed(seed)
        ids = model.generate(**inputs, do_sample=True, temperature=.7, top_p=1., max_new_tokens=512,
                             pad_token_id=tokenizer.eos_token_id)[0, length:]
        return tokenizer.decode(ids, skip_special_tokens=True), int(ids.numel())

    def transition(memory, action, obs, info):
        previous = memory.admissible
        valid = (action.startswith('search[') and 'search[<your query>]' in previous) or action.casefold() in [a.casefold() for a in previous]
        memory.transition(action, valid, obs, action_list(info['available_actions']), info['visible_page'])

    try:
        if stage in ('full', 'reference'):
            anchors = []
            with (out/'reference_trajectories.jsonl').open('x') as records:
                for task in spec['eval_ids']:
                    obs, info, memory = replay_visible(world, 0, task, [], max_steps)
                    history = []; seen = set(); prefix = []
                    for step in range(max_steps):
                        visible = memory.state(); bundle = router.route_many([visible])[0]
                        sid = bundle['selected_skill_id']; skill = bank.get(sid)
                        if sid not in seen:
                            anchors.append({'task_id': task, 'skill_id': sid, 'prefix_actions': list(prefix),
                                'observation': obs, 'admissible_actions': visible['admissible_actions'],
                                'history': list(history), 'anchor_step': step, 'payload_sha256': skill.payload_sha256,
                                'environment_state_sha256': world.state_digest(0), 'visible_state_sha256': digest(visible)})
                            seen.add(sid)
                        text, ntokens = generate(models['old'], build_state_prompt(visible, skill.payload), 1000000+task*100+step)
                        action, valid = project_action(text); previous = obs
                        obs, reward, done, info = world.step_one(0, action)
                        transition(memory, action, obs, info)
                        records.write(json.dumps({'task_id': task, 'step': step, 'skill_id': sid, 'action': action,
                            'raw_output': text, 'reward': reward, 'won': info['won'], 'response_tokens': ntokens, 'valid': valid})+'\n')
                        records.flush(); history.append({'observation': previous, 'action': action}); prefix.append(action)
                        if done:
                            break
                    print(json.dumps({'reference_task': task, 'anchors': len(anchors)}), flush=True)
            (out/'anchors.json').write_text(json.dumps(anchors, ensure_ascii=False)+'\n')
            (out/'reference-complete.json').write_text(json.dumps({'anchors_sha256': digest(anchors),
                'reference_task_ids': spec['eval_ids'], 'forward_contract': contract,
                'old_model_receipt_sha256': model_receipt_hash(), 'bank_sha256': bank.content_sha256})+'\n')
            if stage == 'reference':
                return
        else:
            anchors = json.loads((out/'anchors.json').read_text())
            reference = json.loads((out/'reference-complete.json').read_text())
            if reference['anchors_sha256'] != digest(anchors) or reference['reference_task_ids'] != spec['eval_ids']:
                raise ValueError('Reference anchors or task subset changed')

        def continuation(anchor, checkpoint, control, seed):
            task = anchor['task_id']; target = anchor['skill_id']
            obs, info, memory = replay_visible(world, 0, task, anchor['prefix_actions'], max_steps)
            if (obs != anchor['observation'] or action_list(info['available_actions']) != anchor['admissible_actions']
                    or world.state_digest(0) != anchor['environment_state_sha256']
                    or digest(memory.state()) != anchor['visible_state_sha256']):
                raise ValueError('Environment replay differs from the recorded decision anchor')
            used = 0; score = 0.; success = False
            for step in range(anchor['anchor_step'], max_steps):
                visible = memory.state()
                sid = target if step == anchor['anchor_step'] else router.route_many([visible])[0]['selected_skill_id']
                payload = '' if control and sid == target else bank.get(sid).payload
                prompt = build_state_prompt(visible, payload)
                reference_prompt = build_state_prompt(visible, bank.get(sid).payload) if contract in (BUCKET, ACTIVE) else None
                text, ntokens = generate(models[checkpoint], prompt, seed*1000+step-anchor['anchor_step'], reference_prompt)
                action, _ = project_action(text)
                obs, _, done, info = world.step_one(0, action); used += ntokens
                transition(memory, action, obs, info); score = info['task_score']; success = info['won']
                if done:
                    break
            return {'success': float(success), 'task_score': float(score), 'generated_tokens': used}

        if stage == 'full':
            target_path = out/'paired_anchors.jsonl'
        else:
            (out/'shards').mkdir(exist_ok=True)
            target_path = out/'shards'/f'shard-{shard:04d}.jsonl'
        results = []
        with target_path.open('x') as stream:
            for i, anchor in anchor_partition(anchors, shard, shards):
                paired = []
                for seed in seeds:
                    conditions = {}
                    for checkpoint in ('old', 'new'):
                        for arm in ('skill', 'control'):
                            key = f'{checkpoint}_{arm}'
                            conditions[key] = (old_cache[i]['paired'][list(seeds).index(seed)]['conditions'][key]
                                if checkpoint == 'old' and old_cache is not None
                                else continuation(anchor, checkpoint, arm == 'control', seed))
                    mo = conditions['old_skill']['success']-conditions['old_control']['success']
                    mn = conditions['new_skill']['success']-conditions['new_control']['success']
                    paired.append({'seed': seed, 'conditions': conditions, 'm_old': mo, 'm_new': mn, 'delta_m': mn-mo})
                row = {'anchor_id': i, 'task_id': anchor['task_id'], 'skill_id': anchor['skill_id'], 'paired': paired,
                    **{k: float(np.mean([p[k] for p in paired])) for k in ('m_old', 'm_new', 'delta_m')}}
                stream.write(json.dumps(row)+'\n'); stream.flush(); results.append(row)
                print(json.dumps({'anchor_completed': i, 'shard': shard, 'total_anchors': len(anchors)}), flush=True)
        if stage == 'full':
            merge_paired_rows([target_path], anchors, seeds)
            finish(out, results, spec, seeds, contract, smoke)
        else:
            target_path.with_name(f'shard-{shard:04d}-complete.json').write_text(json.dumps({
                'anchors_sha256': digest(anchors), 'forward_contract': contract, 'shard': shard,
                'shards': shards, 'completed': len(results), 'same_prefix_replay_verified': True,
                'same_visible_memory_replay_verified': True, 'reused_old_from': str(reuse_old_from) if old_cache is not None else None})+'\n')
    finally:
        router.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-dir', type=Path, required=True); parser.add_argument('--new-model', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stage', choices=('full', 'reference', 'continuations', 'merge'), default='full')
    parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--reuse-old-from', type=Path)
    args = parser.parse_args()
    run(args.seed_dir, args.new_model, args.prepared, smoke=args.smoke, stage=args.stage,
        shard=args.shard, shards=args.shards, reuse_old_from=args.reuse_old_from)
