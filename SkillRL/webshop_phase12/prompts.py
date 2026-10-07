import json
import re

GUIDANCE_START = '\n\n## Frozen skill guidance\n'
GUIDANCE_END = '\n## End frozen skill guidance\n'


def policy_inputs(tokenizer,prompt,*,device,budget=16384,compact=False):
    """Match native WebShop prompt width; reject overflow without truncation."""
    import torch
    chat=tokenizer.apply_chat_template([{'role':'user','content':prompt}],
        add_generation_prompt=True,tokenize=False,enable_thinking=False)
    ids=tokenizer(chat,add_special_tokens=False,return_tensors='pt')['input_ids']
    length=ids.shape[-1]
    if length>budget:raise ValueError('Paired continuation prompt exceeds registered dense budget')
    if compact:
        return {'input_ids':ids.to(device),'attention_mask':torch.ones_like(ids).to(device)}
    padding=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    dense=torch.full((1,budget),padding,dtype=torch.long)
    mask=torch.zeros_like(dense)
    dense[:,-length:]=ids;mask[:,-length:]=1
    return {'input_ids':dense.to(device),'attention_mask':mask.to(device)}


def render_visible_state(state):
    """Shared factual packet; references replace exact duplicates, never facts."""
    actions=state['admissible_actions'];current=state['current_observation']
    recent=[]
    for row in state.get('recent_feedback',[]):
        rendered={k:v for k,v in row.items() if k not in ('observation','admissible_actions')}
        if row['observation']==current:
            rendered['observation']='[same as CURRENT PAGE below]'
        else:
            old_actions=row.get('admissible_actions',[])
            rendered['observation']=compact_observation(row['observation'],old_actions)
            rendered['observation_admissible_actions']=old_actions
        recent.append(rendered)
    evidence=json.loads(json.dumps(state.get('product_evidence',{}),ensure_ascii=False))
    for product in evidence.values():
        for versions in product.get('details',{}).values():
            for row in versions:
                if row['text']==current:row['text']='[same as CURRENT PAGE below]'
    sections=[('SHOPPING GOAL',state['task_description']),
        ('COMPLETE EXECUTED ACTION HISTORY',json.dumps(state.get('action_history',[]),ensure_ascii=False,separators=(',',':'))),
        ('PREVIOUSLY OBSERVED PRODUCT EVIDENCE',json.dumps(evidence,ensure_ascii=False,separators=(',',':'))),
        ('RECENT ACTIONS AND PAGE FEEDBACK',json.dumps(recent,ensure_ascii=False,separators=(',',':'))),
        ('CURRENT PAGE',compact_observation(current,actions)),
        ('ADMISSIBLE ACTIONS',json.dumps(actions,ensure_ascii=False,separators=(',',':'))),
        ('PUBLIC PAGE TYPE',json.dumps(state.get('current_page',{}),ensure_ascii=False)),
        ('PROGRESS',f"Step {state['step_index']} of at most {state.get('max_steps',50)} actions.")]
    return '\n'.join(name+':\n'+text+'\n' for name,text in sections)


def build_state_prompt(state,payload):
    prompt='You are an autonomous agent in the WebShop shopping simulator.\n'
    if payload:prompt+=GUIDANCE_START+payload+GUIDANCE_END
    prompt+='Option-label ranges refer to zero-based indices in the corresponding admissible action list.\n'
    prompt+=render_visible_state(state)
    return prompt+'\nChoose one admissible action. Reply only with <action>search[query]</action> or <action>click[label]</action>. Do not include reasoning or think tags.'


def _render_prompt(task, observation, actions, history, payload, *, indexed=False):
    text = 'You are an autonomous agent in the WebShop shopping simulator.\nShopping goal: '+task
    if payload:
        text += GUIDANCE_START+payload+GUIDANCE_END
    if indexed:
        text += '\nObservation option-label ranges refer to zero-based indices in the admissible action list below.'
    text += '\nRecent visible observations and actions: '+json.dumps(history[-2:],ensure_ascii=False)
    text += '\nCurrent observation: '+observation
    text += '\nAdmissible actions: '+json.dumps(actions,ensure_ascii=False)
    text += '\nChoose one admissible action. Reply only with <action>search[query]</action> or <action>click[label]</action>. Do not include reasoning or think tags.'
    return text


def compact_observation(observation, actions):
    """Reference duplicate menu labels; keep every legal command in the action list."""
    indices = {a[6:-1].casefold(): i for i, a in enumerate(actions)
               if a.startswith('click[') and a.endswith(']')}
    result, pending = [], []

    def flush():
        if not pending:
            return
        start = previous = pending[0]
        ranges = []
        for index in pending[1:]:
            if index == previous + 1:
                previous = index
            else:
                ranges.append(str(start) if start == previous else f'{start}-{previous}')
                start = previous = index
        ranges.append(str(start) if start == previous else f'{start}-{previous}')
        result.append('[labels of admissible actions '+','.join(ranges)+']')
        pending.clear()

    rich='\n' in observation
    for part in (observation.splitlines() if rich else observation.split(' [SEP] ')):
        # Preserve clicked markers: they carry selected/visited evidence.
        label=re.fullmatch(r'\s*\[button\]\s*(.*?)\s*\[button_\]\s*',part)
        lookup=label.group(1) if label else part
        if lookup.casefold() in indices:
            pending.append(indices[lookup.casefold()])
        else:
            flush()
            result.append(part)
    flush()
    return ('\n' if rich else ' [SEP] ').join(result)


def build_prompt(task, observation, actions, history, payload):
    original = _render_prompt(task, observation, actions, history, payload)
    if len(original) <= 13000:
        return original
    observation = compact_observation(observation, actions)
    recent = [{**row, 'observation': compact_observation(row['observation'], actions)}
              for row in history[-2:]]
    compacted = _render_prompt(task, observation, actions, recent, payload, indexed=True)
    if len(compacted) > 13000:
        # As in upstream, reduce old observation history for long pages, while
        # preserving recent actions, the shopping goal and the target skill.
        recent = [{'action': row['action']} for row in history[-2:]]
        compacted = _render_prompt(task, observation, actions, recent, payload, indexed=True)
    return compacted


def remove_guidance(prompt, payload):
    segment = GUIDANCE_START+payload+GUIDANCE_END
    if not payload or prompt.count(segment) != 1:
        raise ValueError('Target skill segment must occur exactly once in the recorded prompt')
    return prompt.replace(segment,'',1)


def project_action(text):
    match = re.fullmatch(r'\s*<action>\s*((?:search|click)\[[^\n]+\])\s*</action>\s*',text,re.IGNORECASE)
    if not match:
        return 'invalid', False
    return match.group(1).strip().lower(), True


def action_list(info):
    result = ['search[<your query>]'] if info['has_search_bar'] else []
    return result+[f'click[{a}]' for a in info['clickables']]
