"""Episode-local memory of facts actually displayed by the native browser."""
import copy
import re
from urllib.parse import urlsplit, unquote

VERSION='webshop-visible-evidence-v1'


def public_page(url):
    parts=unquote(urlsplit(url or '').path).strip('/').split('/')
    result={'page_type':parts[0] if parts and parts[0] in ('item_page','item_sub_page','search_results','done') else 'index'}
    if result['page_type'] in ('item_page','item_sub_page') and len(parts)>2:
        result['product_id']=parts[2]
        if result['page_type']=='item_sub_page' and len(parts)>5:
            kind=parts[5].casefold()
            if kind in ('description','features','reviews'):result['detail_kind']=kind
    return result


class VisibleMemory:
    def __init__(self, task, max_steps):
        self.task,self.max_steps=task,int(max_steps)
        self.actions=[];self.feedback=[];self.products={}
        self.current='';self.admissible=[];self.page={'page_type':'index'}

    def _fact(self, rows, text):
        if text and not any(row['text']==text for row in rows):
            rows.append({'text':text,'observed_at_step':len(self.actions)})

    def observe(self, observation, actions, page):
        self.current=observation;self.admissible=list(actions)
        self.page={key:page[key] for key in ('page_type','product_id','detail_kind') if key in page}
        product=self.page.get('product_id')
        if not product:return
        evidence=self.products.setdefault(product,{'titles':[],'prices':[],'selected_options':[],'details':{}})
        if self.page.get('page_type')=='item_sub_page':
            kind=self.page.get('detail_kind')
            if kind:self._fact(evidence['details'].setdefault(kind,[]),observation)
            return
        # Native rich observations put the product title immediately before Price.
        lines=re.split(r'\s*\[SEP\]\s*|\n',observation)
        for index,line in enumerate(lines):
            if re.match(r'^\s*Price:',line,re.I):
                self._fact(evidence['prices'],line.strip())
                if index and lines[index-1].strip():self._fact(evidence['titles'],lines[index-1].strip())
        for match in re.finditer(r'^You have clicked (.+)\.$',observation,re.M):
            self._fact(evidence['selected_options'],match.group(1))

    def transition(self, action, valid, observation, actions, page):
        step=len(self.actions)
        self.actions.append({'step':step,'action':action,'valid':bool(valid)})
        self.feedback.append({'action_step':step,'action':action,'valid':bool(valid),
            'observation':observation,'admissible_actions':list(actions)})
        self.feedback=self.feedback[-2:]
        self.observe(observation,actions,page)

    def state(self):
        return copy.deepcopy({'state_version':VERSION,'task_description':self.task,
            'action_history':self.actions,'recent_feedback':self.feedback,
            'current_observation':self.current,'admissible_actions':self.admissible,
            'current_page':self.page,'product_evidence':self.products,
            'step_index':len(self.actions),'max_steps':self.max_steps})


def replay_visible(world, index, task_id, prefix, max_steps):
    obs,info=world.reset_one(index,task_id)
    from webshop_phase12.prompts import action_list
    memory=VisibleMemory(world.server.goals[task_id]['instruction_text'],max_steps)
    memory.observe(obs,action_list(info['available_actions']),info.get('visible_page',{}))
    for action in prefix:
        before=info['available_actions'];allowed=action_list(before)
        valid=(action.startswith('search[') and before['has_search_bar']) or action.casefold() in [x.casefold() for x in allowed]
        obs,_,done,info=world.step_one(index,action)
        if done:raise ValueError('Reference anchor prefix has already terminated')
        memory.transition(action,valid,obs,action_list(info['available_actions']),info.get('visible_page',{}))
    return obs,info,memory
