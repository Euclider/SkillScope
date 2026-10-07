from __future__ import annotations

import hashlib
import json
import random
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(os.environ.get('WEBSHOP_RUN_ROOT',str(ROOT/'artifacts/webshop'))).expanduser().resolve()
BANK_PATH = ROOT / 'memory_data/webshop/claude_style_skills.json'
BANK_SHA256 = '79c6c60b6757b6b730e7471b537781936ce0e9cdbd8df6188b57c535893aec20'
BASE_MODEL = Path(os.environ.get('WEBSHOP_BASE_MODEL',str(Path.home()/'model/Qwen3.5-4B'))).expanduser().resolve()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Skill:
    skill_id: str
    name: str
    description: str
    payload: str
    payload_sha256: str


class WebshopBank:
    payload_renderer = 'skillscope.webshop_original_fields.v1'
    bank_id = 'skillrl-webshop54-8e66726'

    def __init__(self, path=BANK_PATH):
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != BANK_SHA256:
            raise ValueError('Frozen WebShop source bank changed')
        source = json.loads(raw)
        rows = source['general_skills'] + [s for group in source['task_specific_skills'].values() for s in group]
        skills = []
        for row in rows:
            payload = f"{row['title']}\nPrinciple: {row['principle']}\nWhen to apply: {row['when_to_apply']}"
            skills.append(Skill(row['skill_id'], row['title'], row['principle']+'\n'+row['when_to_apply'], payload, hashlib.sha256(payload.encode()).hexdigest()))
        self._skills = tuple(skills)
        if len(skills) != 54 or len({s.skill_id for s in skills}) != 54:
            raise ValueError('Expected all 54 unique original skills')
        self._by_id = MappingProxyType({s.skill_id:s for s in skills})
        self.content_sha256 = BANK_SHA256
        self.manifest = {'schema_version':'skillscope.webshop54_bank.v1', 'source_sha256':BANK_SHA256, 'renderer':self.payload_renderer,
            'skills':[{'skill_id':s.skill_id,'payload_sha256':s.payload_sha256} for s in skills],
            'source_fields_unchanged':True,'bank_frozen':True,'common_mistakes_injected':False}
        self.manifest_sha256 = digest(self.manifest)

    @property
    def skill_ids(self):
        return tuple(s.skill_id for s in self._skills)

    def __len__(self):
        return len(self._skills)

    def get(self, skill_id):
        return self._by_id[skill_id]

    def router_catalog(self):
        return [{'skill_id':s.skill_id,'name':s.name,'description':s.description} for s in self._skills]


def make_schedule(number_of_goals, *, tasks_per_update=128, eval_tasks=500, eval_seed=20261007):
    if tasks_per_update < 2 or tasks_per_update % 2 or not 1 <= eval_tasks <= 500:
        raise ValueError('Expected positive even training budget and1..500 Eval tasks')
    ntrain = number_of_goals-1500
    per_update = min(tasks_per_update, ntrain//5)//2*2
    if per_update < 2:
        raise ValueError('Insufficient train tasks for five distinct-task updates')
    seeds = {}
    for seed in (404,505):
        seeds[str(seed)] = random.Random(seed).sample(range(1500,number_of_goals),per_update*5)
    evaluation = list(range(500)) if eval_tasks == 500 else sorted(random.Random(eval_seed).sample(range(500), eval_tasks))
    return {'number_of_goals':number_of_goals,'native_split_seed':0,'train_ids':list(range(1500,number_of_goals)),
        'dev_ids':list(range(500,1500)),'eval_ids':evaluation,'eval_sampling_seed':eval_seed,
        'tasks_per_update':per_update,'updates':5,
        'repeats':8,'seeds':seeds,'sampling':'unique tasks across all five updates within each training seed'}
