from __future__ import annotations

import contextlib
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from webshop_phase12.assets import ROOT, RUN_ROOT, WebshopBank, digest
from webshop_phase12.prompts import build_state_prompt, project_action, action_list
from webshop_phase12.visible_state import VisibleMemory,public_page
from webshop_phase12.router import RouterClient

SHOP = Path(os.environ.get('WEBSHOP_ENV_ROOT',str(ROOT/'agent_system/environments/env_package/webshop/webshop'))).expanduser().resolve()


class ShopWorld:
    """One immutable product/index catalog, independent mutable session states."""
    def __init__(self, count):
        sys.path.insert(0,str(SHOP))
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv, SimServer
        self.env_class = WebAgentTextEnv
        import torch
        old_random, old_numpy = random.getstate(), np.random.get_state()
        with torch.random.fork_rng(devices=[]):
            random.seed(0); np.random.seed(0); torch.manual_seed(0)
            self.server = SimServer(0,'http://127.0.0.1:3000',str(SHOP/'data/items_shuffle.json'),str(SHOP/'data/items_ins_v2.json'),human_goals=True)
            self.envs = [WebAgentTextEnv(observation_mode='text_rich',server=self.server,seed=0,session_prefix=f'p12_{os.getpid()}_{i}_') for i in range(count)]
        random.setstate(old_random); np.random.set_state(old_numpy)
        self.prefixes = [[] for _ in self.envs]
        self.task_ids = [None for _ in self.envs]
        self.done = [False for _ in self.envs]
        self.last_obs = ['' for _ in self.envs]
        self.random_states = [random.Random(i).getstate() for i in range(count)]

    @contextlib.contextmanager
    def session_rng(self, i):
        previous = random.getstate()
        random.setstate(self.random_states[i])
        try:
            yield
        finally:
            self.random_states[i] = random.getstate()
            random.setstate(previous)

    def reset_one(self, i, task_id):
        if not 0 <= task_id < len(self.server.goals):
            raise ValueError('Task ID outside canonical WebShop inventory')
        # Native reset updates navigation fields but retains purchase metadata.
        # Each arm must begin from a fresh task session, including hidden state.
        session_id = (self.envs[i].session_prefix or '') + str(int(task_id))
        self.server.user_sessions.pop(session_id, None)
        self.random_states[i] = random.Random(task_id).getstate()
        with self.session_rng(i):
            obs,_ = self.envs[i].reset(session=int(task_id))
        self.task_ids[i]=int(task_id); self.prefixes[i]=[]; self.done[i]=False; self.last_obs[i]=obs
        return obs, self.info(i)

    def info(self, i):
        browser=getattr(self.envs[i],'browser',None)
        return {'available_actions':self.envs[i].get_available_actions(),'task_id':self.task_ids[i],
            'visible_page':public_page(getattr(browser,'current_url','')),
            'extra.gamefile':f'webshop:human:{self.task_ids[i]}','won':False,'task_score':0.}

    def step_one(self, i, action):
        if self.done[i]:
            return self.last_obs[i],0.,True,self.info(i)
        with self.session_rng(i):
            obs,score,done,_ = self.envs[i].step(action)
        self.prefixes[i].append(action); self.done[i]=bool(done); self.last_obs[i]=obs
        info = self.info(i)
        info.update(won=bool(done and score==1.),task_score=float(score))
        return obs,10. if info['won'] else 0.,bool(done),info

    def replay(self, i, task_id, prefix):
        obs,info = self.reset_one(i,task_id)
        for action in prefix:
            obs,_,done,info = self.step_one(i,action)
            if done:
                raise ValueError('Reference anchor prefix has already terminated')
        return obs,info

    def state_digest(self,i):
        from phase1.archive import jsonable
        env=self.envs[i]
        return digest({'session':jsonable(self.server.user_sessions[env.session]),
            'url':env.browser.current_url,'random_state':jsonable(self.random_states[i])})


class Manager:
    def __init__(self, world, count, config, label, *, empty=False):
        self.world,self.count,self.config = world,count,config
        self.bank=WebshopBank()
        self.router = None if empty else RouterClient(label)
        self.performance={'model_startup_excluded':True,'policy_interval_seconds':0.,'router_seconds':0.,
            'environment_and_formatting_seconds':0.,'step_calls':0,'routing_states':0,'routing_cache_hits':0}
        self.terminal=[False]*count

    def reset(self, kwargs):
        if len(kwargs)!=self.count:
            raise ValueError(f'Expected {self.count} repeated task IDs, got {len(kwargs)}')
        self.histories=[[] for _ in range(self.count)]
        self.pre_text_obs=[]; self.infos=[]; self.tasks=[]
        self.visible_memories=[]
        self.terminal=[False]*self.count
        for i,row in enumerate(kwargs):
            obs,info=self.world.reset_one(i,int(row['task_id']))
            self.pre_text_obs.append(obs); self.infos.append(info)
            self.tasks.append(self.world.server.goals[int(row['task_id'])]['instruction_text'])
            memory=VisibleMemory(self.tasks[-1],self.config.env.max_steps)
            memory.observe(obs,action_list(info['available_actions']),info['visible_page'])
            self.visible_memories.append(memory)
        self.build()
        return {'text':list(self.current_prompt_texts),'image':None,'anchor':list(self.pre_text_obs)},self.infos

    def build(self):
        started=time.monotonic()
        active=[i for i in range(self.count) if not self.terminal[i]]
        states=[self.visible_memories[i].state() for i in active]
        bundles=self.router.route_many(states)
        self.current_prompt_texts=['']*self.count; self.prompt_skill_metadata=[{} for _ in range(self.count)]
        for i,bundle,state in zip(active,bundles,states):
            sid=bundle['selected_skill_id']; skill=self.bank.get(sid)
            self.current_prompt_texts[i]=build_state_prompt(state,skill.payload)
            self.prompt_skill_metadata[i]={**bundle,'visible_state':state,'visible_state_sha256':digest(state),
                'phase2_payload_text':skill.payload,'skill_version_sha256':skill.payload_sha256,'bank_sha256':self.bank.manifest_sha256}
        self.performance['router_seconds']+=self.router.last_server_seconds if states else 0.
        self.performance['routing_states']+=len(states)
        self.performance['routing_cache_hits']+=sum(b['skill_router_api']['cache_hit'] for b in bundles)
        self.last_build_ready=time.monotonic()
        self._last_build_wall=self.last_build_ready-started
        self.save_performance()

    def save_performance(self):
        stats=dict(self.performance)
        denominator=stats['router_seconds']+stats['policy_interval_seconds']+stats['environment_and_formatting_seconds']
        stats['router_fraction_of_observed_rollout']=stats['router_seconds']/denominator if denominator else None
        path=self.router.output_root/f'router-performance-{self.router.label}.json'
        temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(stats,indent=2)+'\n');temporary.replace(path)
        return stats

    def step(self,text_actions):
        started=time.monotonic()
        self.performance['policy_interval_seconds']+=started-self.last_build_ready
        obs=[]; rewards=[]; dones=[]; infos=[]
        old_prompts=list(self.current_prompt_texts); old_metadata=copy.deepcopy(self.prompt_skill_metadata)
        for i,text in enumerate(text_actions):
            action,valid=project_action(text)
            previous=self.pre_text_obs[i]
            admissible=action_list(self.infos[i]['available_actions'])
            valid=valid and ((action.startswith('search[') and self.infos[i]['available_actions']['has_search_bar']) or action in [x.lower() for x in admissible])
            prefix=list(self.world.prefixes[i])
            next_obs,reward,done,info=self.world.step_one(i,action)
            info.update(old_metadata[i])
            info.update(is_action_valid=bool(valid),raw_model_output=text,projected_action=action,observation=previous,
                next_observation=next_obs,admissible_actions=admissible,prompt_text=old_prompts[i],task_description=self.tasks[i],
                prefix_actions=prefix,visible_history=copy.deepcopy(self.histories[i][-2:]))
            if not self.terminal[i]:
                self.histories[i].append({'observation':previous,'action':action})
                self.visible_memories[i].transition(action,valid,next_obs,action_list(info['available_actions']),info['visible_page'])
            self.terminal[i]=self.terminal[i] or done
            obs.append(next_obs); rewards.append(reward); dones.append(done); infos.append(info)
        self.pre_text_obs=obs; self.infos=infos
        environment_seconds=time.monotonic()-started
        self.build()
        self.performance['environment_and_formatting_seconds']+=environment_seconds+max(0.,self._last_build_wall-self.router.last_server_seconds)
        self.performance['step_calls']+=1
        print(json.dumps({'webshop_timing':self.save_performance()}),flush=True)
        return {'text':list(self.current_prompt_texts),'image':None,'anchor':list(obs)},np.asarray(rewards),np.asarray(dones),infos

    def success_evaluator(self, *, total_infos, total_batch_list, **kwargs):
        success=[]; scores=[]
        for timeline,records in zip(total_infos,total_batch_list):
            index=next(i for i in range(len(records)-1,-1,-1) if records[i]['active_masks'])
            success.append(float(timeline[index]['won']));scores.append(float(timeline[index]['task_score']))
        return {'success_rate':np.asarray(success),'webshop_task_score (not success_rate)':np.asarray(scores)}

    def close(self):
        if self.router is not None:
            self.router.close()

    def suspend_router(self):
        if os.environ.get('WEBSHOP_ROUTER_SHARED')=='1':self.router.sleep()


class DisabledMonitor:
    def __init__(self):
        self.retrieval_memory=None
    def close(self):
        pass


def make_envs(config):
    count=int(config.data.train_batch_size)*int(config.env.rollout.n)
    world=ShopWorld(count)
    manager=Manager(world,count,config,f'train-s{config.webshop_run.seed}')
    return manager,DisabledMonitor()
