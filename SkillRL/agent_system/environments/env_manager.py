# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        # Add retrieval memory or skills-only memory if configured
        if config.env.get('use_skills_only_memory', False):
            from agent_system.memory import SkillsOnlyMemory
            som_cfg = config.env.skills_only_memory
            self.retrieval_memory = SkillsOnlyMemory(
                skills_json_path=som_cfg.skills_json_path,
                retrieval_mode=som_cfg.get('retrieval_mode', 'template'),
                embedding_model_path=som_cfg.get('embedding_model_path', None),
                task_specific_top_k=som_cfg.get('task_specific_top_k', None),
            )
            self.retrieved_memories = None
            print(f"[SearchEnvironmentManager] Skills-only memory enabled "
                  f"(mode={som_cfg.get('retrieval_mode', 'template')})")
        elif config.env.get('use_retrieval_memory', False):
            from agent_system.memory import RetrievalMemory
            self.retrieval_memory = RetrievalMemory(
                memory_json_path=config.env.retrieval_memory.json_path,
                embedding_model_name=config.env.retrieval_memory.get('embedding_model', 'Qwen/Qwen3-Embedding-0.6B'),
                device=config.env.retrieval_memory.get('device', 'cuda'),
                skills_json_path=config.env.retrieval_memory.get('skills_json_path', None)
            )
            self.retrieved_memories = None  # Store retrieved memories per episode
            print(f"[SearchEnvironmentManager] Retrieval memory enabled")
        else:
            self.retrieval_memory = None
            self.retrieved_memories = None

        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        self.kwargs = kwargs
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs
        self.memory.reset(batch_size=len(obs))
        if self.retrieval_memory is not None:
            self.retrieved_memories = []

            # Determine which config to use
            if self.config.env.get('use_skills_only_memory', False):
                mem_config = self.config.env.skills_only_memory
            else:
                mem_config = self.config.env.retrieval_memory

            for task in self.tasks:
                memories = self.retrieval_memory.retrieve(
                    task_description=task,
                    top_k=mem_config.get('top_k', 10),
                    similarity_threshold=mem_config.get('similarity_threshold', 0.7),
                    max_tokens=mem_config.get('max_tokens', 2000),
                    include_examples=mem_config.get('include_examples', False)
                )
                self.retrieved_memories.append(memories)

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            # Use retrieval memory template if enabled
            use_retrieval = (self.retrieval_memory is not None and
                           self.retrieved_memories is not None and
                           not init)
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            elif use_retrieval:
                # Format retrieved memories for prompt
                memory_context = self.retrieval_memory.format_for_prompt(
                    self.retrieved_memories[i]
                )
                obs_i = SEARCH_TEMPLATE_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    step_count=len(self.memory[i]),
                    memory_context=memory_ctx[i],
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        self.prompt_skill_metadata = None
        self.current_prompt_texts = None
        self.step_skill_router = None
        self.external_skill_routing = False

        # Add retrieval memory or skills-only memory if configured
        if config.env.get('use_skills_only_memory', False):
            from agent_system.memory import FrozenStepSkillRouter, SkillsOnlyMemory
            som_cfg = config.env.skills_only_memory
            self.retrieved_memories = None
            router_cfg = som_cfg.get('step_routing', {})
            backend = router_cfg.get('backend', 'phase')
            if backend == 'external_llm':
                from agent_system.memory.skillnet_runtime import DEFAULT_ROUTER_PROFILE, create_skillnet37_runtime
                if not router_cfg.get('enabled', False):
                    raise ValueError('external_llm requires enabled step routing')
                if not router_cfg.get('cache_path') or 'max_api_calls' not in router_cfg:
                    raise ValueError('external_llm requires an explicit shared cache_path and max_api_calls budget')
                self.retrieval_memory, self.step_skill_router = create_skillnet37_runtime(
                    profile_path=router_cfg.get('profile_path', DEFAULT_ROUTER_PROFILE),
                    cache_path=router_cfg['cache_path'],
                    max_api_calls=router_cfg['max_api_calls'],
                )
                self.external_skill_routing = True
            elif backend in ('skillrl_embedding_state', 'skillrl_embedding_state_batch'):
                from agent_system.memory.skillnet_runtime import DEFAULT_EMBEDDING_ROUTER_PROFILE, create_embedding_skillnet37_runtime
                factory = create_embedding_skillnet37_runtime
                if backend == 'skillrl_embedding_state_batch':
                    from agent_system.memory.skillnet_runtime import DEFAULT_BATCH_EMBEDDING_ROUTER_PROFILE, create_batched_embedding_skillnet37_runtime
                    DEFAULT_EMBEDDING_ROUTER_PROFILE = DEFAULT_BATCH_EMBEDDING_ROUTER_PROFILE
                    factory = create_batched_embedding_skillnet37_runtime
                if not router_cfg.get('enabled', False):
                    raise ValueError('skillrl_embedding_state requires enabled step routing')
                if (not router_cfg.get('cache_path') or not router_cfg.get('model_path')
                        or not router_cfg.get('device') or 'max_local_calls' not in router_cfg):
                    raise ValueError('Embedding routing requires explicit model, device, new cache and local-call limit')
                self.retrieval_memory, self.step_skill_router = factory(
                    profile_path=router_cfg.get('profile_path', DEFAULT_EMBEDDING_ROUTER_PROFILE),
                    model_path=router_cfg['model_path'], device=router_cfg['device'],
                    cache_path=router_cfg['cache_path'], max_local_calls=router_cfg['max_local_calls'],
                )
                self.external_skill_routing = True  # Independent routing, not an external API.
            elif backend == 'local_frozen_hf':
                from agent_system.memory.skillnet_runtime import DEFAULT_LOCAL_ROUTER_PROFILE, create_local_skillnet37_runtime
                if not router_cfg.get('enabled', False):
                    raise ValueError('local_frozen_hf requires enabled step routing')
                if (not router_cfg.get('cache_path') or not router_cfg.get('model_path')
                        or not router_cfg.get('device') or 'max_local_calls' not in router_cfg):
                    raise ValueError('local_frozen_hf requires explicit model, device, new cache and local-call limit')
                self.retrieval_memory, self.step_skill_router = create_local_skillnet37_runtime(
                    profile_path=router_cfg.get('profile_path', DEFAULT_LOCAL_ROUTER_PROFILE),
                    model_path=router_cfg['model_path'], device=router_cfg['device'],
                    cache_path=router_cfg['cache_path'], max_local_calls=router_cfg['max_local_calls'],
                )
                # This legacy flag means independent live-state routing, not a
                # network request; it also skips terminal observations.
                self.external_skill_routing = True
            elif backend in ('phase3_external_llm', 'phase3_skillrl_embedding_state'):
                from phase3.routing import create_runtime
                if not config.get('phase3', {}).get('enabled', False):
                    raise ValueError('Evolving-bank routing requires the explicit Phase3 runner')
                self.retrieval_memory, self.step_skill_router = create_runtime(router_cfg)
                self.external_skill_routing = True
            elif backend == 'phase':
                self.retrieval_memory = SkillsOnlyMemory(
                    skills_json_path=som_cfg.skills_json_path,
                    retrieval_mode=som_cfg.get('retrieval_mode', 'template'),
                    embedding_model_path=som_cfg.get('embedding_model_path', None),
                    task_specific_top_k=som_cfg.get('task_specific_top_k', None),
                )
                if router_cfg.get('enabled', False):
                    self.step_skill_router = FrozenStepSkillRouter(
                        include_common_mistakes=router_cfg.get('include_common_mistakes', False)
                    )
            else:
                raise ValueError('Unknown step router backend; no automatic fallback')
            print(f"[AlfWorldEnvironmentManager] Skills-only memory enabled "
                  f"(mode={som_cfg.get('retrieval_mode', 'template')})")
            if self.step_skill_router is not None:
                print("[AlfWorldEnvironmentManager] Frozen step Skill routing enabled "
                      f"(router={self.step_skill_router.version})")
        elif config.env.get('use_retrieval_memory', False):
            from agent_system.memory import RetrievalMemory
            self.retrieval_memory = RetrievalMemory(
                memory_json_path=config.env.retrieval_memory.json_path,
                embedding_model_name=config.env.retrieval_memory.get('embedding_model', 'Qwen/Qwen3-Embedding-0.6B'),
                device=config.env.retrieval_memory.get('device', 'cuda'),
                skills_json_path=config.env.retrieval_memory.get('skills_json_path', None)
            )
            self.retrieved_memories = None  # Store retrieved memories per episode
            print(f"[AlfWorldEnvironmentManager] Retrieval memory enabled")
        else:
            self.retrieval_memory = None

        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        # Retrieve memories for each task if enabled
        if self.retrieval_memory is not None:
            self.retrieved_memories = []

            # Determine which config to use
            if self.config.env.get('use_skills_only_memory', False):
                mem_config = self.config.env.skills_only_memory
            else:
                mem_config = self.config.env.retrieval_memory

            for task in self.tasks:
                if self.external_skill_routing:
                    self.retrieved_memories.append(self.retrieval_memory.retrieve(task))
                    continue
                top_k = mem_config.get('top_k', 10)
                if self.step_skill_router is not None:
                    top_k = mem_config.get('step_routing', {}).get(
                        'general_top_k', top_k
                    )
                memories = self.retrieval_memory.retrieve(
                    task_description=task,
                    top_k=top_k,
                    similarity_threshold=mem_config.get('similarity_threshold', 0.7),
                    max_tokens=mem_config.get('max_tokens', 2000),
                    include_examples=mem_config.get('include_examples', False)
                )
                self.retrieved_memories.append(memories)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        raw_model_outputs = list(text_actions)
        current_observations = list(self.pre_text_obs)
        current_admissible_actions = [list(items) for items in self.envs.get_admissible_commands]
        current_prompt_texts = list(self.current_prompt_texts or [''] * len(text_actions))
        current_skill_metadata = list(self.prompt_skill_metadata or [{} for _ in text_actions])

        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(
            text_obs, self.envs.get_admissible_commands,
            routing_active=[not bool(done) for done in dones] if self.external_skill_routing else None,
        )
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['raw_model_output'] = raw_model_outputs[i]
            info['projected_action'] = actions[i]
            info['observation'] = current_observations[i]
            info['next_observation'] = text_obs[i]
            info['admissible_actions'] = current_admissible_actions[i]
            info['prompt_text'] = current_prompt_texts[i]
            info['task_description'] = self.tasks[i]
            info.update(current_skill_metadata[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False, routing_active=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        prompt_skill_metadata = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")

        routed = {}
        if (self.retrieval_memory is not None and self.retrieved_memories is not None
                and callable(getattr(self.step_skill_router, 'route_many', None))):
            active = [i for i in range(len(text_obs)) if routing_active is None or routing_active[i]]
            bundles = self.step_skill_router.route_many([
                {'candidate_bundle': self.retrieved_memories[i], 'task_description': self.tasks[i],
                 'current_observation': text_obs[i], 'admissible_actions': admissible_actions[i],
                 'history': self.memory[i], 'step_index': len(self.memory[i])} for i in active])
            if len(bundles) != len(active):
                raise ValueError('Batched router returned a different number of live-state decisions')
            routed = dict(zip(active, bundles))

        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            # Use retrieval memory template if enabled
            use_retrieval = (
                self.retrieval_memory is not None
                and self.retrieved_memories is not None
                and (not init or self.step_skill_router is not None)
                and (routing_active is None or routing_active[i])
            )

            retrieval = (
                self.retrieved_memories[i]
                if self.retrieved_memories is not None
                else {}
            )
            if use_retrieval and self.step_skill_router is not None:
                retrieval = routed[i] if i in routed else self.step_skill_router.route(
                    retrieval,
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    admissible_actions=admissible_actions[i],
                    history=self.memory[i],
                    step_index=len(self.memory[i]),
                )
            prompt_skill_metadata.append({
                'retrieved_skill_ids': list(retrieval.get('retrieved_skill_ids', [])),
                'candidate_skill_ids': list(retrieval.get('candidate_skill_ids', [])),
                # Upstream currently starts without memory injection. Record
                # that explicitly instead of attributing retrieved skills to
                # a prompt which did not contain them.
                'injected_skill_ids': (
                    list(retrieval.get('injected_skill_ids', []))
                    if use_retrieval else []
                ),
                'disabled_skill_ids': list(retrieval.get('disabled_skill_ids', [])),
                'selected_skill_id': retrieval.get('selected_skill_id'),
                'skill_task_type': retrieval.get('task_type'),
                'skill_retrieval_mode': retrieval.get('retrieval_mode'),
                'skill_router_version': retrieval.get('skill_router_version'),
                'skill_router_scores': dict(retrieval.get('skill_router_scores', {})),
                'skill_router_score_details': dict(
                    retrieval.get('skill_router_score_details', {})
                ),
                'skill_router_state_flags': list(
                    retrieval.get('skill_router_state_flags', [])
                ),
                'skill_router_selection_reason': retrieval.get(
                    'skill_router_selection_reason'
                ),
            })
            if 'skill_router_api' in retrieval and use_retrieval:
                prompt_skill_metadata[-1]['skill_router_api'] = dict(retrieval['skill_router_api'])

            if self.config.get("phase2", {}).get("enabled", False):
                prompt_skill_metadata[-1]["phase2_payload_text"] = (
                    self.retrieval_memory.format_for_prompt(retrieval) if use_retrieval else ""
                )

            if self.config.get("phase3", {}).get("enabled", False):
                prompt_skill_metadata[-1].update(
                    phase3_payload_text=self.retrieval_memory.format_for_prompt(retrieval) if use_retrieval else "",
                    bank_sha256=retrieval.get('bank_manifest_sha256'),
                    skill_version_sha256=retrieval.get('skill_version_sha256') if use_retrieval else None,
                )

            if use_retrieval and (init or (self.external_skill_routing and self.config.env.history_length <= 0)):
                memory_context = self.retrieval_memory.format_for_prompt(retrieval)
                obs = ALFWORLD_TEMPLATE_NO_HIS_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    retrieved_memories=memory_context,
                    admissible_actions=reformatted_admissible_actions,
                )
            elif init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            elif use_retrieval:
                # Format retrieved memories for prompt
                memory_context = self.retrieval_memory.format_for_prompt(
                    retrieval
                )
                obs = ALFWORLD_TEMPLATE_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            if self.config.env.alfworld.get('action_only_prompt', False):
                obs = use_action_only_instruction(obs)

            postprocess_text_obs.append(obs)
        self.prompt_skill_metadata = prompt_skill_metadata
        self.current_prompt_texts = list(postprocess_text_obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]

        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

    def save_episode_trajectories(self, batch_data_list, infos_list):
        """
        Save successful/failed trajectories from completed episodes to memory pool.

        Args:
            batch_idx: Index of the batch
            total_batch_list: List of batch data containing trajectories
            infos: List of info dicts containing episode metadata
        """
        if self.retrieval_memory is None:
            return

        save_dir = self.config.env.retrieval_memory.get('save_dir', None)
        if save_dir is None:
            return

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'new_memories.json')

        # Iterate through each environment
        for env_idx in range(len(self.tasks)):
            # Check if episode is done
            # We'll save trajectories when episodes complete
            # This will be called from the trainer after validation/training episodes
            pass  # Actual saving logic will be called from trainer


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()

        # Skills-only memory (same interface as AlfWorldEnvironmentManager)
        if config.env.get('use_skills_only_memory', False):
            from agent_system.memory import SkillsOnlyMemory
            som_cfg = config.env.skills_only_memory
            self.retrieval_memory = SkillsOnlyMemory(
                skills_json_path=som_cfg.skills_json_path,
                retrieval_mode=som_cfg.get('retrieval_mode', 'template'),
                embedding_model_path=som_cfg.get('embedding_model_path', None),
                task_specific_top_k=som_cfg.get('task_specific_top_k', None),
            )
            self.retrieved_memories = None
            print(f"[WebshopEnvironmentManager] Skills-only memory enabled "
                  f"(mode={som_cfg.get('retrieval_mode', 'template')})")
        else:
            self.retrieval_memory = None

        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)

        # Retrieve skills for each task if memory is configured
        if self.retrieval_memory is not None:
            mem_cfg = self.config.env.skills_only_memory
            self.retrieved_memories = [
                self.retrieval_memory.retrieve(
                    task_description=task,
                    top_k=mem_cfg.get('top_k', 6),
                )
                for task in self.tasks
            ]

        observations = {'text': self.build_text_obs(obs, infos, init=True),
                        'image': None,
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size=len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")

        use_retrieval = (
            self.retrieval_memory is not None
            and self.retrieved_memories is not None
            and not init
        )

        for i in range(len(text_obs)):

            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            elif use_retrieval:
                memory_context = self.retrieval_memory.format_for_prompt(
                    self.retrieved_memories[i]
                )
                obs = WEBSHOP_TEMPLATE_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            if len(obs) > 13000:
                print(f"Warning len(obs)={len(obs)} is too long")
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
            'task_types': config.env.alfworld.get('task_types', None),
            'allowed_eval_game_ids': config.env.alfworld.get('allowed_eval_game_ids', None),
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
