"""Real frozen bank + evaluator/branch code, with deterministic toy environment.

No ALFWorld experiment, model weights, encoder, GPU or external API is invoked.
"""
from pathlib import Path

import pytest

from agent_system.memory.frozen_skill_bank import FrozenBankError, FrozenSkillBankMemory, load_skillnet37
from phase1.conditions import SkillCondition, apply_skill_condition
from phase1.first_invocation import PayloadArm, build_anchor


class ToyEnvironment:
    initial = 'Room. Your task is to: inspect the room'

    def __init__(self, *args):
        self.count = 0
        self.closed = False

    def reset(self):
        return self.initial, {'admissible_commands': ['look'], 'won': False}

    def step(self, action):
        assert action == 'look'
        self.count += 1
        return 'Room inspected.', {'admissible_commands': ['look'], 'won': self.count == 2}, self.count == 2

    def close(self):
        self.closed = True


class ToyPolicy:
    def generate(self, prompt, seed, temperature, top_p, max_new_tokens):
        return '<action>look</action>', 100, 5


class SelectedRouter:
    version = 'offline-test-only'

    def __init__(self, memory):
        self.memory = memory

    def route(self, candidates, **state):
        assert candidates['candidate_skill_ids'] == list(self.memory.bank.skill_ids)
        return {**self.memory.selected_bundle(self.memory.bank.skill_ids[0]),
                'disabled_skill_ids': [], 'skill_router_version': self.version}


def test_legacy_full_bank_mask_contract_is_read_only_and_does_not_filter():
    memory = FrozenSkillBankMemory(load_skillnet37())
    before = memory.retrieve('')
    with apply_skill_condition(memory, SkillCondition.FULL_BANK, ''):
        assert memory.disabled_skill_ids == frozenset()
    with pytest.raises(AttributeError):
        memory.disabled_skill_ids = {'anything'}
    with pytest.raises(AttributeError):
        memory.disabled_skill_ids.add('anything')
    with pytest.raises(FrozenBankError, match='mask routing candidates'):
        memory.set_disabled_skill_ids(set())
    assert memory.retrieve('') == before and len(before['candidate_skill_ids']) == 37


@pytest.mark.parametrize('arm', list(PayloadArm))
def test_complete_evaluator_episode_and_anchored_branch_with_frozen_bank(monkeypatch, arm):
    from phase1 import eval_skill_margin as evaluator
    from phase1 import eval_first_invocation_utility as utility
    monkeypatch.setattr(evaluator, 'SingleGameEnvironment', ToyEnvironment)
    monkeypatch.setattr(utility, 'SingleGameEnvironment', ToyEnvironment)
    monkeypatch.setattr(utility, 'resolve_game_file', lambda game: Path(game))
    memory = FrozenSkillBankMemory(load_skillnet37())
    bank_identity = memory.bank.manifest_sha256, memory.bank.content_sha256
    router = SelectedRouter(memory)
    skill = memory.bank.skill_ids[0]
    result = evaluator.run_episode(policy=ToyPolicy(), game_file=Path('/offline/toy-game'),
        memory=memory, condition=SkillCondition.FULL_BANK, skill_id='',
        environment_seed=1404, eval_seed=714040, temperature=.4, top_p=1.,
        max_steps=50, max_new_tokens=512, history_length=2,
        step_skill_router=router, router_general_top_k=37)
    assert result['success'] and result['trajectory_length'] == 2
    assert all(step['disabled_skill_ids'] == [] for step in result['steps'])
    assert all(len(step['candidate_skill_ids']) == 37 for step in result['steps'])
    trajectory = {**result, 'trajectory_id': 'offline-only', 'game_id': '/offline/toy-game',
                  'environment_seed': 1404, 'eval_seed': 714040, 'checkpoint_id': 'toy'}
    anchor = build_anchor(trajectory=trajectory, trajectory_path='/offline/toy.json', skill_id=skill,
                          source_index={'split': 'offline', 'max_steps': 50})
    branch = utility.run_branch(policy=ToyPolicy(), anchor=anchor, memory=memory, router=router,
        arm=arm, target_skill_id=skill, placebo_text='neutral offline placebo',
        temperature=.4, top_p=1., max_new_tokens=512, history_length=2, router_general_top_k=37)
    assert branch['prefix_replay_verified'] and branch['success']
    assert branch['target_selected_count'] == 2
    expected = {PayloadArm.ORIGINAL: memory.bank.get(skill).payload,
                PayloadArm.PLACEBO: 'neutral offline placebo', PayloadArm.NULL: ''}[arm]
    assert all(step['payload_text'] == expected for step in branch['steps'])
    assert memory.disabled_skill_ids == frozenset()
    assert (memory.bank.manifest_sha256, memory.bank.content_sha256) == bank_identity
