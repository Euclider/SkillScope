import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from webshop_phase12.assets import BASE_MODEL, ROOT, WebshopBank
from webshop_phase12.prompts import build_prompt, remove_guidance


@pytest.fixture(scope='module')
def oversized_state():
    return json.loads((Path(__file__).parent / 'fixtures/oversized_menu_state.json').read_text())


@pytest.fixture(scope='module')
def policy_tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)


def test_actual_long_menu_preserves_goal_skill_and_every_legal_action(oversized_state, policy_tokenizer):
    from verl.utils.torch_functional import tokenize_and_postprocess_data

    with initialize_config_dir(config_dir=str(ROOT / 'verl/trainer/config'), version_base=None):
        cfg = compose(config_name='webshop54_phase12_v1')
    payload = WebshopBank().get('gen_001').payload
    state = oversized_state
    original = build_prompt(state['task_description'], state['current_observation'],
                            state['admissible_actions'], state['history'], payload)
    for prompt in (original, remove_guidance(original, payload)):
        chat = policy_tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], add_generation_prompt=True,
            tokenize=False, enable_thinking=False)
        expected_ids = policy_tokenizer(chat, add_special_tokens=False)['input_ids']
        ids, mask = tokenize_and_postprocess_data(
            prompt=chat, tokenizer=policy_tokenizer, max_length=cfg.data.max_prompt_length,
            pad_token_id=policy_tokenizer.pad_token_id, left_pad=True,
            truncation=cfg.data.truncation)
        assert ids[0, mask[0].bool()].tolist() == expected_ids
        assert state['task_description'] in prompt
        assert 'Price: $15.99' in prompt
        assert 'CA Perfume Impression of New York Oud for Man' in prompt
        assert all(action in prompt for action in state['admissible_actions'])
    assert original.count(payload) == 1
    assert payload not in remove_guidance(original, payload)
