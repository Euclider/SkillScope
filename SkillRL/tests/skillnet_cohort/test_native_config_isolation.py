"""CPU/meta-only regression for vLLM -> native reference loading order."""
from copy import deepcopy

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

from verl.utils.model import get_huggingface_actor_config, normalize_transformers_config


@pytest.fixture
def tiny_config():
    return Qwen3_5Config(text_config={
        'vocab_size': 128, 'hidden_size': 64, 'intermediate_size': 128,
        'num_hidden_layers': 1, 'num_attention_heads': 4,
        'num_key_value_heads': 2, 'head_dim': 16,
        'layer_types': ['full_attention'],
        'pad_token_id': 0, 'bos_token_id': 1, 'eos_token_id': 2,
    })


@pytest.fixture
def isolated_registry(monkeypatch):
    # The production fix never changes the registry. Only this fixture isolates
    # vLLM's actual registration side effect from the rest of the test suite.
    monkeypatch.setattr(CONFIG_MAPPING, '_extra_content', dict(CONFIG_MAPPING._extra_content))


def test_native_and_unrelated_configs_are_identity_preserving(tiny_config):
    from transformers import Qwen3Config
    other = Qwen3Config()
    assert normalize_transformers_config(tiny_config) is tiny_config
    assert normalize_transformers_config(tiny_config.text_config) is tiny_config.text_config
    assert normalize_transformers_config(other) is other


@pytest.mark.parametrize('text_only', [False, True])
def test_foreign_config_values_attention_and_source_are_preserved(tiny_config, text_only):
    foreign = pytest.importorskip('vllm.transformers_utils.configs.qwen3_5')
    source = (foreign.Qwen3_5TextConfig(**tiny_config.text_config.to_dict()) if text_only
              else foreign.Qwen3_5Config(**tiny_config.to_dict()))
    source._attn_implementation = 'sdpa'
    before = deepcopy(source.to_dict())
    native = normalize_transformers_config(source)
    assert type(native) is (Qwen3_5TextConfig if text_only else Qwen3_5Config)
    assert type(native.get_text_config()) is Qwen3_5TextConfig
    assert native._attn_implementation == 'sdpa'
    assert native.get_text_config()._attn_implementation == 'sdpa'
    for name in ('vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'layer_types',
                 'linear_conv_kernel_dim', 'linear_key_head_dim', 'linear_value_head_dim',
                 'linear_num_key_heads', 'linear_num_value_heads', 'rope_parameters',
                 'tie_word_embeddings', 'pad_token_id', 'bos_token_id', 'eos_token_id'):
        assert getattr(native.get_text_config(), name) == getattr(source.get_text_config(), name)
    assert source.to_dict() == before


def test_reference_config_after_real_vllm_parser_builds_native_text_model(
        tiny_config, isolated_registry, tmp_path):
    parser = pytest.importorskip('vllm.transformers_utils.config').HFConfigParser()
    tiny_config.save_pretrained(tmp_path)
    parser.parse(tmp_path, trust_remote_code=False)
    contaminated = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    assert type(contaminated).__module__.startswith('vllm.')
    model_class = AutoModelForCausalLM._model_mapping[type(contaminated)]
    assert model_class.config_class is not contaminated.sub_configs['text_config']
    with torch.device('meta'), pytest.raises(AttributeError, match='vocab_size'):
        AutoModelForCausalLM.from_config(contaminated)
    native = get_huggingface_actor_config(str(tmp_path))
    assert model_class.config_class is native.sub_configs['text_config']
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(native)
    assert type(model.config) is Qwen3_5TextConfig
    assert model.config.vocab_size == 128
    assert all(param.device.type == 'meta' for param in model.parameters())
    # Normalizing a native consumer must not disturb the live vLLM config/index.
    assert type(AutoConfig.from_pretrained(tmp_path, local_files_only=True)) is type(contaminated)
