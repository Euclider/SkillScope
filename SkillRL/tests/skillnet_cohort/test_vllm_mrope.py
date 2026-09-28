"""Real installed vLLM protocol/position checks; no engine, model weights or GPU."""
from copy import deepcopy

import pytest
import torch

pytest.importorskip('vllm')

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from vllm.model_executor.models.interfaces import is_hybrid, supports_mrope
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration,
)

from skillnet_cohort.vllm_backend import text_config
from skillnet_cohort.vllm_qwen35 import SkillScopeQwen35Text


def text_adapter():
    # Position construction must not need a CUDA engine, model weights or the
    # multimodal wrapper's vision config.
    model = SkillScopeQwen35Text.__new__(SkillScopeQwen35Text)
    torch.nn.Module.__init__(model)
    model.config = Qwen3_5Config().text_config
    return model


def test_text_adapter_exposes_both_native_required_protocols():
    # Regression: inheriting only the text implementation omitted SupportsMRoPE.
    assert not supports_mrope(Qwen3_5ForCausalLM)
    assert supports_mrope(SkillScopeQwen35Text)
    assert supports_mrope(text_adapter())
    assert is_hybrid(SkillScopeQwen35Text)


@pytest.mark.parametrize('length', [1, 16, 4096, 4608])
def test_text_positions_are_exactly_native_qwen35_positions(length):
    config = Qwen3_5Config()
    tokens = [100] * length
    expected, expected_delta = Qwen3_5ForConditionalGeneration._get_mrope_input_positions(
        tokens, [], config,
    )
    actual, delta = text_adapter().get_mrope_input_positions(tokens, [])
    assert torch.equal(actual, expected)
    assert actual.shape == (3, length)
    assert actual.dtype == torch.int64 and actual.device.type == 'cpu'
    assert actual.is_contiguous()
    assert delta == expected_delta == 0
    # Chunked prefill slices the absolute positions; decode starts at prompt N.
    assert torch.equal(actual[:, length // 2:], expected[:, length // 2:])
    assert length + delta == int(actual.max()) + 1


def test_special_token_ids_without_media_do_not_change_text_positions():
    config = Qwen3_5Config()
    tokens = [config.vision_start_token_id, config.image_token_id,
              config.vision_end_token_id, config.video_token_id, 100]
    expected, expected_delta = Qwen3_5ForConditionalGeneration._get_mrope_input_positions(
        tokens, [], config,
    )
    actual, delta = text_adapter().get_mrope_input_positions(tokens, [])
    assert torch.equal(actual, expected) and delta == expected_delta


def test_text_adapter_rejects_media_and_empty_prompts():
    model = text_adapter()
    with pytest.raises(ValueError, match='text-only'):
        model.get_mrope_input_positions([100], [object()])
    with pytest.raises(ValueError, match='nonempty'):
        model.get_mrope_input_positions([], [])


def test_text_export_keeps_all_rope_settings_unchanged():
    source = Qwen3_5Config(text_config={'rope_parameters': {
        'rope_type': 'default', 'rope_theta': 10000000.0,
        'partial_rotary_factor': 0.25, 'mrope_interleaved': True,
        'mrope_section': [11, 11, 10],
    }})
    before = deepcopy(source.to_dict())
    exported = text_config(source)
    # Compare stand-alone text serializations on both sides: outer to_dict()
    # deliberately strips nested transformers_version metadata.
    expected = deepcopy(source.text_config.to_dict())
    expected['architectures'] = ['SkillScopeQwen35Text']
    assert exported.to_dict() == expected
    assert source.to_dict() == before
    assert exported.rope_parameters['mrope_section'] == source.text_config.rope_parameters['mrope_section']
