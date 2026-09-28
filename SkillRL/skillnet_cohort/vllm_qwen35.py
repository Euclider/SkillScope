"""Text-only registration using vLLM's native Qwen3.5 kernels (not HF generate).

vLLM 0.22 ships the text implementation but registers only the multimodal
wrapper. This adapter exposes that implementation for our text-only exports,
with the SAME native hybrid-cache metadata used by the multimodal wrapper.
Text prompts still use the checkpoint's M-RoPE configuration: their three
position axes are identical and their decode-position delta is zero.
"""
from typing import TYPE_CHECKING

import torch
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration,
)

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec


class SkillScopeQwen35Text(Qwen3_5ForCausalLM, IsHybrid, SupportsMRoPE):
    get_mamba_state_dtype_from_config = Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config
    get_mamba_state_shape_from_config = Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config
    get_mamba_state_copy_func = Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func

    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list['MultiModalFeatureSpec'],
    ) -> tuple[torch.Tensor, int]:
        """Match native Qwen3.5's no-media position branch, without a vision tower.

        The upstream wrapper's helper additionally reads vision configuration
        even for text. Our exported text config intentionally has no such
        fields. Keep the same [T, H, W] positions and zero delta, and reject
        media instead of silently treating it as text. No RoPE settings or
        attention kernels are replaced here.
        """
        if mm_features:
            raise ValueError('SkillScopeQwen35Text accepts text-only prompts, not multimodal features')
        if not input_tokens:
            raise ValueError('M-RoPE requires a nonempty text prompt')
        positions = torch.arange(len(input_tokens), dtype=torch.long, device='cpu')
        return positions.unsqueeze(0).repeat(3, 1), 0

    def load_weights(self, weights):
        from .vllm_backend import normalized_text_weights
        return super().load_weights(normalized_text_weights(weights))
