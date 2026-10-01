"""Qwen3.5 generation builds 4-axis positions from the attention mask."""

from __future__ import annotations


def is_qwen35(model):
    inner = getattr(model, "module", model)
    model_type = getattr(getattr(inner, "config", None), "model_type", None)
    return model_type in {"qwen3_5", "qwen3_5_text"}
