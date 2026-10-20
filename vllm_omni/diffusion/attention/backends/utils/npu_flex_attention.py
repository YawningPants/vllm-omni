# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Full-sequence FlexAttention for Qwen-Image 2.1 NPU prefill."""

from __future__ import annotations

import importlib
from functools import cache
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

logger = init_logger(__name__)


@cache
def _get_flex_attention():
    # Loaded only for explicit attn_path="flex". Import errors propagate.
    # Registers NPU support in the public API; does not call torch.compile.
    importlib.import_module("torch_npu._inductor")
    module = importlib.import_module("torch.nn.attention.flex_attention")
    if not hasattr(module, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG"):
        raise ImportError("torch lacks the FlexAttention debug switch")
    # Process-wide private switch: bypass Flex's internal compile wrapper.
    # This does not disable an outer torch.compile.
    module._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
    return module.flex_attention


def npu_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: AttentionMetadata,
    softmax_scale: float,
) -> torch.Tensor:
    """Run explicit B=1 BF16 prefill, materializing the full attention scores."""
    spans = attn_metadata.full_attn_spans
    if (
        torch.compiler.is_compiling()
        or torch.is_grad_enabled()
        or query.device.type != "npu"
        or query.ndim != 4
        or query.shape[0] != 1
        or query.shape != key.shape
        or query.shape != value.shape
        or any(t.dtype != torch.bfloat16 or t.device != query.device for t in (query, key, value))
        or spans is None
        or len(spans) != 1
        or attn_metadata.query_ranges is not None
        or any(
            t is not None
            for t in (
                attn_metadata.attn_mask,
                attn_metadata.joint_attn_mask,
                attn_metadata.joint_query,
                attn_metadata.joint_key,
                attn_metadata.joint_value,
            )
        )
    ):
        raise ValueError(
            'attn_path="flex" requires uncompiled, inference-only NPU prefill with '
            "B=1, BF16, equal Q/K/V shapes, and span metadata without padding or joint tensors."
        )

    flex_attention = _get_flex_attention()
    full_spans = tuple(spans[0])

    def score_mod(score, batch, head, q_idx, kv_idx):
        visible = q_idx >= kv_idx
        for start, end in full_spans:
            visible = visible | ((q_idx >= start) & (q_idx < end) & (kv_idx >= start) & (kv_idx < end))
        return torch.where(visible, score, float("-inf"))

    output = flex_attention(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        score_mod=score_mod,
        scale=softmax_scale,
    )
    logger.info_once("Qwen-Image 2.1 NPU prefill used FlexAttention (full QKV, score_mod).")
    return output.transpose(1, 2).contiguous()
