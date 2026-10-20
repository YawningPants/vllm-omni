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
_flex_failed = False


@cache
def _get_flex_attention():
    try:
        # Registers NPU support in the public API; does not call torch.compile.
        importlib.import_module("torch_npu._inductor")
        module = importlib.import_module("torch.nn.attention.flex_attention")
        if not hasattr(module, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG"):
            raise ImportError("torch lacks the FlexAttention debug switch")
        flex_attention = module.flex_attention
        # Process-wide private switch: bypass Flex's internal compile wrapper.
        # This does not disable an outer torch.compile.
        module._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
    except Exception as exc:
        # Missing or mismatched torch_npu/Triton wheels must not stop inference.
        logger.warning("NPU FlexAttention unavailable; falling back to FA Score with a full mask. Reason: %s", exc)
        return None
    return flex_attention


def try_npu_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: AttentionMetadata,
    softmax_scale: float,
) -> torch.Tensor | None:
    """Return None outside the validated B=1 BF16, unsharded prefill contract.

    This implementation materializes the full attention scores. A Flex failure
    disables further attempts in this process; the caller supplies masked FA.
    """
    global _flex_failed
    spans = attn_metadata.full_attn_spans
    if (
        _flex_failed
        or torch.compiler.is_compiling()
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
        return None

    flex_attention = _get_flex_attention()
    if flex_attention is None:
        return None
    full_spans = tuple(spans[0])

    def score_mod(score, batch, head, q_idx, kv_idx):
        visible = q_idx >= kv_idx
        for start, end in full_spans:
            visible = visible | ((q_idx >= start) & (q_idx < end) & (kv_idx >= start) & (kv_idx < end))
        return torch.where(visible, score, float("-inf"))

    try:
        output = flex_attention(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            score_mod=score_mod,
            scale=softmax_scale,
        )
        output = output.transpose(1, 2).contiguous()
    except Exception as exc:
        _flex_failed = True
        logger.warning(
            "NPU FlexAttention failed; falling back to FA Score with a full mask for this process. Reason: %s", exc
        )
        return None
    logger.info_once("Qwen-Image 2.1 NPU prefill used FlexAttention (full QKV, score_mod).")
    return output
