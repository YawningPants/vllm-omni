# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU lock: adjacent condition images stay separate token blocks.

Two condition images with no text between them are one run of True in
``image_pad_mask``. Block ids come from ``img_shapes`` token counts, so they
must not collapse into one bidirectional attention span.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
    QwenImage21Transformer2DModel,
    current_omni_platform,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_adjacent_condition_images_stay_separate_blocks() -> None:
    # text(2) | cond0(2) | cond1(3) | text(1) | target(2)
    # cond0 and cond1 are one contiguous True run.
    image_pad_mask = torch.tensor([False, False, True, True, True, True, True, False, True, True])
    img_shapes = [(1, 1, 2), (1, 1, 3), (1, 1, 2)]

    image_ids, target_token_mask = QwenImage21Transformer2DModel.build_token_metadata(image_pad_mask, img_shapes)

    assert image_ids.tolist() == [-1, -1, 0, 0, 1, 1, 1, -1, 2, 2]
    assert target_token_mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    assert QwenImage21Transformer2DModel._build_full_attn_spans(image_ids) == [
        (2, 4),
        (4, 7),
        (8, 10),
    ]


@pytest.mark.parametrize(
    "is_npu,hint,padding,expected",
    [
        (True, "auto", False, "mask"),
        (True, "auto", True, "mask"),
        (True, "mask", False, "mask"),
        (True, "piecewise", False, "piecewise"),
        (True, "flex", False, "flex"),
        (False, "auto", False, "piecewise"),
        (False, "auto", True, "mask"),
    ],
)
def test_npu_defaults_to_mask_and_flex_requires_explicit_hint(monkeypatch, is_npu, hint, padding, expected):
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: is_npu)
    backend = SimpleNamespace(supports_piecewise_spans=True, get_name=lambda: "FLASH_ATTN")
    model = SimpleNamespace(
        causal_block=True,
        transformer_blocks=[SimpleNamespace(attn=SimpleNamespace(attn=SimpleNamespace(attn_backend=backend)))],
    )
    assert QwenImage21Transformer2DModel._resolve_attn_path(model, hint, padding, False) == expected


@pytest.mark.parametrize("unsupported", ["platform", "backend", "padding", "sp_padding", "non_block_causal"])
def test_explicit_flex_rejects_unsupported_model_paths(monkeypatch, unsupported):
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: unsupported != "platform")
    backend = SimpleNamespace(get_name=lambda: "TORCH_SDPA" if unsupported == "backend" else "FLASH_ATTN")
    model = SimpleNamespace(
        causal_block=unsupported != "non_block_causal",
        transformer_blocks=[SimpleNamespace(attn=SimpleNamespace(attn=SimpleNamespace(attn_backend=backend)))],
    )
    with pytest.raises(ValueError, match='attn_path="flex" requires'):
        QwenImage21Transformer2DModel._resolve_attn_path(
            model, "flex", unsupported == "padding", unsupported == "sp_padding"
        )
