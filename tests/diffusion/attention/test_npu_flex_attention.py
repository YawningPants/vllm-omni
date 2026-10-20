# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contract tests; NPU input doubles delegate arithmetic to CPU tensors."""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.utils import npu_flex_attention as flex

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _NPUInput:
    device = SimpleNamespace(type="npu")

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


@pytest.fixture(autouse=True)
def clear_initializer_cache():
    flex._get_flex_attention.cache_clear()
    yield
    flex._get_flex_attention.cache_clear()


@pytest.mark.parametrize(
    "image_ids",
    [
        [-1, -1, 0, 0, 0, 0],  # T2I
        [-1, 0, 0, 1, 1, -1, 2, 2, -1],  # Adjacent references and trailing text
    ],
)
@torch.inference_mode()
def test_flex_output_matches_independent_dense_mask(monkeypatch, image_ids):
    module = importlib.import_module("torch.nn.attention.flex_attention")
    monkeypatch.setattr(module, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG", True)
    monkeypatch.setattr(flex, "_get_flex_attention", lambda: module.flex_attention)

    def fail_compile(*args, **kwargs):
        pytest.fail("this attention path must not invoke torch.compile")

    monkeypatch.setattr(torch, "compile", fail_compile)
    generator = torch.Generator().manual_seed(42)
    q, k, v = [torch.randn(1, len(image_ids), 2, 16, generator=generator).bfloat16() for _ in range(3)]
    ids = torch.tensor(image_ids)
    spans = [(int((ids == i).nonzero()[0]), int((ids == i).nonzero()[-1]) + 1) for i in ids.unique() if i >= 0]
    metadata = AttentionMetadata(full_attn_spans=[spans])
    output = flex.npu_flex_attention(*[_NPUInput(t) for t in (q, k, v)], metadata, 0.37)

    positions = torch.arange(len(image_ids))
    keep = (positions[:, None] >= positions[None, :]) | ((ids[:, None] >= 0) & (ids[:, None] == ids[None, :]))
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=keep, scale=0.37
    ).transpose(1, 2)
    assert output.shape == q.shape and output.dtype == q.dtype
    assert output.is_contiguous() and torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, atol=5e-3, rtol=2e-2)


def test_initializer_registers_npu_before_loading_public_api(monkeypatch):
    imports = []
    sentinel = object()
    module = SimpleNamespace(_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG=False, flex_attention=sentinel)

    def fake_import(name):
        imports.append(name)
        return module

    monkeypatch.setattr(flex.importlib, "import_module", fake_import)
    assert flex._get_flex_attention() is sentinel
    assert flex._get_flex_attention() is sentinel
    assert imports == ["torch_npu._inductor", "torch.nn.attention.flex_attention"]
    assert module._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG is True


@pytest.mark.parametrize(
    "error",
    [ImportError("missing NPU support"), OSError("missing Triton library"), RuntimeError("version mismatch"), None],
)
def test_incompatible_environment_raises(monkeypatch, error):
    def unavailable(name):
        if error is not None:
            raise error
        return SimpleNamespace(flex_attention=lambda: None)

    monkeypatch.setattr(flex.importlib, "import_module", unavailable)
    with pytest.raises(type(error) if error is not None else ImportError):
        flex._get_flex_attention()


@pytest.mark.parametrize(
    "unsupported",
    ["cpu", "fp32", "batch", "decode", "gqa", "mask", "joint", "query_ranges", "no_spans", "grad", "compile"],
)
def test_unsupported_inputs_do_not_initialize_flex(monkeypatch, unsupported):
    query = _NPUInput(torch.zeros(1, 6, 2, 16, dtype=torch.bfloat16))
    key = value = query
    metadata = AttentionMetadata(full_attn_spans=[[(2, 6)]])
    if unsupported == "cpu":
        query = key = value = query.tensor
    elif unsupported == "fp32":
        query = key = value = _NPUInput(query.tensor.float())
    elif unsupported == "batch":
        query = key = value = _NPUInput(query.tensor.expand(2, -1, -1, -1))
    elif unsupported == "decode":
        query = _NPUInput(query.tensor[:, -2:])
    elif unsupported == "gqa":
        key = value = _NPUInput(key.tensor[:, :, :1])
    elif unsupported == "mask":
        metadata.attn_mask = torch.ones(1, 6, dtype=torch.bool)
    elif unsupported == "joint":
        metadata.joint_key = key.tensor
    elif unsupported == "query_ranges":
        metadata.query_ranges = ((0, 6),)
    elif unsupported == "no_spans":
        metadata.full_attn_spans = None
    elif unsupported == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def fail_init():
        pytest.fail("unsupported inputs must be rejected before initializing Flex")

    monkeypatch.setattr(flex, "_get_flex_attention", fail_init)
    with torch.set_grad_enabled(unsupported == "grad"):
        with pytest.raises(ValueError, match='attn_path="flex" requires'):
            flex.npu_flex_attention(query, key, value, metadata, 0.25)


@torch.inference_mode()
@pytest.mark.parametrize(
    "error",
    [torch.OutOfMemoryError("full attention scores"), ValueError("bad shape"), RuntimeError("execution failed")],
)
def test_execution_failure_propagates_without_disabling_flex(monkeypatch, error):
    query = _NPUInput(torch.zeros(1, 6, 2, 16, dtype=torch.bfloat16))
    metadata = AttentionMetadata(full_attn_spans=[[(2, 6)]])

    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise error

    monkeypatch.setattr(flex, "_get_flex_attention", lambda: fail)
    for _ in range(2):
        with pytest.raises(type(error)) as caught:
            flex.npu_flex_attention(query, query, query, metadata, 0.25)
        assert caught.value is error
    assert calls == [True, True]


@torch.inference_mode()
def test_api_without_npu_registration_raises(monkeypatch):
    query = _NPUInput(torch.zeros(1, 6, 2, 16, dtype=torch.bfloat16))
    metadata = AttentionMetadata(full_attn_spans=[[(2, 6)]])

    def reject_device(*args, **kwargs):
        raise ValueError(
            "FlexAttention is only supported on CUDA, CPU or HPU devices. Found input tensors on npu device."
        )

    monkeypatch.setattr(flex, "_get_flex_attention", lambda: reject_device)
    with pytest.raises(ValueError, match="Found input tensors on npu device"):
        flex.npu_flex_attention(query, query, query, metadata, 0.25)
