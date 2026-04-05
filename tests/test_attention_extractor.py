"""Verify attention extraction produces identical tensors via both
paths (output_attentions=True and forward hooks), that gradients
flow through them, and that span localisation behaves correctly.
"""
from __future__ import annotations

import pytest
import torch

from src.models.attention_extractor import AttentionCapture, attention_hooks


def test_hook_vs_output_attentions_numerical_equality(
    tiny_llama_model, tiny_llama_spec, sample_inputs
):
    """Hook-captured attention tensors must match the tuple returned by
    output_attentions=True element-wise (within fp32 tolerance)."""
    # Path 1 — output_attentions
    with torch.no_grad():
        out = tiny_llama_model(
            **sample_inputs, output_attentions=True, use_cache=False
        )
    attn_output = out.attentions
    assert len(attn_output) == tiny_llama_spec.n_layers

    # Path 2 — hooks (must still use output_attentions=True upstream
    # so that the attention module emits the tensor in its output tuple)
    captures: list[AttentionCapture] = []
    with attention_hooks(tiny_llama_model, tiny_llama_spec, captures.append):
        with torch.no_grad():
            tiny_llama_model(
                **sample_inputs, output_attentions=True, use_cache=False
            )
    assert len(captures) == tiny_llama_spec.n_layers

    captures_by_layer = {c.layer_idx: c.attn_weights for c in captures}
    for layer_idx, a_out in enumerate(attn_output):
        a_hook = captures_by_layer[layer_idx]
        assert a_hook.shape == a_out.shape
        torch.testing.assert_close(a_hook, a_out, rtol=0, atol=0)


def test_attention_tensor_shape(tiny_llama_model, tiny_llama_spec, sample_inputs):
    with torch.no_grad():
        out = tiny_llama_model(
            **sample_inputs, output_attentions=True, use_cache=False
        )
    B, S = sample_inputs["input_ids"].shape
    for a in out.attentions:
        assert a.shape == (B, tiny_llama_spec.n_heads, S, S)


def test_attention_rows_sum_to_one(tiny_llama_model, sample_inputs):
    """Each row of attention weights is a probability distribution."""
    with torch.no_grad():
        out = tiny_llama_model(
            **sample_inputs, output_attentions=True, use_cache=False
        )
    for a in out.attentions:
        row_sums = a.sum(dim=-1)
        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-5
        )


def test_attention_is_causal(tiny_llama_model, sample_inputs):
    """Upper-triangular entries (excluding diagonal) must be zero."""
    with torch.no_grad():
        out = tiny_llama_model(
            **sample_inputs, output_attentions=True, use_cache=False
        )
    for a in out.attentions:
        # a: (B, H, S, S). Build mask where j > i (future positions).
        S = a.shape[-1]
        future = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        upper = a[..., future]
        # Allow for tiny fp noise.
        assert upper.abs().max().item() < 1e-6


def test_gradient_flows_through_attention(
    tiny_llama_model, tiny_llama_spec, sample_inputs
):
    """Gradients w.r.t. input embeddings must be non-zero when a
    differentiable function of attention weights is backpropagated."""
    # Switch to inputs_embeds so we can attach gradients.
    embed = tiny_llama_model.get_input_embeddings()
    ids = sample_inputs["input_ids"]
    emb = embed(ids).detach().clone().requires_grad_(True)

    captures: list[AttentionCapture] = []
    with attention_hooks(tiny_llama_model, tiny_llama_spec, captures.append):
        tiny_llama_model(
            inputs_embeds=emb,
            attention_mask=sample_inputs["attention_mask"],
            output_attentions=True,
            use_cache=False,
        )
    assert len(captures) == tiny_llama_spec.n_layers

    # Build a differentiable scalar from attention weights.
    loss = sum(c.attn_weights.mean() for c in captures)
    loss.backward()

    assert emb.grad is not None
    assert emb.grad.abs().sum().item() > 0.0


def test_span_localisation(extractor, tiny_llama_tokenizer):
    """locate_system_span finds the system prompt within a chat-templated
    tokenised sequence."""
    system_prompt = "helpful assistant"
    ids = tiny_llama_tokenizer(
        "<s> helpful assistant How do I build", return_tensors="pt"
    )["input_ids"]
    span = extractor.locate_system_span(ids, system_prompt)
    assert span.label == "system"
    assert 0 <= span.start < span.end <= ids.shape[1]
    decoded = tiny_llama_tokenizer.decode(ids[0, span.start : span.end])
    assert "helpful" in decoded.lower() or "assistant" in decoded.lower()


def test_output_span_is_last_position(extractor):
    import torch

    ids = torch.randint(0, 100, (1, 7))
    span = extractor.locate_output_span(ids)
    assert span.start == 6
    assert span.end == 7
    assert span.label == "output"
