"""Verify Safety Attention Score (SAS) matches its formal definition.

We cover:

1. **Manual numeric check** — a 4-token, 1-layer, 2-head toy with hand-
   specified attention weights. SAS is computed by hand from the
   formula and compared to ``compute_sas_dense``.

2. **Consistency** — streaming ``compute_sas`` == dense
   ``compute_sas_dense`` on the same inputs.

3. **Differentiability** — ``compute_sas`` produces a scalar whose
   gradient w.r.t. input embeddings is non-zero.

4. **Monotonicity sanity check** — zero safety attention gives
   SAS = 0; uniform attention gives the expected fraction.
"""
from __future__ import annotations

import pytest
import torch

from src.models.attention_extractor import AttentionCapture, TokenSpan
from src.models.safety_attention_score import (
    SASAccumulator,
    compute_sas,
    compute_sas_dense,
)


def _make_toy_attentions(S: int = 4, H: int = 2, L: int = 1) -> list[torch.Tensor]:
    """Return L tensors of shape (1, H, S, S) that are valid (row-stochastic,
    lower-triangular) attention matrices with hand-chosen values."""
    attns = []
    for _ in range(L):
        A = torch.zeros(1, H, S, S)
        for h in range(H):
            for i in range(S):
                row = torch.zeros(S)
                # Uniform over positions 0..i (causal)
                row[: i + 1] = 1.0 / (i + 1)
                A[0, h, i] = row
        attns.append(A)
    return attns


def test_sas_manual_formula():
    """4 tokens, 1 layer, 2 heads, uniform-causal attention. Safety = {0}.
    Output = {3}. By formula:

        SAS = (1/(L·H·|P_out|)) · Σ_l Σ_h Σ_i Σ_j a_{ij}
            = (1/(1·2·1)) · 2 · a_{3,0}
            = a_{3,0}
            = 1/4     (uniform causal at position 3 spreads over 0..3)
    """
    attns = _make_toy_attentions(S=4, H=2, L=1)
    sas = compute_sas_dense(
        attns, output_positions=[3], safety_positions=[0]
    )
    assert pytest.approx(sas.item(), abs=1e-6) == 0.25


def test_sas_multi_safety_positions_sums_not_averages():
    """Safety = {0, 1}. Output = {3}. Per formula (no |P_safety| divisor):

        SAS = (1/2) · (a_{3,0} + a_{3,1}) · H_heads_sum
             = (1/(L·H·|P_out|)) · L · H · (a_{3,0}+a_{3,1})
             = a_{3,0} + a_{3,1}
             = 1/4 + 1/4 = 1/2
    """
    attns = _make_toy_attentions(S=4, H=2, L=1)
    sas = compute_sas_dense(
        attns, output_positions=[3], safety_positions=[0, 1]
    )
    assert pytest.approx(sas.item(), abs=1e-6) == 0.5


def test_sas_multi_output_positions_averaged():
    """Output = {2, 3}. Safety = {0}. Per formula (divide by |P_out|=2):

        SAS = (1/(1·2·2)) · 2 · (a_{2,0} + a_{3,0})
            = (1/4) · 2 · (1/3 + 1/4)
            = (1/2) · (7/12) = 7/24 ≈ 0.29167
    """
    attns = _make_toy_attentions(S=4, H=2, L=1)
    sas = compute_sas_dense(
        attns, output_positions=[2, 3], safety_positions=[0]
    )
    assert pytest.approx(sas.item(), abs=1e-6) == 7.0 / 24.0


def test_sas_multi_layer_averaged():
    """2 layers, should average to one layer's value (layers identical here)."""
    attns = _make_toy_attentions(S=4, H=2, L=2)
    sas = compute_sas_dense(
        attns, output_positions=[3], safety_positions=[0]
    )
    assert pytest.approx(sas.item(), abs=1e-6) == 0.25


def test_sas_zero_when_safety_unattended():
    """If safety positions come AFTER output (impossible under causal
    attention), SAS collapses to 0."""
    attns = _make_toy_attentions(S=4, H=2, L=1)
    # Safety = {3} (last), Output = {1} (second): a_{1,3} = 0 (causal mask)
    sas = compute_sas_dense(
        attns, output_positions=[1], safety_positions=[3]
    )
    assert sas.item() == 0.0


def test_streaming_accumulator_matches_dense():
    """The streaming accumulator used by ``compute_sas`` must match the
    reference dense implementation to floating-point precision."""
    attns = _make_toy_attentions(S=5, H=3, L=4)
    dense = compute_sas_dense(
        attns, output_positions=[3, 4], safety_positions=[0, 1]
    )

    acc = SASAccumulator(
        output_positions=[3, 4],
        safety_positions=[0, 1],
        n_layers_expected=len(attns),
    )
    for i, a in enumerate(attns):
        acc(AttentionCapture(layer_idx=i, attn_weights=a))
    streamed = acc.finalize()

    torch.testing.assert_close(streamed, dense, rtol=1e-6, atol=1e-6)


def test_sas_differentiable():
    """Streaming SAS returns a tensor whose gradient can propagate."""
    S, H = 5, 3
    A = torch.softmax(torch.randn(1, H, S, S), dim=-1)
    # Force causal mask
    mask = torch.tril(torch.ones(S, S))
    A = (A * mask) / ((A * mask).sum(-1, keepdim=True) + 1e-12)
    A = A.requires_grad_(True)

    acc = SASAccumulator(
        output_positions=[S - 1],
        safety_positions=[0],
        n_layers_expected=1,
    )
    acc(AttentionCapture(layer_idx=0, attn_weights=A))
    sas = acc.finalize()
    sas.backward()
    assert A.grad is not None
    assert A.grad.abs().sum().item() > 0.0


def test_sas_via_extractor_matches_dense(
    tiny_llama_model, tiny_llama_spec, extractor, sample_inputs
):
    """End-to-end: the streaming path through ``AttentionExtractor``
    agrees with the dense reference implementation."""
    with torch.no_grad():
        out = tiny_llama_model(
            **sample_inputs, output_attentions=True, use_cache=False
        )
    dense = compute_sas_dense(
        list(out.attentions),
        output_positions=[sample_inputs["input_ids"].shape[1] - 1],
        safety_positions=[0, 1, 2],
    )
    streamed = compute_sas(
        extractor,
        sample_inputs,
        output_span=TokenSpan(sample_inputs["input_ids"].shape[1] - 1, sample_inputs["input_ids"].shape[1]),
        safety_spans=[TokenSpan(0, 3)],
    )
    torch.testing.assert_close(streamed, dense, rtol=1e-5, atol=1e-5)
