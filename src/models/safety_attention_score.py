"""Safety Attention Score (SAS) metric.

Formal definition (see theory/framework.md §3 and the paper §3):

    SAS(x) = (1 / (L · H · |P_out|))
             · Σ_{l=1}^{L} Σ_{h=1}^{H} Σ_{i ∈ P_out} Σ_{j ∈ P_safety}  a_{i,j}^{l,h}

where
    L, H         number of layers, attention heads
    P_out ⊂ [n]  output-generating token positions
    P_safety ⊂ [n]  safety-critical token positions (system prompt span,
                 refusal-trigger span, etc.)
    a_{i,j}^{l,h}  post-softmax attention weight from position i to j at
                 layer l, head h.

The metric is the average attention mass that output positions allocate
to safety tokens, averaged over all (layer, head) pairs. No
normalisation by |P_safety| — bigger safety spans legitimately deserve
more attention mass.

Why layer-by-layer streaming
----------------------------
For an S-token sequence on an L-layer, H-head model, the full attention
tensor is ``L · H · S²``. At L=32, H=32, S=2048 that is 4.3 × 10⁹ float16
entries ≈ 8 GB — impractical to materialise at once during gradient
optimisation on a 7B model. We use ``attention_hooks`` to process one
layer at a time, reducing the tensor to a scalar contribution that
retains its computation graph back to the model parameters and input
embeddings. Peak additional memory is then a single ``(B, H, S, S)``
tensor per layer.

Differentiability
-----------------
``compute_sas`` returns a zero-dim ``torch.Tensor`` whose gradient can be
taken w.r.t. the model's input embeddings (this is what ARA does). The
surrounding autograd graph is preserved because the hook only indexes
into the attention weights — no ``.detach()``, no ``.item()``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch import Tensor

from .attention_extractor import (
    AttentionCapture,
    AttentionExtractor,
    TokenSpan,
)

log = logging.getLogger(__name__)


@dataclass
class SASAccumulator:
    """Streaming reducer for SAS. Intended to be passed as the
    ``consumer`` callback to ``AttentionExtractor.stream_attentions``.
    """

    output_positions: Sequence[int]
    safety_positions: Sequence[int]
    n_layers_expected: int
    # Populated during the forward pass:
    partial: Tensor = field(default_factory=lambda: torch.tensor(0.0))
    layers_seen: int = 0
    heads_per_layer: int = 0

    def __call__(self, cap: AttentionCapture) -> None:
        # cap.attn_weights: (batch, n_heads, seq_q, seq_k)
        aw = cap.attn_weights
        if aw.dim() != 4:
            raise ValueError(
                f"attention tensor must be 4-D, got {tuple(aw.shape)}"
            )
        b, h, sq, sk = aw.shape
        # Sanity-check positions.
        if any(p >= sq for p in self.output_positions):
            raise IndexError(
                f"output position beyond seq len: {self.output_positions} vs sq={sq}"
            )
        if any(p >= sk for p in self.safety_positions):
            raise IndexError(
                f"safety position beyond seq len: {self.safety_positions} vs sk={sk}"
            )
        # Slice: (batch, n_heads, |P_out|, |P_safety|)
        out_idx = torch.as_tensor(list(self.output_positions), device=aw.device)
        saf_idx = torch.as_tensor(list(self.safety_positions), device=aw.device)
        sub = aw.index_select(-2, out_idx).index_select(-1, saf_idx)
        # Sum over (output, safety) axes — keeps (batch, n_heads).
        s = sub.sum(dim=(-2, -1))
        # Sum over heads, mean over batch, keep scalar with grad.
        layer_contribution = s.sum(dim=-1).mean()
        # Move accumulator to the same device/dtype as the first capture
        # to avoid dtype upcasts mid-sum.
        if self.layers_seen == 0:
            self.partial = self.partial.to(device=aw.device, dtype=aw.dtype)
            self.heads_per_layer = h
        self.partial = self.partial + layer_contribution
        self.layers_seen += 1

    def finalize(self) -> Tensor:
        if self.layers_seen == 0:
            raise RuntimeError("no layers observed; did you forget to register hooks?")
        if self.layers_seen != self.n_layers_expected:
            log.warning(
                "SAS observed %d/%d layers", self.layers_seen, self.n_layers_expected
            )
        denom = (
            float(self.layers_seen)
            * float(self.heads_per_layer)
            * float(max(1, len(self.output_positions)))
        )
        return self.partial / denom


def compute_sas(
    extractor: AttentionExtractor,
    inputs: dict[str, Tensor],
    *,
    output_span: TokenSpan,
    safety_spans: Sequence[TokenSpan],
    layers: Optional[Sequence[int]] = None,
) -> Tensor:
    """Compute SAS(x) via streaming hooks. Differentiable w.r.t. model
    inputs (e.g. ``inputs_embeds`` when ARA passes continuous embeddings).

    Parameters
    ----------
    extractor : loaded ``AttentionExtractor``
    inputs : dict with ``input_ids`` or ``inputs_embeds`` (and mask)
    output_span : positions whose attention we inspect (query side)
    safety_spans : positions treated as safety-critical (key side).
        Spans are unioned; overlapping positions are deduplicated.
    layers : optional subset of layer indices; defaults to all layers.

    Returns
    -------
    sas : zero-dim Tensor, differentiable, on the model's device.
    """
    safety_positions = sorted({p for s in safety_spans for p in s.indices()})
    output_positions = list(output_span.indices())
    if not safety_positions:
        raise ValueError("empty safety span union")
    if not output_positions:
        raise ValueError("empty output span")

    n_layers = extractor.spec.n_layers if layers is None else len(list(layers))
    acc = SASAccumulator(
        output_positions=output_positions,
        safety_positions=safety_positions,
        n_layers_expected=n_layers,
    )
    extractor.stream_attentions(inputs, acc, layers=layers)
    return acc.finalize()


@dataclass
class SASPerHeadAccumulator:
    """Streaming reducer producing one SAS value per (layer, head).

    Output layout after ``finalize()``: tensor of shape (L, H).
    """

    output_positions: Sequence[int]
    safety_positions: Sequence[int]
    n_layers_expected: int
    # Populated layer-by-layer:
    rows: list = field(default_factory=list)     # list of (layer_idx, Tensor(H,))
    heads_per_layer: int = 0

    def __call__(self, cap) -> None:  # AttentionCapture
        aw = cap.attn_weights                              # (B, H, Sq, Sk)
        if aw.dim() != 4:
            raise ValueError(f"need 4-D, got {tuple(aw.shape)}")
        _, h, _, sk = aw.shape
        out_idx = torch.as_tensor(list(self.output_positions), device=aw.device)
        saf_idx = torch.as_tensor(list(self.safety_positions), device=aw.device)
        sub = aw.index_select(-2, out_idx).index_select(-1, saf_idx)
        # Sum over (output, safety), mean over batch → (H,)
        per_head = sub.sum(dim=(-2, -1)).mean(dim=0)
        if self.heads_per_layer == 0:
            self.heads_per_layer = h
        self.rows.append((cap.layer_idx, per_head))

    def finalize(self) -> Tensor:
        if not self.rows:
            raise RuntimeError("no layers observed")
        # Rows may not arrive in order if a subset of layers is hooked.
        self.rows.sort(key=lambda x: x[0])
        stacked = torch.stack([r[1] for r in self.rows], dim=0)  # (L', H)
        # Normalise by |P_out| to match scalar SAS definition.
        return stacked / float(max(1, len(self.output_positions)))

    def layer_indices(self) -> list[int]:
        return sorted(i for i, _ in self.rows)


def compute_sas_per_layer(
    extractor,
    inputs: dict[str, Tensor],
    *,
    output_span,
    safety_spans,
    layers: Optional[Sequence[int]] = None,
) -> Tensor:
    """Return per-layer SAS as a 1-D tensor of length |layers|.

    Each entry is the average-over-heads SAS for that layer.
    """
    per_head = compute_sas_per_head(
        extractor, inputs,
        output_span=output_span, safety_spans=safety_spans, layers=layers,
    )
    # per_head: (L', H) → mean over heads
    return per_head.mean(dim=-1)


def compute_sas_per_head(
    extractor,
    inputs: dict[str, Tensor],
    *,
    output_span,
    safety_spans,
    layers: Optional[Sequence[int]] = None,
) -> Tensor:
    """Return per-(layer, head) SAS as a 2-D tensor of shape (L', H)."""
    safety_positions = sorted({p for s in safety_spans for p in s.indices()})
    output_positions = list(output_span.indices())
    if not safety_positions or not output_positions:
        raise ValueError("empty spans")
    n_layers = extractor.spec.n_layers if layers is None else len(list(layers))
    acc = SASPerHeadAccumulator(
        output_positions=output_positions,
        safety_positions=safety_positions,
        n_layers_expected=n_layers,
    )
    extractor.stream_attentions(inputs, acc, layers=layers)
    return acc.finalize()


def compute_sas_targeted(
    extractor,
    inputs: dict[str, Tensor],
    *,
    output_span,
    safety_spans,
    target_heads: Sequence[tuple[int, int]],
) -> Tensor:
    """Return SAS scalar averaged over the specified (layer, head) pairs.

    ``target_heads`` is a list of (layer_idx, head_idx). The returned
    tensor is zero-dim, differentiable, suitable for backward().
    """
    if not target_heads:
        raise ValueError("target_heads is empty")
    layer_set = sorted({l for l, _ in target_heads})
    per_head = compute_sas_per_head(
        extractor, inputs,
        output_span=output_span, safety_spans=safety_spans,
        layers=layer_set,
    )
    # per_head is indexed by position in layer_set, not by absolute layer idx.
    layer_pos = {l: i for i, l in enumerate(layer_set)}
    vals = []
    for l, h in target_heads:
        vals.append(per_head[layer_pos[l], h])
    return torch.stack(vals).mean()


def compute_sas_dense(
    attentions: Sequence[Tensor],
    *,
    output_positions: Sequence[int],
    safety_positions: Sequence[int],
) -> Tensor:
    """Non-streaming SAS (from a precomputed tuple of attention tensors).

    Used by unit tests and by visualisations that already materialised
    ``outputs.attentions``. Behaviour must match ``compute_sas`` to
    numerical precision; this function is our reference implementation.
    """
    if len(attentions) == 0:
        raise ValueError("empty attentions tuple")
    # Stack across layers: (L, batch, H, Sq, Sk)
    stacked = torch.stack(list(attentions), dim=0)
    out_idx = torch.as_tensor(list(output_positions), device=stacked.device)
    saf_idx = torch.as_tensor(list(safety_positions), device=stacked.device)
    sub = stacked.index_select(-2, out_idx).index_select(-1, saf_idx)
    # sub: (L, batch, H, |P_out|, |P_safety|)
    L, B, H = sub.shape[0], sub.shape[1], sub.shape[2]
    P_out = len(output_positions)
    total = sub.sum()  # sum over L·B·H·|P_out|·|P_safety|
    # Average over batch, and divide by L·H·|P_out|.
    return total / (L * B * H * P_out)
