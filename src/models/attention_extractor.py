"""Extract per-layer, per-head attention weights from HF chat models.

Two extraction paths are provided:

1. **Output path** — the standard ``output_attentions=True`` call. Safe
   for short sequences; stores a tuple of ``(batch, n_heads, seq, seq)``
   tensors in memory. Gradients propagate back through the softmax in
   eager mode, which is why ``model_loader.load_model_and_tokenizer``
   forces ``attn_implementation="eager"``.

2. **Hook path** — registers forward hooks on each attention module and
   consumes attention weights layer-by-layer via a user-supplied
   callable. This avoids simultaneously materialising L layers worth of
   attention tensors, which is the only practical option for long
   contexts on 7B+ models.

The SAS metric (``safety_attention_score.py``) uses the hook path so
that we never hold more than one layer's attention tensor at a time,
while still retaining the computation graph needed for ARA's gradient
step.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .model_loader import ModelSpec, iter_attention_modules

log = logging.getLogger(__name__)


def _map_normalised_to_original(original: str, norm_idx: int) -> int:
    """Map an index in whitespace-normalised text back to the original.

    Used by ``locate_system_span`` for its final fallback. We walk
    ``original`` character by character; every run of whitespace in
    the original contributes exactly one char in the normalised form.
    """
    import re
    out = 0
    i = 0
    while i < len(original) and out < norm_idx:
        if original[i].isspace():
            # collapse a whitespace run into a single char in norm.
            out += 1
            while i < len(original) and original[i].isspace():
                i += 1
        else:
            out += 1
            i += 1
    return i


@dataclass
class TokenSpan:
    """Half-open range [start, end) of token positions in a tokenised prompt."""

    start: int
    end: int
    label: str = ""

    def __post_init__(self):
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"bad span [{self.start}, {self.end})")

    def indices(self) -> list[int]:
        return list(range(self.start, self.end))

    def __len__(self) -> int:
        return self.end - self.start


@dataclass
class AttentionCapture:
    """Container yielded by hook path — one tensor per processed layer."""

    layer_idx: int
    attn_weights: Tensor  # shape: (batch, n_heads, seq_q, seq_k)


AttentionHookFn = Callable[[AttentionCapture], None]


# ---------------------------------------------------------------------------
# Hook path
# ---------------------------------------------------------------------------

def _expand_gqa_weights(attn_weights: Tensor, n_q_heads: int) -> Tensor:
    """Expand grouped-query attention weights to full per-head shape.

    If the model uses GQA (n_kv_heads < n_q_heads), the HF eager path
    *usually* returns weights already expanded to ``(B, n_q_heads, S, S)``.
    Some architectures (GPT-OSS, custom impls) may return the raw
    ``(B, n_kv_heads, S, S)`` tensor instead.  This helper repeats the
    KV heads to match ``n_q_heads`` so downstream code always sees the
    same shape.
    """
    n_kv = attn_weights.shape[1]
    if n_kv == n_q_heads or n_q_heads == 0:
        return attn_weights
    if n_q_heads % n_kv != 0:
        # Cannot cleanly expand; return as-is (SAS will adapt).
        return attn_weights
    repeats = n_q_heads // n_kv
    return attn_weights.repeat_interleave(repeats, dim=1)


def _attention_forward_hook(layer_idx: int, sink: AttentionHookFn,
                            n_q_heads: int = 0):
    """Build a forward hook that parses the tuple returned by a HF
    attention module and hands the attention-weight tensor to ``sink``.

    In HF eager mode (LLaMA/Mistral/Gemma2/GPT-OSS) the attention
    module returns:
        (attn_output, attn_weights, past_key_value)          (len 3)
      or
        (attn_output, attn_weights)                          (len 2)
    when ``output_attentions=True`` was requested upstream.

    For GQA models the weights may have fewer heads than queries; we
    expand via ``_expand_gqa_weights``.

    For sparse / locally-banded layers the weights tensor may be
    smaller along the key dimension or contain explicit zeros for
    out-of-band positions.  Both cases are handled transparently —
    zero entries simply contribute 0 to downstream SAS sums, and
    shorter key dims are zero-padded to (S, S) so indexing is uniform.
    """

    def hook(module, inputs, output):  # noqa: ANN001
        attn_weights: Optional[Tensor] = None
        if isinstance(output, tuple):
            for item in output:
                if (
                    isinstance(item, Tensor)
                    and item.dim() == 4
                    and item.shape[-1] == item.shape[-2]
                ):
                    attn_weights = item
                    break
            # Fallback: also accept 4-D where last two dims differ (banded
            # sparse attention returns (B, H, Sq, window) instead of (B, H, Sq, Sk)).
            if attn_weights is None:
                for item in output:
                    if isinstance(item, Tensor) and item.dim() == 4:
                        attn_weights = item
                        break
        if attn_weights is None:
            return output

        # --- GQA expansion ------------------------------------------------
        attn_weights = _expand_gqa_weights(attn_weights, n_q_heads)

        # --- Sparse / banded: pad to (B, H, S, S) if key dim < query dim --
        B, H, Sq, Sk = attn_weights.shape
        if Sk < Sq:
            pad = torch.zeros(B, H, Sq, Sq - Sk,
                              device=attn_weights.device,
                              dtype=attn_weights.dtype)
            attn_weights = torch.cat([attn_weights, pad], dim=-1)

        sink(AttentionCapture(layer_idx=layer_idx, attn_weights=attn_weights))
        return output

    return hook


@contextlib.contextmanager
def attention_hooks(
    model: PreTrainedModel,
    spec: ModelSpec,
    sink: AttentionHookFn,
    *,
    layers: Optional[Sequence[int]] = None,
):
    """Context manager that registers hooks on the requested layers and
    removes them on exit. Safe under exceptions.

    Usage
    -----
    >>> captures = []
    >>> with attention_hooks(model, spec, captures.append):
    ...     out = model(**batch, output_attentions=True)
    """
    handles = []
    want = set(range(spec.n_layers) if layers is None else layers)
    n_q = spec.n_heads  # pass to hook for GQA expansion
    for i, attn in iter_attention_modules(model, spec):
        if i not in want:
            continue
        handles.append(
            attn.register_forward_hook(_attention_forward_hook(i, sink, n_q_heads=n_q))
        )
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# High-level extractor
# ---------------------------------------------------------------------------

@dataclass
class AttentionExtractor:
    """Thin facade over the model+tokenizer providing the operations
    that ARA + SAS need.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    spec: ModelSpec

    # --------------- tokenisation -----------------------------------------
    def tokenize_chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        *,
        add_generation_prompt: bool = True,
    ) -> dict[str, Tensor]:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        enc = self.tokenizer(text, return_tensors="pt")
        return {k: v.to(self._device()) for k, v in enc.items()}

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    # --------------- span localisation ------------------------------------
    def locate_system_span(
        self,
        input_ids: Tensor,
        system_prompt: str,
    ) -> TokenSpan:
        """Return [start, end) of the system-prompt tokens within a
        tokenised chat sequence.

        Chat templates (LLaMA-2's ``[INST]<<SYS>>``, LLaMA-3's
        ``<|start_header_id|>system<|end_header_id|>``, Mistral's,
        Gemma's) wrap the system prompt with role markers. BPE encodes
        leading whitespace, so tokenising the system prompt in isolation
        rarely produces the same token id sequence as in context. We
        therefore use a three-stage strategy:

        1. direct subsequence match (cheap; may succeed for simple cases);
        2. subsequence match after trying common leading prefixes
           (``\"\"``, ``\" \"``, ``\"\\n\"``, ``\"\\n\\n\"``);
        3. char-level alignment: decode each token individually,
           accumulate character spans, then locate the system prompt
           in the concatenated text and map back to token indices.
        """
        if not system_prompt:
            raise ValueError("empty system prompt")
        haystack = input_ids[0].tolist()

        # (1) direct subsequence
        needle = self.tokenizer(system_prompt, add_special_tokens=False)["input_ids"]
        if needle:
            L = len(needle)
            for i in range(len(haystack) - L + 1):
                if haystack[i : i + L] == needle:
                    return TokenSpan(i, i + L, label="system")

        # (2) leading-prefix variants (handles BPE context sensitivity)
        for prefix in (" ", "\n", "\n\n", "  "):
            variant = self.tokenizer(
                prefix + system_prompt, add_special_tokens=False
            )["input_ids"]
            if not variant:
                continue
            Lv = len(variant)
            for i in range(len(haystack) - Lv + 1):
                if haystack[i : i + Lv] == variant:
                    # Drop the first variant token (the prefix) from the span.
                    return TokenSpan(i + 1, i + Lv, label="system")

        # (3) char-level alignment — decode each token and build offsets.
        pieces = [
            self.tokenizer.decode([tid], skip_special_tokens=False)
            for tid in haystack
        ]
        full_text = "".join(pieces)
        # Try the prompt both verbatim and with common surrounding whitespace.
        pos = -1
        for probe in (system_prompt, " " + system_prompt, "\n" + system_prompt,
                      "\n\n" + system_prompt):
            j = full_text.find(probe)
            if j >= 0:
                # Skip over any prefix we added in the probe.
                pos = j + (len(probe) - len(system_prompt))
                break
        if pos < 0:
            # Try whitespace-normalised fallback: both sides collapsed.
            import re
            norm_text = re.sub(r"\s+", " ", full_text)
            norm_sys = re.sub(r"\s+", " ", system_prompt).strip()
            j = norm_text.find(norm_sys)
            if j < 0:
                raise RuntimeError(
                    "system prompt could not be located in tokenised chat sequence; "
                    "inspect the chat template render for "
                    f"family={self.spec.family!r}"
                )
            # Map normalised index back to original text by char walk.
            orig_j = _map_normalised_to_original(full_text, j)
            pos = orig_j
        end_pos = pos + len(system_prompt)

        # Map character offsets to token indices.
        cursor = 0
        start = end = None
        for i, piece in enumerate(pieces):
            piece_start, piece_end = cursor, cursor + len(piece)
            if start is None and piece_end > pos:
                start = i
            if start is not None and piece_start >= end_pos:
                end = i
                break
            cursor = piece_end
        if end is None:
            end = len(pieces)
        if start is None or end <= start:
            raise RuntimeError("char-level span localisation failed")
        return TokenSpan(start, end, label="system")

    def locate_output_span(self, input_ids: Tensor) -> TokenSpan:
        """The 'output-generating' positions per spec are the last-token
        positions that produce the next-token prediction. For prefill we
        treat all positions beyond the user prompt as output-generating
        candidates; minimally, the single final position.
        """
        n = int(input_ids.shape[1])
        return TokenSpan(n - 1, n, label="output")

    def locate_refusal_trigger_span(
        self,
        input_ids: Tensor,
        triggers: Iterable[str] = ("refuse", "cannot", "sorry", "illegal", "harmful"),
    ) -> list[TokenSpan]:
        """Locate positions whose decoded token overlaps any trigger word.
        Used to enlarge the set of safety-critical positions when the
        system prompt is short or absent.
        """
        haystack = input_ids[0].tolist()
        out: list[TokenSpan] = []
        for i, tok in enumerate(haystack):
            piece = self.tokenizer.decode([tok]).strip().lower()
            if piece and any(t in piece for t in triggers):
                out.append(TokenSpan(i, i + 1, label=f"trigger:{piece}"))
        return out

    # --------------- full-tensor extraction (short sequences only) --------
    def extract_all_attentions(
        self,
        inputs: dict[str, Tensor],
    ) -> tuple[Tensor, ...]:
        """Forward pass with ``output_attentions=True``. Returns tuple of
        L tensors, each ``(batch, n_heads, seq, seq)``. Memory:
        ``O(L * H * S^2)``. Only viable for S <= ~2048 on a 7B model.
        """
        outputs = self.model(**inputs, output_attentions=True, use_cache=False)
        if outputs.attentions is None:
            raise RuntimeError(
                "model did not return attentions; "
                "ensure attn_implementation='eager'"
            )
        return outputs.attentions

    # --------------- streaming extraction (long sequences) ----------------
    def stream_attentions(
        self,
        inputs: dict[str, Tensor],
        consumer: AttentionHookFn,
        *,
        layers: Optional[Sequence[int]] = None,
    ) -> None:
        """Forward pass that delivers each layer's attention tensor to
        ``consumer`` the moment it is computed, then discards the
        reference. Upstream ``consumer`` is expected to reduce the
        tensor to a scalar (e.g. SAS contribution) so the tensor itself
        can be garbage-collected after the hook returns.
        """
        with attention_hooks(self.model, self.spec, consumer, layers=layers):
            self.model(**inputs, output_attentions=True, use_cache=False)
