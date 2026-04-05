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

def _attention_forward_hook(layer_idx: int, sink: AttentionHookFn):
    """Build a forward hook that parses the tuple returned by a HF
    attention module and hands the attention-weight tensor to ``sink``.

    In HF eager mode (LLaMA/Mistral/Gemma2) the attention module returns:
        (attn_output, attn_weights, past_key_value)          (len 3)
      or
        (attn_output, attn_weights)                          (len 2)
    when ``output_attentions=True`` was requested upstream.
    """

    def hook(module, inputs, output):  # noqa: ANN001  (torch hook signature)
        attn_weights: Optional[Tensor] = None
        if isinstance(output, tuple):
            # Look for a 4-D float tensor whose last two dims match.
            for item in output:
                if (
                    isinstance(item, Tensor)
                    and item.dim() == 4
                    and item.shape[-1] == item.shape[-2]
                ):
                    attn_weights = item
                    break
        if attn_weights is None:
            # Upstream did not request output_attentions. Nothing to do.
            return output
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
    for i, attn in iter_attention_modules(model, spec):
        if i not in want:
            continue
        handles.append(attn.register_forward_hook(_attention_forward_hook(i, sink)))
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

        We tokenise the system prompt in isolation, then find its token
        id subsequence inside ``input_ids``. If the subsequence is not
        found (which can happen when the chat template inserts role
        markers mid-prompt), we fall back to byte-level offsets.
        """
        needle = self.tokenizer(system_prompt, add_special_tokens=False)["input_ids"]
        if not needle:
            raise ValueError("empty system prompt")
        haystack = input_ids[0].tolist()
        L = len(needle)
        for i in range(len(haystack) - L + 1):
            if haystack[i : i + L] == needle:
                return TokenSpan(i, i + L, label="system")
        # Fallback: locate via decoded offsets.
        full = self.tokenizer.decode(haystack)
        pos = full.find(system_prompt)
        if pos < 0:
            raise RuntimeError(
                "system prompt could not be located in tokenised chat sequence"
            )
        # Approximate: walk the tokens, cumulating decoded length.
        cum = 0
        start = end = None
        for i, tok in enumerate(haystack):
            piece = self.tokenizer.decode([tok])
            if start is None and cum + len(piece) > pos:
                start = i
            cum += len(piece)
            if start is not None and cum >= pos + len(system_prompt):
                end = i + 1
                break
        if start is None or end is None:
            raise RuntimeError("fallback span localisation failed")
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
