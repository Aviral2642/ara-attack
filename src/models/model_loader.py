"""Unified loader for LLaMA-2/3, Mistral, and Gemma-2 chat models.

Design goals
------------
* Force ``attn_implementation="eager"`` so that attention weights are
  materialised explicitly and remain differentiable. The flash/SDPA paths
  fuse softmax into the kernel and do NOT return attention weights.
* Always return the HF ``PreTrainedModel`` **together with** its
  tokenizer, chat template, and a structured ``ModelSpec`` describing
  layer counts, head counts, and family-specific attention module paths.
* Support reproducible loading via an explicit ``seed``.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import random
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODELS_CFG = _REPO_ROOT / "configs" / "models.yaml"


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """Static metadata about a chat model.

    Attributes
    ----------
    name : canonical short name (key in models.yaml)
    hf_id : HuggingFace repo id
    family : architecture family (llama / mistral / gemma2)
    n_layers, n_heads, d_model, context_len : architecture
    chat_template : one of {llama2, llama3, mistral, gemma}
    attn_module_path : dotted attribute path from model root to each
        per-layer attention module, with ``{i}`` placeholder for the
        layer index. Used by ``AttentionExtractor``.
    """

    name: str
    hf_id: str
    family: str
    n_layers: int
    n_heads: int
    d_model: int
    context_len: int
    chat_template: str
    attn_module_path: str = "model.layers.{i}.self_attn"


# Family-specific attention paths. Extendable.
_FAMILY_ATTN_PATH = {
    "llama": "model.layers.{i}.self_attn",
    "mistral": "model.layers.{i}.self_attn",
    "gemma2": "model.layers.{i}.self_attn",
    "gpt-oss": "model.layers.{i}.self_attn",     # same transformer-block convention
}


def _load_models_cfg() -> dict:
    with _MODELS_CFG.open() as f:
        return yaml.safe_load(f)


def get_model_spec(name: str) -> ModelSpec:
    cfg = _load_models_cfg()
    if name not in cfg["models"]:
        raise KeyError(
            f"Model '{name}' not in {_MODELS_CFG}. "
            f"Available: {list(cfg['models'])}"
        )
    m = cfg["models"][name]
    family = m["family"]

    def _int_or_auto(val, default=0):
        if val == "auto" or val is None:
            return default
        return int(val)

    return ModelSpec(
        name=name,
        hf_id=m["hf_id"],
        family=family,
        n_layers=_int_or_auto(m["n_layers"]),
        n_heads=_int_or_auto(m["n_heads"]),
        d_model=_int_or_auto(m["d_model"]),
        context_len=_int_or_auto(m["context_len"]),
        chat_template=m.get("chat_template", "auto"),
        attn_module_path=_FAMILY_ATTN_PATH.get(family, "model.layers.{i}.self_attn"),
    )


def _resolve_device_map(device_map: Optional[str]) -> str:
    if device_map is not None:
        return device_map
    if torch.cuda.is_available():
        return "auto"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype: Optional[str | torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype in mapping:
        return mapping[dtype]
    # bf16 requires hardware support (A100+, Apple Silicon M2+)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(
    name: str,
    *,
    device_map: Optional[str] = None,
    torch_dtype: Optional[str | torch.dtype] = "bfloat16",
    hf_token: Optional[str] = None,
    seed: int = 0,
    trust_remote_code: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, ModelSpec]:
    """Load a chat model and its tokenizer with eager attention.

    Parameters
    ----------
    name : canonical short name from models.yaml
    device_map : passed to ``from_pretrained``. Defaults to CUDA/MPS/CPU auto.
    torch_dtype : bfloat16 on supported GPUs, fp32 fallback.
    hf_token : overrides ``HF_TOKEN`` env var for gated models.
    seed : RNG seed for reproducibility.
    trust_remote_code : NEVER True for published models; opt-in only.

    Returns
    -------
    (model, tokenizer, spec)
    """
    seed_everything(seed)
    spec = get_model_spec(name)
    token = hf_token or os.environ.get("HF_TOKEN")

    # Per-model trust_remote_code override (needed for gpt-oss).
    cfg_all = _load_models_cfg()
    model_cfg = cfg_all["models"].get(name, {})
    if model_cfg.get("trust_remote_code", False):
        trust_remote_code = True

    device_map_resolved = _resolve_device_map(device_map)
    dtype_resolved = _resolve_dtype(torch_dtype)

    log.info(
        "loading %s (%s) dtype=%s device_map=%s",
        name,
        spec.hf_id,
        dtype_resolved,
        device_map_resolved,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id,
        token=token,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        torch_dtype=dtype_resolved,
        device_map=device_map_resolved,
        token=token,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        attn_implementation="eager",   # REQUIRED for output_attentions + grads
    )
    model.eval()

    # Auto-fill ModelSpec fields from the loaded model's config when
    # the YAML entry uses "auto" (e.g. gpt-oss-20b).
    cfg = model.config
    if spec.n_layers == 0:
        spec = dataclasses.replace(
            spec,
            n_layers=getattr(cfg, "num_hidden_layers", spec.n_layers),
            n_heads=getattr(cfg, "num_attention_heads", spec.n_heads),
            d_model=getattr(cfg, "hidden_size", spec.d_model),
            context_len=getattr(cfg, "max_position_embeddings",
                        getattr(cfg, "max_seq_len", spec.context_len)),
        )
        log.info("auto-detected spec: n_layers=%d n_heads=%d d_model=%d ctx=%d",
                 spec.n_layers, spec.n_heads, spec.d_model, spec.context_len)

    return model, tokenizer, spec


# Convenience alias matching the short-name convention used in docs
# and verification scripts.
load_model = load_model_and_tokenizer


def get_attention_module(model: PreTrainedModel, spec: ModelSpec, layer_idx: int):
    """Traverse the dotted attribute path to reach a per-layer attention module."""
    path = spec.attn_module_path.format(i=layer_idx)
    obj = model
    for attr in path.split("."):
        if attr.isdigit():
            obj = obj[int(attr)]
        else:
            obj = getattr(obj, attr)
    return obj


def iter_attention_modules(model: PreTrainedModel, spec: ModelSpec):
    """Yield (layer_idx, attention_module) for every transformer layer."""
    for i in range(spec.n_layers):
        yield i, get_attention_module(model, spec, i)
