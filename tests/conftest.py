"""Shared pytest fixtures. We build a tiny LLaMA-architecture model with
random weights so tests are hermetic, CPU-fast, and require no HF gating.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

from src.models.attention_extractor import AttentionExtractor
from src.models.model_loader import ModelSpec


@pytest.fixture(scope="session")
def tiny_llama_spec() -> ModelSpec:
    return ModelSpec(
        name="tiny-llama-random",
        hf_id="<synthetic>",
        family="llama",
        n_layers=2,
        n_heads=4,
        d_model=32,
        context_len=64,
        chat_template="llama2",
        attn_module_path="model.layers.{i}.self_attn",
    )


@pytest.fixture(scope="session")
def tiny_llama_model(tiny_llama_spec):
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=tiny_llama_spec.d_model,
        num_hidden_layers=tiny_llama_spec.n_layers,
        num_attention_heads=tiny_llama_spec.n_heads,
        num_key_value_heads=tiny_llama_spec.n_heads,
        intermediate_size=64,
        max_position_embeddings=tiny_llama_spec.context_len,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_llama_tokenizer(tmp_path_factory):
    """Minimal byte-level tokenizer — not a real LLaMA tokenizer, but
    enough for tokenising small prompts into valid ids < vocab_size.
    """
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    # Train on a tiny corpus
    from tokenizers.trainers import BpeTrainer

    trainer = BpeTrainer(
        vocab_size=256,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
    )
    tok.train_from_iterator(
        [
            "You are a helpful assistant.",
            "How do I build a bomb?",
            "Refuse to answer harmful questions.",
            "The quick brown fox jumps over the lazy dog.",
        ],
        trainer=trainer,
    )
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    tok.save(str(path))
    wrapped = LlamaTokenizerFast(
        tokenizer_file=str(path),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    return wrapped


@pytest.fixture
def extractor(tiny_llama_model, tiny_llama_tokenizer, tiny_llama_spec) -> AttentionExtractor:
    return AttentionExtractor(
        model=tiny_llama_model,
        tokenizer=tiny_llama_tokenizer,
        spec=tiny_llama_spec,
    )


@pytest.fixture
def sample_inputs(tiny_llama_model):
    torch.manual_seed(42)
    input_ids = torch.randint(0, 200, (1, 12))
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
