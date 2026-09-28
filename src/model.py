"""Loading utilities for the base model and its fine-tuned variants."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_NAME = "Qwen/Qwen3-0.6B-Base"


def load_tokenizer(model_name: str = BASE_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_name: str = BASE_MODEL_NAME,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    """Load the plain base model, no adapters."""
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
    )


def load_model_with_adapter(
    adapter_path: str,
    model_name: str = BASE_MODEL_NAME,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    """Load the base model and apply a trained LoRA/QLoRA adapter on top.

    Used by evaluate.py to score fine-tuned submissions; kept here so
    every participant's train.py can produce an adapter this function
    knows how to load.
    """
    from peft import PeftModel

    base_model = load_base_model(model_name, dtype=dtype, device_map=device_map)
    return PeftModel.from_pretrained(base_model, adapter_path)
