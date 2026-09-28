"""Shared helpers: reproducibility, answer parsing, metric collection."""

import json
import random
import re

import numpy as np
import torch

_NUMBER_RE = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_predicted_answer(generated_text: str) -> str | None:
    """Pull the final number out of a model completion.

    Looks for a GSM8K-style "#### <number>" tag first; if absent, falls
    back to the last number that appears in the text. Returns None if no
    number is found at all.
    """
    tag_match = re.search(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", generated_text)
    if tag_match:
        return tag_match.group(1).replace(",", "")

    numbers = _NUMBER_RE.findall(generated_text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def answers_match(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    try:
        return float(predicted) == float(gold)
    except ValueError:
        return predicted.strip() == gold.strip()


def count_parameters(model) -> tuple[int, int]:
    """Return (trainable_parameters, total_parameters)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
