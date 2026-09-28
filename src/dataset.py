"""GSM8K loading and prompt formatting.

This module is shared by train.py and evaluate.py so that every
participant works from the exact same data and prompt format.
"""

import re

from datasets import load_dataset

GSM8K_DATASET = "openai/gsm8k"
GSM8K_SUBSET = "main"

# Fixed 4-shot exemplars (taken from the GSM8K train split) used to prompt
# the base model into producing step-by-step reasoning followed by a final
# "#### <number>" answer, matching GSM8K's own answer format.
FEWSHOT_EXEMPLARS = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72",
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $10.\n#### 10",
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": "Betty has 100/2 = $50.\nHer grandparents gave her 15*2 = $30.\nIn total Betty has 50+15+30 = $95.\nBetty still needs 100-95 = $5.\n#### 5",
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": "James writes 3*2 = 6 pages each time.\nHe writes 6*2 = 12 pages a week.\nIn a year he writes 12*52 = 624 pages.\n#### 624",
    },
]

_ANSWER_TAG_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def load_gsm8k(split: str):
    """Load a GSM8K split ("train" or "test"; "test" is what the official
    evaluator scores on)."""
    return load_dataset(GSM8K_DATASET, GSM8K_SUBSET, split=split)


def extract_gold_answer(answer_text: str) -> str:
    """Extract the final numeric answer from a GSM8K reference solution
    (the part after "####"), stripping thousands separators.
    """
    match = _ANSWER_TAG_RE.search(answer_text)
    if not match:
        raise ValueError(f"Could not find '#### <answer>' in: {answer_text!r}")
    return match.group(1).replace(",", "")


def build_fewshot_prefix() -> str:
    """Build the fixed few-shot prefix prepended to every question."""
    blocks = [
        f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in FEWSHOT_EXEMPLARS
    ]
    return "\n\n".join(blocks)


_CALC_ANNOTATION_RE = re.compile(r"<<[^>]*>>")


def build_training_target(answer_text: str) -> str:
    """Clean a raw GSM8K reference solution into the completion the model
    is trained to produce: strip the "<<calculator>>" annotations so the
    style matches the few-shot exemplars used in build_prompt().
    """
    return _CALC_ANNOTATION_RE.sub("", answer_text).strip()


def build_prompt(question: str) -> str:
    """Build the fixed evaluation prompt for a single GSM8K question.

    Same prompt format must be used at training and evaluation time so
    results are comparable across submissions.
    """
    prefix = build_fewshot_prefix()
    return f"{prefix}\n\nQuestion: {question}\nAnswer:"
