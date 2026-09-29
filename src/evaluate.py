"""Official arithmetic evaluator.

Fixed prompt, fixed decoding, fixed answer extraction — this script must
stay identical for every submission so accuracy numbers are comparable.
Do not modify it as part of a fine-tuning submission.

Usage:
    python -m src.evaluate --config configs/ale_mul_rank8_8000.yaml
    python -m src.evaluate --config configs/ale_mul_rank8_8000.yaml --adapter path/to/adapter
"""

import argparse
import os
import time

import torch
import yaml
from tqdm import tqdm

from src.dataset import build_prompt, configure, extract_gold_answer, load_split
from src.model import BASE_MODEL_NAME, load_base_model, load_model_with_adapter, load_tokenizer
from src.utils import answers_match, extract_predicted_answer, save_json, set_seed

BASELINE_PATH = "submissions/_baseline.json"

MAX_NEW_TOKENS = 512


@torch.no_grad()
def run_evaluation(
    model,
    tokenizer,
    examples,
    batch_size: int = 8,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0,
):
    model.eval()
    tokenizer.padding_side = "left"

    num_correct = 0
    records = []

    for start in tqdm(range(0, len(examples), batch_size), desc="Evaluating"):
        batch = examples[start : start + batch_size]
        prompts = [build_prompt(ex["question"]) for ex in batch]
        golds = [extract_gold_answer(ex["answer"]) for ex in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        if temperature > 0:
            sampling = dict(do_sample=True, temperature=temperature, top_k=20, top_p=0.8)
        else:
            sampling = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **sampling,
        )

        completions = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        for question, gold, completion in zip(
            [ex["question"] for ex in batch], golds, completions
        ):
            completion = completion.split("Question:")[0]
            predicted = extract_predicted_answer(completion)
            correct = answers_match(predicted, gold)
            num_correct += int(correct)
            records.append(
                {
                    "question": question,
                    "gold": gold,
                    "predicted": predicted,
                    "correct": correct,
                    "completion": completion,
                }
            )

    accuracy = num_correct / len(examples)
    return accuracy, records


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on synthetic arithmetic.")
    parser.add_argument(
        "--config",
        required=True,
        help="Training config whose `arithmetic:` block defines the test split",
    )
    parser.add_argument("--model", default=BASE_MODEL_NAME)
    parser.add_argument("--adapter", default=None, help="Path to a LoRA/QLoRA adapter")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate on a subset only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0, help="0 = greedy")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help=f"Save this run's accuracy to {BASELINE_PATH}, used by src/submit.py as baseline_accuracy",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    with open(args.config) as f:
        configure(**yaml.safe_load(f).get("arithmetic", {}))

    tokenizer = load_tokenizer(args.model)
    if args.adapter:
        model = load_model_with_adapter(args.adapter, model_name=args.model)
    else:
        model = load_base_model(args.model)

    examples = load_split("test")
    if args.limit:
        examples = examples[: args.limit]

    start_time = time.time()
    accuracy, records = run_evaluation(
        model,
        tokenizer,
        examples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    elapsed = time.time() - start_time

    print(f"\nAccuracy: {accuracy:.4f} ({sum(r['correct'] for r in records)}/{len(records)})")
    print(f"Evaluation time: {elapsed:.1f}s")

    if args.save_baseline:
        os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
        save_json(
            {"model": args.model, "task": "arithmetic", "accuracy": accuracy}, BASELINE_PATH
        )
        print(f"Saved baseline accuracy to {BASELINE_PATH}")


if __name__ == "__main__":
    main()
