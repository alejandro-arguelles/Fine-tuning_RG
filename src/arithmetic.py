"""Synthetic multi-digit arithmetic tasks (Goat-style): addition/subtraction
and multiplication with a step-by-step chain of thought.

Produces examples as {"question", "answer"} dicts, with the answer ending in
"#### <number>". Train and test use different seeds; test questions are
removed from train.
"""

import random

TRAIN_SEED = 1234
TEST_SEED = 4321


def _make_example(rng: random.Random, min_digits: int, max_digits: int) -> dict:
    digits = rng.randint(min_digits, max_digits)
    a = rng.randint(10 ** (digits - 1), 10**digits - 1)
    b = rng.randint(10 ** (digits - 1), 10**digits - 1)
    if rng.random() < 0.5:
        op, result = "+", a + b
    else:
        a, b = max(a, b), min(a, b)  # keep results non-negative
        op, result = "-", a - b
    return {
        "question": f"What is {a} {op} {b}?",
        "answer": f"{a} {op} {b} = {result}\n#### {result}",
    }


def format_mul(a: int, b: int) -> str:
    """Goat-style chain of thought for a x b: split b by place value, multiply
    a by each part, then add the partial products left to right."""
    parts = [int(ch) * 10**i for i, ch in enumerate(reversed(str(b))) if ch != "0"]
    parts.reverse()
    prods = [a * p for p in parts]
    text = f"{a} x {b} = "
    if len(parts) > 1:
        text += f"{a} x ({' + '.join(map(str, parts))}) = "
        text += " + ".join(f"{a} x {p}" for p in parts) + " = "
        text += " + ".join(map(str, prods))
        acc = prods[0]
        for i in range(1, len(prods) - 1):
            acc += prods[i]
            text += f" = {acc} + " + " + ".join(map(str, prods[i + 1 :]))
        text += f" = {a * b}"
    else:
        text += str(a * b)
    return text + f"\n#### {a * b}"


def _make_mul_example(rng: random.Random, digits: int) -> dict:
    a = rng.randint(10 ** (digits - 1), 10**digits - 1)
    b = rng.randint(10 ** (digits - 1), 10**digits - 1)
    return {"question": f"What is {a} x {b}?", "answer": format_mul(a, b)}


def generate_multiplication(
    split: str, n: int, min_digits: int, max_digits: int
) -> list[dict]:
    """Digits are cycled so every length is equally represented."""
    seed = TEST_SEED if split == "test" else TRAIN_SEED
    rng = random.Random(seed + 1)
    lengths = list(range(min_digits, max_digits + 1))
    out = [_make_mul_example(rng, lengths[i % len(lengths)]) for i in range(n)]
    if split == "train":
        test_rng = random.Random(TEST_SEED + 1)
        forbidden = {
            _make_mul_example(test_rng, lengths[i % len(lengths)])["question"] for i in range(2000)
        }
        out = [e for e in out if e["question"] not in forbidden]
    return out


def generate_arithmetic(
    split: str, n: int, min_digits: int = 2, max_digits: int = 8
) -> list[dict]:
    test_rng = random.Random(TEST_SEED)
    test = [_make_example(test_rng, min_digits, max_digits) for _ in range(n if split == "test" else 0)]
    if split == "test":
        return test
    # Over-generate on the test seed so we can exclude every test question from train.
    test_rng = random.Random(TEST_SEED)
    forbidden = {_make_example(test_rng, min_digits, max_digits)["question"] for _ in range(5000)}
    rng = random.Random(TRAIN_SEED)
    out = []
    while len(out) < n:
        ex = _make_example(rng, min_digits, max_digits)
        if ex["question"] not in forbidden:
            out.append(ex)
    return out
