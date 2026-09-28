"""Send a training run to the Fine-Tuning Arena leaderboard.

Reads the results.json produced by src/train.py in a submission directory,
translates it into the API's payload format, and POSTs it.

Usage:
    python -m src.evaluate --model Qwen/Qwen3-0.6B-Base --save-baseline  # once, before any submit
    python -m src.submit --submission-dir submissions/mi_equipo --student alice
"""

import argparse
import json
import os
import subprocess

import requests

API_URL = "https://finetuning-arena.onrender.com/api/submissions"
BASELINE_PATH = "submissions/_baseline.json"

METHOD_MAP = {"full": "full_ft", "partial": "partial_ft", "lora": "lora", "qlora": "qlora"}


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_baseline_accuracy(base_model: str, baseline_path: str) -> float:
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"No baseline found at {baseline_path}. Run once before submitting:\n"
            f"  python -m src.evaluate --model {base_model} --save-baseline"
        )
    baseline = load_json(baseline_path)
    if baseline["model"] != base_model:
        raise ValueError(
            f"Baseline at {baseline_path} was computed for '{baseline['model']}', "
            f"not '{base_model}'. Re-run:\n"
            f"  python -m src.evaluate --model {base_model} --save-baseline"
        )
    return baseline["gsm8k_accuracy"]


def build_payload(results: dict, student: str, run_name: str, baseline_accuracy: float) -> dict:
    payload = {
        "student": student,
        "run_name": run_name,
        "base_model": results["base_model"],
        "method": METHOD_MAP[results["method"]],
        "trainable_parameters": results["trainable_parameters"],
        "total_parameters": results["total_parameters"],
        "gsm8k_accuracy": results["gsm8k_accuracy"],
        "baseline_accuracy": baseline_accuracy,
    }
    if results.get("rank") is not None:
        payload["lora_rank"] = results["rank"]
    if results.get("target_modules") is not None:
        payload["target_modules"] = results["target_modules"]
    for key in ("training_examples", "training_time_seconds", "peak_vram_mb", "adapter_bytes"):
        if results.get(key) is not None:
            payload[key] = results[key]

    git_commit = get_git_commit()
    if git_commit:
        payload["git_commit"] = git_commit

    return payload


def submit(payload: dict) -> dict:
    response = requests.post(API_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Submit a run to the Fine-Tuning Arena leaderboard.")
    parser.add_argument("--submission-dir", required=True, help="e.g. submissions/mi_equipo")
    parser.add_argument("--student", required=True, help="Your name/username on the leaderboard")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Defaults to the run_name stored in results.json (or the submission dir name)",
    )
    parser.add_argument("--baseline-path", default=BASELINE_PATH)
    args = parser.parse_args()

    results = load_json(os.path.join(args.submission_dir, "results.json"))
    run_name = args.run_name or results.get("run_name") or os.path.basename(
        args.submission_dir.rstrip("/")
    )
    baseline_accuracy = load_baseline_accuracy(results["base_model"], args.baseline_path)

    payload = build_payload(results, args.student, run_name, baseline_accuracy)
    submitted = submit(payload)

    print(f"Submitted #{submitted['id']} — accuracy_gain={submitted['accuracy_gain']:.4f}")
    print(json.dumps(submitted, indent=2))


if __name__ == "__main__":
    main()
