"""Configurable SFT trainer: full FT, partial FT, LoRA or QLoRA.

Participants edit/extend this file freely. What must stay fixed for a
submission to be comparable is: the dataset (src/dataset.py), the base
model, and the evaluator (src/evaluate.py) used at the end of this script
to produce gsm8k_accuracy.

Usage:
    python -m src.train --config configs/example_lora.yaml
"""

import argparse
import os
import shutil
import time

import torch
import yaml
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from src.dataset import build_prompt, build_training_target, load_gsm8k
from src.evaluate import run_evaluation
from src.model import load_base_model, load_tokenizer
from src.utils import count_parameters, save_json, set_seed


class GSM8KSFTDataset(Dataset):
    """Tokenizes (prompt, target) pairs and masks the prompt tokens out of
    the loss so the model is only trained to produce the completion.
    """

    def __init__(self, examples, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = build_prompt(example["question"])
        target = " " + build_training_target(example["answer"]) + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + target_ids)[: self.max_seq_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_seq_length]

        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch, pad_token_id: int):
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, labels, attention_mask = [], [], []
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
        labels.append(ex["labels"] + [-100] * pad_len)
        attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


def resolve_lora_layers(layers_cfg: dict | None, num_hidden_layers: int) -> list[int] | None:
    """Turn the config's `lora.layers` block into an explicit list of layer
    indices for PEFT's `layers_to_transform` (None = every layer).
    """
    if not layers_cfg:
        return None
    mode = layers_cfg.get("mode", "all")
    if mode == "all":
        return None
    if mode == "first_n":
        return list(range(layers_cfg["n"]))
    if mode == "last_n":
        n = layers_cfg["n"]
        return list(range(num_hidden_layers - n, num_hidden_layers))
    if mode == "indices":
        return list(layers_cfg["indices"])
    raise ValueError(f"Unknown lora.layers mode: {mode}")


def build_model(cfg: dict, tokenizer):
    """Returns (model, target_layers). target_layers is the list of decoder
    layer indices the LoRA adapters were applied to (None means "all
    layers" or "not applicable", e.g. for full/partial FT).
    """
    method = cfg["method"]
    model_name = cfg["model_name"]

    if method == "lora":
        from peft import LoraConfig, get_peft_model

        model = load_base_model(model_name)
        target_layers = resolve_lora_layers(
            cfg["lora"].get("layers"), model.config.num_hidden_layers
        )
        lora_cfg = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"]["target_modules"],
            layers_to_transform=target_layers,
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, lora_cfg), target_layers

    if method == "qlora":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        target_layers = resolve_lora_layers(
            cfg["lora"].get("layers"), model.config.num_hidden_layers
        )
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"]["target_modules"],
            layers_to_transform=target_layers,
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, lora_cfg), target_layers

    if method == "full":
        return load_base_model(model_name), None

    if method == "partial":
        model = load_base_model(model_name)
        patterns = cfg["partial"]["trainable_name_patterns"]
        for name, param in model.named_parameters():
            param.requires_grad = any(p in name for p in patterns)
        return model, None

    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-0.6B-Base on GSM8K.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 0))
    output_dir = cfg["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(output_dir, "config.yaml"))

    tokenizer = load_tokenizer(cfg["model_name"])
    model, target_layers = build_model(cfg, tokenizer)

    train_examples = list(load_gsm8k("train"))
    max_train_examples = cfg["data"].get("max_train_examples")
    if max_train_examples:
        train_examples = train_examples[:max_train_examples]

    train_dataset = GSM8KSFTDataset(
        train_examples, tokenizer, cfg["data"]["max_seq_length"]
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        logging_steps=cfg["training"]["logging_steps"],
        bf16=cfg["training"].get("bf16", True),
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    trainer.train()
    training_time_seconds = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    trainable_params, total_params = count_parameters(model)

    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    adapter_bytes = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(adapter_dir)
        for f in files
    )

    test_examples = list(load_gsm8k("test"))
    eval_limit = cfg["eval"].get("limit")
    if eval_limit:
        test_examples = test_examples[:eval_limit]
    accuracy, _ = run_evaluation(
        model, tokenizer, test_examples, batch_size=cfg["eval"]["batch_size"]
    )

    results = {
        "team": cfg["team"],
        "run_name": cfg.get("run_name", os.path.basename(output_dir.rstrip("/"))),
        "base_model": cfg["model_name"],
        "method": cfg["method"],
        "rank": cfg.get("lora", {}).get("r"),
        "target_modules": cfg.get("lora", {}).get("target_modules"),
        "target_layers": target_layers,  # None = all layers (or not applicable for full/partial FT)
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "training_examples": len(train_examples),
        "training_time_seconds": training_time_seconds,
        "peak_vram_mb": peak_vram_mb,
        "adapter_bytes": adapter_bytes,
        "gsm8k_accuracy": accuracy,
    }
    save_json(results, os.path.join(output_dir, "results.json"))
    print(results)


if __name__ == "__main__":
    main()
