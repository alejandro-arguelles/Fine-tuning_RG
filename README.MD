# Fine-tuning RG: Multi Digit Arithmetic with Supervised Fine Tuning


# The task
To Fine-tune an LLM to perform better at multiple digit multiplication (47 × 812 → 38,164)

What you have to do: take the base model (`Qwen/Qwen3-0.6B-Base`) and fine-tune it with LoRA so it gets a better accuracy than the base model. You can change the config (rank, layers, number of examples, etc) and also the code in `src/train.py`. Then you evaluate it and send the result to the leaderboard

Every run is defined by one `.yaml` file in `configs/`. Copy `configs/example_lora.yaml` and change what you want. The main parts are:

```yaml
user: YOUR_NAME               # your name, it goes in results.json
seed: 0
model_name: Qwen/Qwen3-0.6B-Base

arithmetic:
  ops: mul                    # mul | addsub
  min_digits: 2               # digits per number in train
  max_digits: 5
  test_min_digits: 2          # digits per number in test (6 = more than in train)
  test_max_digits: 6
  n_train: 200                # how many train examples are generated
  n_test: 50

method: lora                  # lora | qlora | full | partial

lora:
  r: 16                       # rank
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "v_proj"]   # which layers get an adapter
  layers:
    mode: all                 # all | first_n | last_n | indices
    # n: 4                    # only for first_n or last_n
    # indices: []             # only for indices, e.g. [0, 1, 26, 27]

data:
  max_train_examples: null    # null = use all the generated examples
  max_seq_length: 640

training:
  output_dir: submissions/YOUR_CONFIG    # where adapter and results.json are saved
  num_train_epochs: 3
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 1
  learning_rate: 2.0e-4
  logging_steps: 10
  bf16: true

eval:
  limit: null                 # null = all the test examples
  batch_size: 50
  temperature: 0              # 0 = greedy
  max_new_tokens: 512
```

About the layers:
- `target_modules` are the projections inside each transformer block where LoRA is added: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. More modules = more parameters to train
- `layers.mode` chooses in which blocks of the model: `all`, only the `first_n`, only the `last_n`, or a list of `indices`
- `r` is the rank of the LoRA. Bigger rank = more parameters. `alpha` is usually 2 × `r`
- If you use `method: partial` you need a `partial:` block with `trainable_name_patterns` (a list of parameter names to train)



## Setup


```bash
pip install -r requirements.txt
```


## Data

The data is made synthetically with `src/arithmetic.py`


- **`mul`** — multiplication, `What is A x B?`, where A and B have the same number of digits. The answer is a step by step reasoning like in [Goat](https://arxiv.org/abs/2305.14201): split B by place value, multiply A by each part, and add the partial results from left to right.
- **`addsub`** — addition and subtraction, `What is A + B?` / `What is A - B?` (in subtraction A ≥ B), with a direct answer.

All answers end with `#### <number>`, and accuracy is an exact match of that number. Train and test use different seeds, and no test question appears in train. The prompt has only one solved example.

The digits for train and test are set separately. For example, training on 2–5 digits and testing on 2–6 digits shows if the model can generalize to a size it has not seen.

## Training

Copy the starter config to edit hyperparameters (`method`, `lora.r`, `target_modules`, `arithmetic` block, `training.output_dir`, etc)

```bash
cp configs/example_lora.yaml configs/YOUR_CONFIG.yaml
```


Train with your config. It generates `submissions/YOUR_CONFIG/{config.yaml, adapter/, results.json}` and evaluates on the config's test set when finished


###################################################################
```bash                                                           #
python -m src.train --config configs/YOUR_CONFIG.yaml             #
```                                                               #
###################################################################





## Evaluation 

Evalate the BASE model in your config 

```bash
python -m src.evaluate --config configs/YOUR_CONFIG.yaml
```


Same but just with 50 examples. All examples may take some time

```bash
python -m src.evaluate --config configs/YOUR_CONFIG.yaml --limit 50
```


Evaluate the base model + your Adapter

```bash
python -m src.evaluate --config configs/YOUR_CONFIG.yaml --adapter submissions/YOUR_CONFIG/adapter
```


## Submit to the leaderboard (Fine-Tuning Arena)

Save the BASE model accuracy in `submissions/_baseline.json` (needed by `submit.py`). Repeat it if you change the task (`mul` ↔ `addsub`) or the test digits, because the baseline is only comparable with the same task

```bash
python -m src.evaluate --config configs/YOUR_CONFIG.yaml --save-baseline
```


Send your submission. It reads `results.json` from that folder (with the `task` and `accuracy` fields) and POSTs it to `https://finetuning-arena.onrender.com/api/submissions`

```bash
python -m src.submit --submission-dir submissions/YOUR_CONFIG --student YOUR_NAME
```


## Leaderboard

https://alejandro-arguelles.github.io/finetuning-arena/


## Reference results

Multiplication, LoRA on the 7 projections and all layers, trained on 2–5 digits and tested on 300 examples of 2–6 digits (60 per length, greedy):

| Run | Rank | Train examples | Accuracy |
|---|---|---|---|
| Base without fine-tuning | — | — | 0.227 |
| `ale_mul_rank4_4000` | 4 | 3.943 | 0.673 |
| `ale_mul_rank16_4000` | 16 | 3.943 | 0.677 |
| `ale_mul_rank8_8000` | 8 | 7.875 | 0.720 |

Accuracy by number of digits (base → `ale_mul_rank8_8000`):

| Digits | Base | Fine-tuned |
|---|---|---|
| 2 | 0.82 | 1.00 |
| 3 | 0.30 | 0.98 |
| 4 | 0.02 | 0.90 |
| 5 | 0.00 | 0.60 |
| 6 (not in train) | 0.00 | 0.12 |

Addition and subtraction (`ale_arith_rank8_5000`, rank 8, 5,000 examples, 2–8 digits, 300 test examples): 0.757 for the base and 0.987 for the fine-tuned.

With 60 examples per length, differences of a few points between runs are not significant.

