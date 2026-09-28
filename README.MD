# Fine-tuning RG — GSM8K SFT Challenge

Todos parten del mismo modelo (`Qwen/Qwen3-0.6B-Base`) y la misma tarea (GSM8K). Cada equipo elige su estrategia de fine-tuning (full FT, partial FT, LoRA, QLoRA, distintos rangos/módulos) y se compara accuracy vs coste.

## Setup

```bash
pip install -r requirements.txt
```
Instala dependencias (torch, transformers, datasets, peft, bitsandbytes).

## Evaluar (fijo para todos, no modificar `src/evaluate.py`)

```bash
python -m src.evaluate --model Qwen/Qwen3-0.6B-Base
```
Evalúa el modelo base en el test set completo de GSM8K (accuracy exact-match).

```bash
python -m src.evaluate --model Qwen/Qwen3-0.6B-Base --limit 50
```
Igual, pero solo en 50 ejemplos (prueba rápida).

```bash
python -m src.evaluate --model Qwen/Qwen3-0.6B-Base --adapter submissions/mi_equipo/adapter
```
Evalúa el modelo base + un adapter LoRA/QLoRA entrenado.

## Entrenar (editable, cada equipo hace el suyo)

```bash
cp configs/example_lora.yaml configs/mi_equipo.yaml
```
Copia el config de partida para editar hiperparámetros (`method`, `lora.r`, `target_modules`, `training.output_dir`, etc).

```bash
python -m src.train --config configs/mi_equipo.yaml
```
Entrena según el config y genera `submissions/mi_equipo/{config.yaml, adapter/, results.json}`.

## Enviar al leaderboard (Fine-Tuning Arena)

```bash
python -m src.evaluate --model Qwen/Qwen3-0.6B-Base --save-baseline
```
Corre una sola vez: guarda el accuracy del modelo base en `submissions/_baseline.json` (lo necesita `submit.py`).

```bash
python -m src.submit --submission-dir submissions/mi_equipo --student mi_nombre
```
Lee `results.json` de esa carpeta y lo envía por POST a `https://finetuning-arena.onrender.com/api/submissions`.

## Estructura

```
src/dataset.py    # carga GSM8K + prompt fijo — no modificar
src/model.py      # carga del modelo base y de adapters — no modificar
src/evaluate.py   # evaluador oficial — no modificar
src/utils.py      # helpers (seed, conteo de parámetros, JSON) — no modificar
src/train.py      # entrenamiento — cada equipo lo edita/extiende
configs/*.yaml    # un config por experimento
submissions/<equipo>/  # config.yaml + adapter/ + results.json de cada submission
leaderboard/      # sitio estático que lee submissions/*/results.json (pendiente)
```
