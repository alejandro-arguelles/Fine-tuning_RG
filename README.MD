# Fine-tuning RG — Aritmética multi-dígito con SFT

Todos parten del mismo modelo (`Qwen/Qwen3-0.6B-Base`) y la misma tarea: aritmética sintética de varios dígitos. Cada participante elige su estrategia de fine-tuning (full FT, partial FT, LoRA, QLoRA, distintos rangos/módulos) y se compara accuracy vs coste.

## La tarea

Los datos se generan con un script (`src/arithmetic.py`), no hay que descargar nada. Dos variantes, elegidas con `arithmetic.ops` en el config:

- **`mul`** — multiplicación `What is A x B?`, con A y B del mismo número de dígitos. La respuesta sigue una cadena de razonamiento estilo [Goat](https://arxiv.org/abs/2305.14201): descompone B por valor posicional, multiplica A por cada parte y suma los parciales de izquierda a derecha.
- **`addsub`** — suma y resta `What is A + B?` / `What is A - B?` (en la resta A ≥ B), con respuesta directa.

Todas las respuestas terminan en `#### <número>`, y la accuracy es coincidencia exacta de ese número. Train y test usan semillas distintas y ninguna pregunta del test aparece en train. El prompt lleva un único ejemplo resuelto.

Los dígitos de train y de test se controlan por separado. Por ejemplo, entrenar en 2–5 dígitos y evaluar en 2–6 mide si el modelo generaliza a un tamaño que no ha visto.

## Setup

```bash
pip install -r requirements.txt
```
Instala dependencias (torch, transformers, peft, bitsandbytes, etc.).

## Evaluar (fijo para todos, no modificar `src/evaluate.py`)

```bash
python -m src.evaluate --config configs/ale_mul_rank8_8000.yaml
```
Evalúa el modelo base en el test definido por el bloque `arithmetic:` de ese config (greedy, accuracy exact-match).

```bash
python -m src.evaluate --config configs/ale_mul_rank8_8000.yaml --limit 50
```
Igual, pero solo en 50 ejemplos (prueba rápida).

```bash
python -m src.evaluate --config configs/ale_mul_rank8_8000.yaml --adapter submissions/ale_mul_rank8_8000/adapter
```
Evalúa el modelo base + un adapter LoRA/QLoRA entrenado.

## Entrenar (editable, cada uno hace el suyo)

```bash
cp configs/example_lora.yaml configs/mi_usuario.yaml
```
Copia el config de partida para editar hiperparámetros (`method`, `lora.r`, `target_modules`, bloque `arithmetic`, `training.output_dir`, etc).

```bash
python -m src.train --config configs/mi_usuario.yaml
```
Entrena según el config y genera `submissions/mi_usuario/{config.yaml, adapter/, results.json}`. Al terminar evalúa sobre el test del config.

Si te quedas sin memoria de GPU con secuencias largas (multiplicación de 5–6 dígitos), baja `per_device_train_batch_size` y sube `gradient_accumulation_steps` para mantener el batch efectivo.

## Enviar al leaderboard (Fine-Tuning Arena)

```bash
python -m src.evaluate --config configs/mi_usuario.yaml --save-baseline
```
Guarda el accuracy del modelo base en `submissions/_baseline.json` (lo necesita `submit.py`). Hay que repetirlo si cambias de tarea (`mul` ↔ `addsub`) o de dígitos de test, porque el baseline solo es comparable con la misma tarea.

```bash
python -m src.submit --submission-dir submissions/mi_usuario --student mi_nombre
```
Lee `results.json` de esa carpeta (con los campos `task` y `accuracy`) y lo envía por POST a `https://finetuning-arena.onrender.com/api/submissions`.

## Resultados de referencia

Multiplicación, LoRA en las 7 proyecciones y todas las capas, entrenada en 2–5 dígitos y evaluada en 300 ejemplos de 2–6 dígitos (60 por longitud, greedy):

| Run | Rank | Ejemplos train | Accuracy |
|---|---|---|---|
| Base sin fine-tuning | — | — | 0.227 |
| `ale_mul_rank4_4000` | 4 | 3.943 | 0.673 |
| `ale_mul_rank16_4000` | 16 | 3.943 | 0.677 |
| `ale_mul_rank8_8000` | 8 | 7.875 | 0.720 |

Accuracy por número de dígitos (base → `ale_mul_rank8_8000`):

| Dígitos | Base | Fine-tuneado |
|---|---|---|
| 2 | 0.82 | 1.00 |
| 3 | 0.30 | 0.98 |
| 4 | 0.02 | 0.90 |
| 5 | 0.00 | 0.60 |
| 6 (fuera de train) | 0.00 | 0.12 |

Suma y resta (`ale_arith_rank8_5000`, rank 8, 5.000 ejemplos, 2–8 dígitos, 300 ejemplos de test): 0.757 el base y 0.987 el fine-tuneado.

Con 60 ejemplos por longitud, las diferencias de pocos puntos entre runs no son significativas.

## Estructura

```
src/arithmetic.py  # generador de datos sintéticos (suma/resta y multiplicación)
src/dataset.py     # ajustes de la tarea + prompt fijo — no modificar
src/model.py       # carga del modelo base y de adapters — no modificar
src/evaluate.py    # evaluador oficial — no modificar
src/utils.py       # helpers (seed, conteo de parámetros, JSON) — no modificar
src/train.py       # entrenamiento — cada participante lo edita/extiende
src/submit.py      # envío al leaderboard
configs/           # configs de entrenamiento (example_*.yaml y runs de referencia)
submissions/       # salida de cada run (adapter, config, results.json)
```

`Structured Synthetic Reasoning Data for Arithmetic Fine-Tuning of Small Language Models.pdf` es el artículo de referencia incluido en el repositorio.
