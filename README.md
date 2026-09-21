# TP Integrador - Redes Neuronales (CNN + RNN)

## Estructura del repo

```
src/     pipelines de entrenamiento (CNN, RNN, análisis de gradientes, figuras) y evaluación
models/  modelos entrenados listos para inferencia (.keras)
tools/   generador del informe .docx (opcional, requiere Node)
```

Los datasets (CIFAR-10, Jena Climate) se descargan solos la primera vez que
corrés un script; no hace falta bajar nada a mano. Las salidas se escriben en
`artifacts/`, que no está versionado — cada uno genera las suyas al correr los
comandos de abajo.

## Instalación

Requiere **Python 3.9 a 3.12** (probado con 3.12): son las versiones que
soporta `tensorflow>=2.16` vía pip. Con Python 3.13+ la instalación de
tensorflow falla.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Tiempos estimados

Todo corre en CPU (TensorFlow no tiene soporte nativo de GPU en Windows sin
WSL2). Referencia tomada en una notebook de gama media-alta; en una compu más
floja puede tardar bastante más.

| Comando | Duración aprox. |
|---|---|
| CNN completa (5 folds, 60 épocas) | ~2 a 2.5 horas |
| CNN sin regularización (1 fold, 30 épocas) | ~10-15 min |
| RNN completa (3 celdas, con early stopping) | ~15-20 min |
| Gradientes + figuras | segundos a un par de minutos |

**Si solo querés verificar que todo corre sin esperar horas**, usá la
combinación reducida de abajo (CNN de 1 fold + RNN de una sola celda): entrena
modelos reales de punta a punta y debería quedar bien por debajo de 1 hora
incluso en hardware modesto. No reproduce los números exactos del informe
(esos salen de la corrida completa), solo confirma que el pipeline funciona.

```bash
python src/cnn_pipeline.py --max-folds 1 --epochs 15
python src/rnn_pipeline.py --cells gru --epochs 15
```

## Modelos exportados

`models/` contiene los modelos finales del informe, versionados junto con el
código. Cada `.keras` incluye arquitectura y pesos, así que se cargan directo:

| Archivo | Modelo |
|---|---|
| `cnn_best.keras` | CNN de CIFAR-10 (mejor pliegue, regularizada) |
| `rnn_simple_rnn.keras` | SimpleRNN, Jena Climate (ventana 120 h, horizonte 24 h) |
| `rnn_gru.keras` | GRU, misma configuración |
| `rnn_lstm.keras` | LSTM, misma configuración |

Inferencia directa:

```python
from tensorflow import keras
model = keras.models.load_model("models/cnn_best.keras", compile=False)
probs = model.predict(images)  # images: (N, 32, 32, 3), valores 0-255
```

Las RNN esperan ventanas de forma `(N, 120, 18)` normalizadas con la media y el
desvío del tramo de entrenamiento; `src/rnn_pipeline.py` implementa ese
preprocesamiento.

Evaluación sobre el conjunto de prueba, sin reentrenar:

```bash
python src/evaluate.py
python src/evaluate.py --task cnn
python src/evaluate.py --task rnn --output eval.json
```

Reproduce las métricas del informe (CNN: exactitud 0,8634 y F1 macro 0,8627;
RNN: MAE de prueba 2,3253 / 2,1958 / 2,2461 °C para SimpleRNN / GRU / LSTM,
contra 2,5079 °C del baseline ingenuo).

## Módulo CNN — CIFAR-10

```bash
python src/cnn_pipeline.py --folds 5 --epochs 60
```

Prueba rápida (un solo fold):

```bash
python src/cnn_pipeline.py --folds 5 --max-folds 1 --epochs 15
```

Corrida de contraste para el análisis de sobreajuste:

```bash
python src/cnn_pipeline.py --no-regularization --max-folds 1 --outdir artifacts/cnn_plain
```

Salidas en `artifacts/cnn/`: pesos por fold, `cnn_best.keras`, `cnn_summary.json`
(accuracy media ± desvío entre folds, brecha de generalización, reporte sobre test),
`cnn_histories.json` y `confusion_matrix.csv`.

## Módulo RNN — Jena Climate

```bash
python src/rnn_pipeline.py --window 120 --horizon 24
```

Un solo modelo:

```bash
python src/rnn_pipeline.py --cells gru --epochs 20
```

Salidas en `artifacts/rnn/`: pesos y modelo por celda, `rnn_summary.json`
(MAE en °C por partición, baseline ingenuo, cantidad de parámetros) y
`rnn_histories.json`.

## Correspondencia con las consignas

| Requisito | Dónde se resuelve |
|---|---|
| Convolución 2D, padding, stride | `conv_block`: `Conv2D(3x3, padding="same")` + `MaxPooling2D(2, strides=2)` |
| Pooling / equivarianza traslacional | `MaxPooling2D` por bloque + `GlobalAveragePooling2D` final |
| Partición rigurosa sin leakage | `StratifiedKFold` sobre train; test de CIFAR-10 nunca se toca hasta la evaluación final |
| Análisis de sesgo y varianza | Media y desvío del accuracy entre folds; `generalization_gap` por fold; corrida `--no-regularization` como contraste |
| Métricas pertinentes | `classification_report` (precision/recall/F1 por clase, macro y weighted) + matriz de confusión |
| Orden secuencial y memoria temporal | Dos capas recurrentes apiladas, la primera con `return_sequences=True` |
| Justificación de la celda recurrente | `--cells simple_rnn gru lstm` entrena las tres con idéntico presupuesto y compara |
| Mitigación de gradiente | Compuertas de LSTM/GRU + `clipnorm=1.0` en el optimizador |
| Time split cronológico | `chronological_split` 70/15/15 por posición temporal; normalización con estadísticos **solo** de train |
| Pérdida secuencial e interpretabilidad | MAE y RMSE desnormalizados a °C, contrastados contra el baseline ingenuo |

## Análisis de gradientes y figuras

Requiere haber corrido antes los pipelines de CNN y RNN (usa sus modelos y
`_histories.json` como entrada).

```bash
python src/gradient_analysis.py --models artifacts/rnn/rnn_simple_rnn.keras artifacts/rnn/rnn_gru.keras artifacts/rnn/rnn_lstm.keras --window 120 --horizon 24 --outdir artifacts/gradients
python src/make_figures.py
```

Salidas: `artifacts/gradients/` (normas de gradiente por paso temporal) y
`artifacts/figures/` (curvas de entrenamiento y gráfico de gradientes).

## Informe .docx (opcional)

Genera un `.docx` con los resultados ya redactados (el texto y las tablas
están embebidos en el script) e inserta las figuras de
`artifacts/figures/fig_curvas_cnn.png` y `fig_gradientes.png` — hace falta
haber corrido antes `python src/make_figures.py`. Requiere Node.js.

```bash
cd tools
npm install
npm run build
```

Escribe `tools/informe_tecnico.docx`.

## Notas

- Ambos scripts fijan semilla (`--seed`) para que los resultados sean reproducibles.
- El baseline ingenuo de la RNN (predecir que la temperatura dentro de 24 h es la
  actual) es el número contra el que hay que justificar el modelo: si el MAE no
  baja de ahí, la red no aporta nada.
- La ventana por defecto es de 120 pasos horarios (5 días), suficiente para que
  una `SimpleRNN` muestre degradación del gradiente frente a GRU/LSTM.
