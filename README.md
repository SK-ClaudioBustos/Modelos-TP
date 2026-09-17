# TP Integrador - Redes Neuronales (CNN + RNN)

## Estructura del repo

```
src/    pipelines de entrenamiento (CNN, RNN, análisis de gradientes, figuras)
docs/   informes y notas del TP
tools/  generador del informe .docx (opcional, requiere Node)
```

Los datasets (CIFAR-10, Jena Climate) se descargan solos la primera vez que
corrés un script; no hace falta bajar nada a mano. Las salidas se escriben en
`artifacts/`, que no está versionado — cada uno genera las suyas al correr los
comandos de abajo.

## Instalación

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

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

Genera un `.docx` con los resultados ya redactados (no lee `artifacts/`, los
números están embebidos en el script). Requiere Node.js.

```bash
cd tools
npm install
npm run build
```

## Documentación adicional

En `docs/` están el resumen de resultados (`RESUMEN_RESULTADOS.md`), la guía
de generación de figuras (`GENERACION_FIGURAS.md`) y las corridas de
verificación adicionales (`VERIFICACION_RESULTADOS.md`, `INSTRUCCIONES_AGENTE.md`).

## Notas

- Ambos scripts fijan semilla (`--seed`) para que los resultados sean reproducibles.
- El baseline ingenuo de la RNN (predecir que la temperatura dentro de 24 h es la
  actual) es el número contra el que hay que justificar el modelo: si el MAE no
  baja de ahí, la red no aporta nada.
- La ventana por defecto es de 120 pasos horarios (5 días), suficiente para que
  una `SimpleRNN` muestre degradación del gradiente frente a GRU/LSTM.
