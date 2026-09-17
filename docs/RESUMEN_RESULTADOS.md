# Resumen de resultados — TP Redes Neuronales (CNN + RNN)

Corridas completas ejecutadas el 2026-09-13, en CPU (Windows, sin GPU — TensorFlow >=2.11 no soporta GPU nativa en Windows).

## CNN — CIFAR-10

Comando: `python src/cnn_pipeline.py --folds 5 --epochs 60`

### Validación cruzada (5 folds, StratifiedKFold)

| Fold | Train acc | Val acc | Gap (train-val) | Epochs corridos |
|---|---|---|---|---|
| 1 | 0.8971 | 0.8586 | 0.0385 | 60 |
| 2 | 0.8970 | 0.8612 | 0.0358 | 60 |
| 3 (mejor) | 0.9078 | 0.8670 | 0.0408 | 60 |
| 4 | 0.8997 | 0.8616 | 0.0381 | 59 |
| 5 | 0.8997 | 0.8589 | 0.0408 | 60 |

- **Val accuracy media ± desvío**: 0.8615 ± 0.0030
- **Generalization gap medio**: 0.0388
- Alta consistencia entre folds (desvío bajo) → modelo estable, sin overfitting severo (BatchNorm + Dropout + data augmentation activos).

### Evaluación final sobre test de CIFAR-10 (modelo del fold 3, mejor val accuracy)

| Métrica | Valor |
|---|---|
| Accuracy | 0.8634 |
| Precision (macro / weighted) | 0.8651 / 0.8651 |
| Recall (macro / weighted) | 0.8634 / 0.8634 |
| F1-score (macro / weighted) | 0.8627 / 0.8627 |

Matriz de confusión completa: `artifacts/cnn/confusion_matrix.csv`.

### Artefactos

- `artifacts/cnn/cnn_best.keras` — mejor modelo (fold 3)
- `artifacts/cnn/cnn_fold{1..5}.weights.h5` — pesos por fold
- `artifacts/cnn/cnn_summary.json` — resumen completo (config, métricas por fold, classification report)
- `artifacts/cnn/cnn_histories.json` — curvas de loss/accuracy por epoch y fold
- `artifacts/cnn/confusion_matrix.csv`

---

## RNN — Jena Climate

Comando: `python src/rnn_pipeline.py --window 120 --horizon 24` (celdas: `simple_rnn`, `gru`, `lstm`)

- Ventana: 120 pasos horarios (5 días) · Horizonte: 24 h
- Split cronológico 70/15/15 (train=49063, val=10514, test=10514)
- Normalización con estadísticos solo de train
- **Baseline ingenuo (test)**: 2.5079 °C MAE

| Celda | Parámetros | Train MAE (°C) | Val MAE (°C) | Test MAE (°C) | Test RMSE (°C) | Gap (val-train) | Epochs corridos |
|---|---|---|---|---|---|---|---|
| SimpleRNN | 8,449 | 2.3474 | 2.3955 | 2.3253 | 2.9369 | 0.0481 | 11 |
| **GRU (mejor)** | 25,569 | 2.0623 | 2.2725 | **2.1958** | **2.8261** | 0.2102 | 19 |
| LSTM | 33,697 | 2.1736 | 2.3183 | 2.2461 | 2.8792 | 0.1447 | 9 |

- Los tres modelos superan claramente el baseline ingenuo.
- **GRU** obtiene el mejor test MAE y RMSE.
- SimpleRNN converge antes (menos epochs, menor gap) pero con peor error absoluto — consistente con la degradación de gradiente esperada frente a celdas con compuertas.

### Artefactos

- `artifacts/rnn/rnn_{simple_rnn,gru,lstm}.keras` — modelos entrenados
- `artifacts/rnn/rnn_{simple_rnn,gru,lstm}.weights.h5` — pesos
- `artifacts/rnn/rnn_summary.json` — resumen completo (config, métricas por celda)
- `artifacts/rnn/rnn_histories.json` — curvas de loss/rmse por epoch y celda

---

## Entorno de ejecución

- Python 3.12 (venv en `.venv/`, separado del Python 3.14 del sistema porque TensorFlow aún no soporta 3.14)
- TensorFlow 2.21.0, Keras 3.15.1
- Entrenamiento en CPU únicamente (GPU disponible en la máquina: NVIDIA RTX 5070 Laptop, no utilizada por falta de soporte nativo de TF en Windows — requeriría WSL2 para aprovecharla)
