# Verificación de resultados

Tareas ejecutadas: 1, 2 y 3. **Tarea 4 (ventana larga) no se ejecutó**, por decisión explícita — no por límite de tiempo.

---

## Tarea 1 — Gradientes BPTT

Comando: `python src/gradient_analysis.py --models artifacts/rnn/rnn_simple_rnn.keras artifacts/rnn/rnn_gru.keras artifacts/rnn/rnn_lstm.keras --window 120 --horizon 24 --outdir artifacts/gradients`

| Modelo | Norma paso más nuevo | Norma paso más viejo | decay_ratio_oldest_over_newest | effective_memory_fraction |
|---|---|---|---|---|
| SimpleRNN | 1.0784e-03 | 1.6948e-11 | **1.5716e-08** | 0.2333 |
| GRU | 4.8631e-04 | 3.8014e-09 | **7.8169e-06** | 0.4000 |
| LSTM | 3.2433e-04 | 7.0475e-10 | **2.1729e-06** | 0.3417 |

Salidas: `artifacts/gradients/gradient_norms.csv`, `artifacts/gradients/gradient_summary.json`.

---

## Tarea 2 — CNN sin regularización

Comando: `python src/cnn_pipeline.py --no-regularization --max-folds 1 --epochs 30 --outdir artifacts/cnn_plain`

| | train_accuracy | val_accuracy | generalization_gap |
|---|---|---|---|
| Fold 1 (sin regularización, 30 epochs) | 0.9865 | 0.7871 | **0.1994** |
| Referencia: corrida original (regularizada, media 5 folds) | 0.8997 (aprox.) | 0.8615 | 0.0388 |

Salidas: `artifacts/cnn_plain/cnn_summary.json`.

---

## Tarea 3 — RNN sin features cíclicas

Comando: `python src/rnn_pipeline.py --cells simple_rnn gru --no-time-features --window 120 --epochs 20 --outdir artifacts/rnn_notime`

- `naive_baseline_test_mae_degc` (de esta corrida): **2.5079 °C**

| Celda | params | train_mae_degc | val_mae_degc | test_mae_degc | generalization_gap_degc | epochs_run |
|---|---|---|---|---|---|---|
| SimpleRNN | 8,193 | 2.3354 | 2.5138 | 2.3721 | 0.1783 | 20 |
| GRU | 24,801 | 2.1943 | 2.3716 | 2.2942 | 0.1773 | 18 |

Referencia — corrida original (con features cíclicas): SimpleRNN test MAE 2.3253 °C, GRU test MAE 2.1958 °C, brecha 0.1295 °C.

Brecha SimpleRNN − GRU en esta corrida (sin features cíclicas): 2.3721 − 2.2942 = **0.0779 °C**.

Salidas: `artifacts/rnn_notime/rnn_summary.json`.

---

## Tareas no ejecutadas

- **Tarea 4** (RNN ventana larga, `--window 336`): no ejecutada por decisión explícita del usuario, no por costo ni fallo.

---

## Observaciones

1. **¿El decay del gradiente distingue a la SimpleRNN de las celdas con compuertas? ¿Por cuántos órdenes de magnitud?**
   Sí. `decay_ratio_oldest_over_newest` de SimpleRNN es 1.57e-08, frente a 7.82e-06 en GRU y 2.17e-06 en LSTM. La SimpleRNN decae **~3 órdenes de magnitud más** que GRU (factor ≈497×) y **~2 órdenes de magnitud más** que LSTM (factor ≈138×).

2. **¿La corrida sin regularización muestra un gap mayor? ¿Cuánto?**
   Sí. El gap pasa de 0.0388 (corrida original, media de 5 folds regularizados) a 0.1994 (fold único sin regularización) — **más de 5 veces mayor** (+0.1606 en términos absolutos).

3. **Sin features cíclicas, ¿se abre la brecha entre SimpleRNN y GRU respecto de los 0.13 °C actuales?**
   No. La brecha se achica: de 0.1295 °C (corrida original) a 0.0779 °C (sin features cíclicas) — una **reducción**, no una apertura. Ambos modelos empeoran en términos absolutos al quitar las features cíclicas (SimpleRNN +0.047 °C, GRU +0.098 °C), pero GRU empeora más, por lo que la diferencia entre ambos se reduce en vez de crecer. Este resultado contradice la hipótesis de que la memoria recurrente por sí sola compensaría mejor la falta de features cíclicas en la celda con compuertas.
