# Instrucciones — corridas de verificación (TP Redes Neuronales)

## Contexto

El proyecto ya tiene dos pipelines funcionando y corridas completas ejecutadas:

- `cnn_pipeline.py` — CIFAR-10, StratifiedKFold 5 folds. Val accuracy
  0.8615 ± 0.0030, test accuracy 0.8634, gap medio 0.0388.
- `rnn_pipeline.py` — Jena Climate, ventana 120 / horizonte 24 h. Baseline
  ingenuo 2.5079 °C; SimpleRNN 2.3253, LSTM 2.2461, GRU 2.1958 (test MAE).

Esas corridas **no se rehacen**. Los artefactos en `artifacts/cnn/` y
`artifacts/rnn/` son entregables: no se tocan ni se sobrescriben. Cada tarea
escribe en su propio `--outdir`.

Todo corre en CPU, así que las tareas están ordenadas por costo. **Ejecutar en
orden y parar cuando el tiempo disponible se agote**: la Tarea 1 sola ya cubre
lo más importante.

Archivos requeridos en el directorio de trabajo: `src/rnn_pipeline.py`,
`src/cnn_pipeline.py`, `src/gradient_analysis.py`. Verificar antes de empezar:

```bash
grep -c "no-time-features" src/rnn_pipeline.py   # debe devolver 1
ls src/gradient_analysis.py
```

Si el grep devuelve 0 o falta el script, pedir los archivos actualizados antes
de continuar.

## Motivo

Dos afirmaciones del informe no están respaldadas por los resultados actuales:

1. "Regularizamos para prevenir sobreajuste" — no hay corrida sin
   regularización contra la cual comparar.
2. "La SimpleRNN rinde peor por degradación del gradiente" — obtiene 2.3253 °C
   con 8.449 parámetros contra 2.1958 °C de la GRU con 25.569. Por parámetro es
   la más eficiente, así que el error final **no** demuestra degradación de
   gradiente.

El objetivo es conseguir evidencia, no confirmar una conclusión. Si los
resultados contradicen la hipótesis, se reportan tal cual.

---

## Tarea 1 — gradientes BPTT (minutos, sin entrenamiento)

Mide la norma de ∂loss/∂x_t para cada paso temporal sobre los modelos **ya
entrenados**. Es evidencia directa del comportamiento del gradiente, no inferida
del error final.

```bash
python src/gradient_analysis.py --models artifacts/rnn/rnn_simple_rnn.keras artifacts/rnn/rnn_gru.keras artifacts/rnn/rnn_lstm.keras --window 120 --horizon 24 --outdir artifacts/gradients
```

Salidas: `artifacts/gradients/gradient_norms.csv` (curva por modelo) y
`gradient_summary.json`.

Reportar por modelo: `decay_ratio_oldest_over_newest` y
`effective_memory_fraction`.

Esperado: la SimpleRNN con un decay varios órdenes de magnitud menor que GRU y
LSTM. Si las tres dan parecido, decirlo explícitamente.

## Tarea 2 — contraste sin regularización (CNN, ~10% del costo de la corrida original)

```bash
python src/cnn_pipeline.py --no-regularization --max-folds 1 --epochs 30 --outdir artifacts/cnn_plain
```

Un solo fold: el objetivo es la brecha, no la varianza.

Reportar de `artifacts/cnn_plain/cnn_summary.json`, fold 1: `train_accuracy`,
`val_accuracy`, `generalization_gap`.

Esperado: gap bastante mayor a 0.0388.

## Tarea 3 — sin features cíclicas (RNN, costo similar a la corrida original)

Solo si queda tiempo después de las dos anteriores.

```bash
python src/rnn_pipeline.py --cells simple_rnn gru --no-time-features --window 120 --epochs 20 --outdir artifacts/rnn_notime
```

Quita las columnas seno/coseno (18 → 14 features). La única vía para capturar el
ciclo diario pasa a ser la memoria recurrente.

Reportar de `artifacts/rnn_notime/rnn_summary.json`:
`naive_baseline_test_mae_degc` y, por celda, `params`, `train_mae_degc`,
`val_mae_degc`, `test_mae_degc`, `generalization_gap_degc`, `epochs_run`.

## Tarea 4 — ventana larga (RNN, ~3× el costo de la corrida original)

**Opcional, la más cara.** Ejecutar solo si sobra tiempo de máquina.

```bash
python src/rnn_pipeline.py --cells simple_rnn gru --window 336 --epochs 20 --outdir artifacts/rnn_long
```

Reportar los mismos campos que la Tarea 3.

---

## Advertencias

- No cambiar `--seed` (default 42) ni otros hiperparámetros. La comparación
  contra las corridas originales solo vale si cambia únicamente la variable bajo
  estudio.
- El baseline ingenuo se recalcula en cada corrida. En la Tarea 4 va a diferir
  de 2.5079 porque la ventana más larga desplaza el tramo evaluado. Comparar
  cada modelo contra el baseline **de su propia corrida**.
- Si una corrida supera las 2 horas, cortarla y reportar el estado en vez de
  dejarla colgada.
- Si algo falla, reportar el traceback completo sin intentar arreglar el
  pipeline por cuenta propia.

## Formato de entrega

Un archivo `VERIFICACION_RESULTADOS.md` con una sección por tarea ejecutada, las
tablas de métricas pedidas, y al final "Observaciones" respondiendo solo:

1. ¿El decay del gradiente distingue a la SimpleRNN de las celdas con
   compuertas? ¿Por cuántos órdenes de magnitud?
2. ¿La corrida sin regularización muestra un gap mayor? ¿Cuánto?
3. (Si se corrió) Sin features cíclicas, ¿se abre la brecha entre SimpleRNN y
   GRU respecto de los 0.13 °C actuales?

Indicar qué tareas quedaron sin ejecutar. No redactar conclusiones para el
informe ni interpretar más allá de esas preguntas.
