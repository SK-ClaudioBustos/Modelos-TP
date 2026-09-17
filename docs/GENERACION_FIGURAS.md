# Instrucciones — generación de figuras (TP Redes Neuronales)

## Objetivo

Producir dos figuras para el informe técnico a partir de artefactos que **ya
existen**. No se entrena nada, no se modifica ningún modelo, no se sobrescribe
ningún resultado.

## Requisitos previos

`make_figures.py` debe estar en el directorio de trabajo, junto a la carpeta
`artifacts/`. Verificar que existan los tres insumos:

```bash
ls make_figures.py
ls artifacts/gradients/gradient_norms.csv
ls artifacts/cnn/cnn_histories.json
ls artifacts/cnn_plain/cnn_histories.json
```

Si falta alguno, detenerse y reportar cuál. No intentar regenerarlo.

Dependencia: `matplotlib`. Instalar en el mismo venv de Python 3.12 que se usó
para entrenar:

```bash
pip install matplotlib
```

## Ejecución

```bash
python src/make_figures.py
```

Salidas esperadas en `artifacts/figures/`:

- `fig_gradientes.png` — norma del gradiente respecto de la entrada, en escala
  logarítmica, contra pasos hacia el pasado. Una curva por celda.
- `fig_curvas_cnn.png` — exactitud de entrenamiento y validación por época, dos
  paneles: con y sin regularización.

## Posibles fallos y qué hacer

El script asume dos claves concretas dentro de los JSON de historiales:
`fold3` en la corrida regularizada (fue el mejor pliegue) y `fold1` en la
corrida sin regularización. Si aparece un `KeyError`, listar las claves reales:

```bash
python -c "import json;print(list(json.load(open('artifacts/cnn/cnn_histories.json'))))"
python -c "import json;print(list(json.load(open('artifacts/cnn_plain/cnn_histories.json'))))"
```

y reportar la salida **sin editar el script**.

Cualquier otro error: reportar el traceback completo sin intentar arreglarlo.

## Verificación

Abrir las dos imágenes y confirmar visualmente:

1. En `fig_gradientes.png` el eje vertical es logarítmico y las tres curvas son
   distinguibles. La curva de SimpleRNN debe quedar por debajo de las otras dos
   en la zona de pasos antiguos. El eje horizontal va de 0 (instante actual) a
   119 (paso más antiguo).
2. En `fig_curvas_cnn.png` el panel sin regularización debe mostrar las dos
   curvas separándose de forma marcada, y el panel regularizado debe mostrarlas
   próximas. Los dos paneles tienen distinta cantidad de épocas (60 y 30); eso
   es correcto y no hay que uniformarlo.
3. Ningún texto de ejes, título o leyenda aparece cortado o superpuesto.

Si alguna figura sale vacía, con una sola curva, o con los ejes ilegibles,
reportarlo describiendo qué se ve, sin modificar el script.

## Entrega

Adjuntar los dos archivos PNG y un archivo `FIGURAS.md` que indique:

- Si el script corrió sin errores.
- El resultado de los tres puntos de verificación, en una línea cada uno.
- Cualquier advertencia que haya emitido matplotlib.

No interpretar las figuras ni redactar texto para el informe.
