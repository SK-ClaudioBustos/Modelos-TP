const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
} = require("docx");

const FONT = "Calibri";
const W = 9360; // ancho util A4 con margenes de 1"

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 100, line: 259 },
    alignment: opts.align ?? AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: FONT, size: opts.size ?? 19, italics: opts.italics })],
  });
}

function h(text, level) {
  return new Paragraph({
    heading: level,
    spacing: { before: 180, after: 90 },
    children: [new TextRun({ text, font: FONT, bold: true, size: level === HeadingLevel.HEADING_1 ? 24 : 20, color: "1F3864" })],
  });
}

function cell(text, { bold = false, shaded = false, width } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shaded ? { type: ShadingType.CLEAR, fill: "DCE6F1" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({
      spacing: { after: 0 },
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, font: FONT, bold, size: 17 })],
    })],
  });
}

function table(header, rows, widths) {
  return new Table({
    columnWidths: widths,
    width: { size: W, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "8EA9DB" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "8EA9DB" },
      left: { style: BorderStyle.SINGLE, size: 4, color: "8EA9DB" },
      right: { style: BorderStyle.SINGLE, size: 4, color: "8EA9DB" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "B4C6E7" },
      insideVertical: { style: BorderStyle.SINGLE, size: 2, color: "B4C6E7" },
    },
    rows: [
      new TableRow({
        tableHeader: true,
        children: header.map((t, i) => cell(t, { bold: true, shaded: true, width: widths[i] })),
      }),
      ...rows.map(r => new TableRow({
        children: r.map((t, i) => cell(String(t), { width: widths[i], bold: i === 0 })),
      })),
    ],
  });
}

function caption(text) {
  return new Paragraph({
    spacing: { before: 60, after: 160 },
    alignment: AlignmentType.LEFT,
    children: [new TextRun({ text, font: FONT, size: 16, italics: true, color: "555555" })],
  });
}

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 19 } } } },
  sections: [{
    properties: { page: { margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
    children: [
      new Paragraph({
        spacing: { after: 40 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Trabajo Práctico Integrador — Redes Neuronales", font: FONT, bold: true, size: 26 })],
      }),
      new Paragraph({
        spacing: { after: 40 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Arquitecturas convolucionales y recurrentes en Keras", font: FONT, size: 20, color: "444444" })],
      }),
      new Paragraph({
        spacing: { after: 200 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Inteligencia Artificial — Lic. en Sistemas de Información — FCyT", font: FONT, size: 17, color: "666666" })],
      }),
      new Paragraph({
        spacing: { after: 240 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Integrantes: [Apellido 1], [Apellido 2]", font: FONT, size: 18 })],
      }),

      h("1. Objetivo y datasets", HeadingLevel.HEADING_1),
      p("Se implementaron dos arquitecturas de Deep Learning que explotan estructuras distintas de los datos: una red convolucional que aprovecha la localidad espacial y una red recurrente que modela dependencias temporales. Ambas se construyeron en Keras 3.15 sobre TensorFlow 2.21, con entrenamiento en CPU."),
      p("Para el módulo convolucional se utilizó CIFAR-10 (60.000 imágenes RGB de 32×32 px, 10 clases balanceadas). Para el módulo recurrente se utilizó Jena Climate 2009–2016, remuestreado a frecuencia horaria: 70.091 registros con 14 variables meteorológicas, ampliadas a 18 mediante codificación cíclica seno/coseno de los ciclos diario y anual."),

      h("2. Módulo CNN", HeadingLevel.HEADING_1),
      h("2.1. Arquitectura y justificación", HeadingLevel.HEADING_2),
      p("La red consta de tres bloques convolucionales con 32, 64 y 128 filtros. Cada bloque aplica dos convoluciones discretas 2D de kernel 3×3 con padding «same» y stride 1, seguidas de normalización por lotes y activación ReLU, y cierra con max-pooling 2×2 de stride 2 más dropout."),
      p("El kernel 3×3 apilado responde a un argumento de eficiencia paramétrica: dos convoluciones 3×3 consecutivas cubren un campo receptivo efectivo de 5×5 empleando 18 pesos por par de canales frente a los 25 de un kernel 5×5 único, e introducen una no linealidad adicional entre ambas. El padding «same» preserva la resolución dentro del bloque, de modo que la profundidad no queda limitada por el encogimiento espacial; la reducción de resolución se concentra deliberadamente en el pooling."),
      p("El max-pooling con stride 2 cumple dos funciones. Al tomar el máximo sobre una vecindad, la salida resulta invariante a traslaciones menores al tamaño de la ventana, lo que otorga la equivarianza traslacional parcial buscada; simultáneamente cuadruplica el campo receptivo relativo de las capas posteriores al reducir el mapa a un cuarto de su área."),
      p("Se eligió ReLU por su derivada unitaria en el semieje positivo, que evita la atenuación multiplicativa del gradiente propia de sigmoide y tanh, cuyas derivadas están acotadas por 0,25 y 1 respectivamente. El riesgo de unidades muertas se mitiga con la normalización por lotes previa, que mantiene las preactivaciones centradas y suaviza la superficie de pérdida. El clasificador emplea global average pooling en lugar de aplanamiento: reduce el tensor 4×4×128 a un vector de 128 componentes y elimina así más de 250.000 parámetros densos, actuando como regularizador estructural. La salida es una capa densa de 10 unidades con softmax, optimizada con entropía cruzada categórica dispersa."),
      p("Hiperparámetros: Adam con tasa inicial 10⁻³, lotes de 128 muestras, hasta 60 épocas. Se aplicaron early stopping sobre la exactitud de validación (paciencia 12, restauración de los mejores pesos) y reducción de la tasa de aprendizaje por meseta (factor 0,5; paciencia 5; mínimo 10⁻⁵). El aumento de datos comprende volteo horizontal, traslación y zoom aleatorios de hasta el 10 %, transformaciones que preservan la etiqueta semántica en este dominio."),

      h("2.2. Protocolo de partición", HeadingLevel.HEADING_2),
      p("El conjunto de prueba oficial de CIFAR-10 (10.000 imágenes) se reservó íntegro y no intervino en ninguna decisión de diseño. Sobre las 50.000 imágenes restantes se aplicó validación cruzada estratificada de 5 particiones con semilla fija, preservando la proporción de clases en cada pliegue. La estratificación garantiza que ninguna imagen participe simultáneamente del entrenamiento y la validación de un mismo pliegue, eliminando la fuga de información, y el desvío entre pliegues provee una estimación empírica directa de la varianza del estimador."),

      h("2.3. Resultados y análisis de sesgo y varianza", HeadingLevel.HEADING_2),
      table(
        ["Pliegue", "Exactitud entren.", "Exactitud valid.", "Brecha"],
        [["1","0,8971","0,8586","0,0385"],["2","0,8970","0,8612","0,0358"],["3","0,9078","0,8670","0,0408"],["4","0,8997","0,8616","0,0381"],["5","0,8997","0,8589","0,0408"],["Media","0,9003","0,8615 ± 0,0030","0,0388"]],
        [2000, 2450, 2660, 2250]
      ),
      caption("Tabla 1. Validación cruzada estratificada de 5 pliegues sobre CIFAR-10."),
      p("El modelo del pliegue 3 alcanzó sobre el conjunto de prueba una exactitud de 0,8634, con F1 macro y ponderado de 0,8627. La coincidencia entre ambas variantes del F1 es esperable dado el balance de clases, y que la métrica de prueba caiga dentro del intervalo estimado en validación confirma la ausencia de fuga."),
      p("El desvío de 0,0030 entre pliegues indica varianza baja: el rendimiento no depende de la partición concreta. La brecha media de 0,0388 señala un sobreajuste leve y controlado. Para cuantificar el aporte de la regularización se entrenó una variante con la misma topología convolucional pero sin normalización por lotes, dropout ni aumento de datos."),
      table(
        ["Configuración", "Exactitud entren.", "Exactitud valid.", "Brecha"],
        [["Regularizada (media 5 pliegues)","0,9003","0,8615","0,0388"],["Sin regularización (1 pliegue)","0,9865","0,7871","0,1994"]],
        [3600, 1920, 1920, 1920]
      ),
      caption("Tabla 2. Ablación de los mecanismos de regularización."),
      p("La variante sin regularización memoriza el conjunto de entrenamiento —alcanza 98,65 % de exactitud— mientras su rendimiento en validación cae 7,4 puntos porcentuales. La brecha se quintuplica, pasando de 0,0388 a 0,1994. Se trata de un régimen de varianza alta y sesgo bajo: capacidad suficiente, capacidad de generalización insuficiente. La comparación involucra 30 épocas y un pliegue frente a 60 épocas y cinco, pero la magnitud del efecto excede holgadamente esa diferencia de protocolo. Ninguna de las dos configuraciones muestra subajuste: el error de entrenamiento es bajo en ambas."),

      h("3. Módulo RNN", HeadingLevel.HEADING_1),
      h("3.1. Formulación y partición cronológica", HeadingLevel.HEADING_2),
      p("La tarea consiste en pronosticar la temperatura con 24 horas de anticipación a partir de ventanas de 120 pasos horarios (5 días) sobre las 18 variables disponibles. La serie se dividió por posición temporal en 70 % entrenamiento (49.063 registros), 15 % validación (10.514) y 15 % prueba (10.514), sin mezcla aleatoria."),
      p("La normalización se calculó exclusivamente con la media y el desvío del tramo de entrenamiento y se aplicó a los tres subconjuntos. Estandarizar con estadísticos globales constituiría una fuga sutil pero real, ya que la media del conjunto completo incorpora información del futuro. El desfase entre ventana y objetivo se verificó explícitamente para descartar solapamientos."),
      p("Como referencia se adoptó el predictor ingenuo que asume que la temperatura dentro de 24 horas coincide con la actual, cuyo error absoluto medio sobre el tramo de prueba es de 2,5079 °C. Toda mejora debe medirse contra este valor."),

      h("3.2. Elección de la celda recurrente", HeadingLevel.HEADING_2),
      p("En una celda simple el estado evoluciona según hₜ = tanh(W hₜ₋₁ + U xₜ + b). Al retropropagar a través del tiempo, la derivada del estado respecto de un instante k pasos anterior es un producto de k factores de la forma Wᵀ·diag(tanh′). Dado que |tanh′| ≤ 1, la norma del producto decae exponencialmente cuando el radio espectral de W es menor que uno, y diverge cuando lo supera. Las celdas con compuertas introducen una vía aditiva —el estado de celda en LSTM, la compuerta de actualización en GRU— cuya derivada se aproxima a la identidad cuando la compuerta permanece abierta, sustituyendo el producto de jacobianos por una suma y preservando el gradiente a lo largo de horizontes más extensos."),
      p("La arquitectura apila dos capas recurrentes de 64 y 32 unidades con dropout 0,2 entre ellas y una salida densa lineal, apropiada para regresión. Se optimizó con Adam (10⁻³) y recorte de gradiente por norma en 1,0, salvaguarda explícita contra la explosión del gradiente. La pérdida es el error absoluto medio, preferido sobre el cuadrático por su menor sensibilidad a valores atípicos y por ser directamente interpretable en grados Celsius."),

      h("3.3. Evidencia empírica del comportamiento del gradiente", HeadingLevel.HEADING_2),
      p("Para verificar el argumento anterior se midió la norma de la derivada de la pérdida respecto de la entrada en cada paso de la ventana, sobre los modelos ya entrenados. El cociente entre el paso más antiguo y el más reciente cuantifica la atenuación acumulada."),
      table(
        ["Celda", "Norma paso reciente", "Norma paso antiguo", "Cociente", "Memoria efectiva"],
        [["SimpleRNN","1,08×10⁻³","1,69×10⁻¹¹","1,57×10⁻⁸","28 de 120 pasos"],["LSTM","3,24×10⁻⁴","7,05×10⁻¹⁰","2,17×10⁻⁶","41 de 120 pasos"],["GRU","4,86×10⁻⁴","3,80×10⁻⁹","7,82×10⁻⁶","48 de 120 pasos"]],
        [1500, 2100, 2100, 1700, 1960]
      ),
      caption("Tabla 3. Atenuación del gradiente a lo largo de la ventana temporal. La memoria efectiva indica cuántos pasos conservan al menos el 1 % de la norma del paso más reciente."),
      p("La celda simple atenúa el gradiente 497 veces más que la GRU y 138 veces más que la LSTM, diferencias de dos y tres órdenes de magnitud que confirman cuantitativamente el análisis del BPTT. La memoria efectiva ordena las tres celdas en el mismo sentido: la SimpleRNN propaga señal útil durante poco más de un día, mientras la GRU alcanza el doble. La medición se realizó sobre redes entrenadas, de modo que refleja la interacción entre la arquitectura y lo que cada modelo aprendió a atender, no únicamente la topología en su inicialización."),

      h("3.4. Resultados y estabilidad predictiva", HeadingLevel.HEADING_2),
      table(
        ["Celda", "Parám.", "MAE entren.", "MAE valid.", "MAE prueba", "RMSE prueba", "Mejora"],
        [["Ingenuo","—","—","—","2,5079","—","—"],["SimpleRNN","8.449","2,3474","2,3955","2,3253","2,9369","7,3 %"],["LSTM","33.697","2,1736","2,3183","2,2461","2,8792","10,4 %"],["GRU","25.569","2,0623","2,2725","2,1958","2,8261","12,4 %"]],
        [1400, 1180, 1300, 1300, 1300, 1400, 1480]
      ),
      caption("Tabla 4. Desempeño sobre Jena Climate, en grados Celsius. La mejora se calcula respecto del predictor ingenuo."),
      p("Las tres celdas superan al predictor ingenuo de forma consistente y la GRU obtiene el menor error, con una mejora del 12,4 %. El ordenamiento coincide con el del análisis de gradiente, pero la magnitud de la diferencia entre celdas es modesta: la SimpleRNN queda apenas 0,13 °C por detrás de la GRU utilizando un tercio de los parámetros."),
      p("Esta aparente tensión admite una explicación. Pronosticar temperatura a 24 horas resulta en gran medida resoluble con las últimas horas de historia sumadas a la estacionalidad codificada explícitamente en las variables cíclicas, de modo que la tarea no exige la memoria prolongada que las compuertas habilitan. La ventaja arquitectónica existe y está medida, pero el problema no la demanda en toda su extensión."),
      p("Una ablación adicional respalda esta lectura. Al eliminar las cuatro variables cíclicas, ambos modelos se degradan, pero la GRU lo hace en mayor medida (+0,098 °C) que la celda simple (+0,047 °C), y la distancia entre ambas se reduce de 0,13 a 0,08 °C. El resultado contradice la hipótesis de partida —que sin codificación temporal explícita la memoria recurrente se volvería determinante— y sugiere que la ventaja de la GRU provenía de integrar mejor esas variables con el estado recurrente, antes que de recordar información remota."),
      p("Respecto de la estabilidad, las brechas entre entrenamiento y validación se mantienen acotadas: 0,05 °C en la celda simple, 0,14 °C en la LSTM y 0,21 °C en la GRU. La GRU exhibe la mayor brecha y simultáneamente el mejor error de prueba, un sobreajuste leve que no compromete la generalización. Cabe señalar que el error de prueba resulta inferior al de validación en los tres modelos; esto no indica mejor generalización sino que el último tramo cronológico presenta menor variabilidad térmica, motivo por el cual toda comparación se realiza contra el predictor ingenuo evaluado sobre ese mismo tramo."),

      h("4. Conclusiones", HeadingLevel.HEADING_1),
      p("La red convolucional alcanza 86,3 % de exactitud en prueba con una varianza entre pliegues de 0,30 puntos porcentuales. La ablación de los mecanismos de regularización quintuplica la brecha de generalización, lo que cuantifica su aporte en lugar de presuponerlo."),
      p("En el módulo recurrente, la medición directa del gradiente a lo largo de la ventana constituye evidencia de que las celdas con compuertas mitigan la atenuación característica del BPTT, con diferencias de hasta tres órdenes de magnitud. Sin embargo, esa ventaja se traduce en una mejora acotada del error de pronóstico, porque la tarea abordada no requiere memoria prolongada. La distinción entre una propiedad arquitectónica y su impacto sobre un problema concreto constituye el principal hallazgo del trabajo: la GRU es preferible en este caso, aunque no exactamente por el motivo que la teoría sugeriría de manera inmediata."),
    ],
  }],
});

Packer.toBuffer(doc).then(b => fs.writeFileSync("informe_tecnico.docx", b));
