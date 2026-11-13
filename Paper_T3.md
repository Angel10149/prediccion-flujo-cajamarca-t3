# Pronóstico de aforos vehiculares en la intersección Hoyos Rubio y Andrés Zevallos de Cajamarca

Luis Asencio, José Muñoz, María Quispe, Rodrigo Sánchez y Yeber Tasilla

Universidad Privada del Norte · Sistemas Inteligentes y Machine Learning · Avance T3 · 2026

## Resumen
Se implementó un prototipo de pronóstico de volumen vehicular a 15 minutos con datos secundarios de la intersección Hoyos Rubio y Andrés Zevallos, Cajamarca. Se extrajeron 64 conteos de dos campañas documentadas por Cerna Loyola (2020), agrupados en cuatro movimientos y dos franjas horarias. Los rezagos se construyeron dentro de cada bloque, produciendo 24 pares para ajuste y 24 para prueba independiente. Se compararon Ridge, árbol de decisión, Random Forest y KNN, junto con persistencia. La selección mediante validación por bloques de la primera campaña favoreció Ridge. En prueba obtuvo MAE de 22,23 vehículos, RMSE de 25,42, R² de 0,352 y MAPE de 20,27 %. La interfaz local reproduce los modelos entrenados y permite contrastar predicciones con aforos publicados. La muestra histórica y reducida limita la generalización al tránsito actual.

Palabras clave — aforo vehicular, predicción, aprendizaje supervisado, Cajamarca, validación.

## Abstract
A 15-minute traffic-volume forecasting prototype was implemented using secondary observations from the Hoyos Rubio and Andrés Zevallos intersection in Cajamarca. Sixty-four counts from two campaigns were recovered from Cerna Loyola (2020). Lagged pairs were formed within each movement and time block, yielding 24 training and 24 independent test examples. Ridge regression, a decision tree, Random Forest and KNN were compared against persistence. Group-based validation on the first campaign selected Ridge. Test MAE was 22.23 vehicles, RMSE 25.42, R² 0.352 and MAPE 20.27%. A local interface performs inference with the exported trained models. The small historical sample limits generalization to current traffic conditions.

Keywords — traffic counts, forecasting, supervised learning, Cajamarca, validation.

## I Introducción
### 1.1 Antecedentes
La planificación de una solución predictiva requiere conteos trazables, definición del horizonte y evaluación sobre observaciones reservadas. Cerna Loyola (2020) documenta mediciones de campo en la intersección Hoyos Rubio y Andrés Zevallos para construir y validar una microsimulación con Vissim. Sus tablas permiten recuperar la composición vehicular y los volúmenes de dos campañas. El estudio constituye el antecedente local y la fuente empírica de este avance. Sus resultados sobre microsimulación y nivel de servicio tienen una finalidad distinta de los errores de pronóstico que se calculan aquí.

Hyndman y Athanasopoulos (2021) explican la evaluación de pronósticos con datos reservados y el uso de referencias simples. Este enfoque orienta la inclusión de persistencia y la separación entre ajuste, selección y prueba. Breiman (2001) desarrolla Random Forest como combinación de árboles, que ofrece una referencia metodológica para explorar relaciones no lineales. La documentación actual de scikit-learn permite implementar los estimadores y su procesamiento de forma reproducible.

### 1.2 Planteamiento del problema
El equipo dispone de un notebook inicial basado en aforos sintéticos y de una propuesta amplia para tres corredores. Para aportar evidencia verificable en T3 se reformula el alcance experimental hacia una intersección con registros publicados. La pregunta es qué error presentan cuatro modelos supervisados al estimar el siguiente aforo de 15 minutos en una campaña reservada, usando volumen previo, franja horaria y movimiento vial.

La utilidad esperada es apoyar el análisis del comportamiento del flujo. La reducción efectiva de congestión requeriría una intervención y mediciones adicionales. Los conteos usados fueron obtenidos históricamente y la presente implementación se realiza en 2026.

### 1.3 Objetivos
El objetivo general es implementar y evaluar, antes de la entrega T3, un prototipo reproducible de pronóstico a 15 minutos en la intersección seleccionada, comparando cuatro modelos con una referencia de persistencia.

OE1. Recuperar 64 conteos de las 16 tablas de aforo, documentar su procedencia y comprobar sus sumas y discrepancias antes de entrenar.

OE2. Construir 24 pares de ajuste y 24 de prueba con rezagos internos a cada bloque, entrenar cuatro estimadores y seleccionar uno por validación interna agrupada durante esta entrega.

OE3. Reportar MAE, RMSE, R² y MAPE sobre la campaña reservada y verificar que la interfaz reproduce las predicciones exportadas, antes de cerrar el paquete de entrega.

El criterio operativo es completar la trazabilidad, reproducibilidad y concordancia entre código e interfaz. Como referencia de desempeño se compara el modelo con persistencia. Se conservan como metas de aspiración R² ≥ 0,85 y MAPE ≤ 15 %, declarando expresamente su resultado al evaluar. Un resultado inferior informa el desarrollo posterior.

## II Marco teórico
### 2.1 Herramientas de ingeniería
Python implementa el procesamiento y la evaluación. pypdf recupera las tablas y NumPy representa matrices. scikit-learn proporciona estimadores, escalamiento y validación agrupada. Matplotlib produce las figuras. HTML y JavaScript ejecutan la inferencia local desde parámetros exportados. El notebook incorpora los datos y permite repetir el entrenamiento en Colab.

El CSV de detalle conserva cada categoría y sus cuatro conteos; el CSV agregado conserva campaña, movimiento, franja, intervalo y página de origen. Esta separación hace auditables tanto la extracción como las variables del modelo. El hash del PDF permite identificar la fuente exacta utilizada.

### 2.2 Algoritmos y evaluación
Ridge ajusta una combinación lineal de variables con penalización cuadrática sobre sus coeficientes. El escalamiento usa únicamente los ejemplos del ajuste correspondiente. La regularización controla la magnitud de los coeficientes en una muestra pequeña. Un árbol particiona las entradas en regiones y asigna el promedio de cada hoja. Random Forest promedia múltiples árboles entrenados con remuestreo, controlando su profundidad. KNN promedia las respuestas de los cinco ejemplos de entrenamiento más cercanos en el espacio escalado. Persistencia asigna al siguiente intervalo el último conteo observado.

MAE expresa el error absoluto medio en vehículos por intervalo. RMSE penaliza más los errores grandes. R² compara el error cuadrático con la variación alrededor de la media observada. MAPE expresa el error relativo medio porcentual; todos los conteos de prueba son positivos. La comparación exige la misma campaña y los mismos 24 objetivos para cada modelo.

## III Metodología
### 3.1 Materiales y métodos
El diseño es cuantitativo, retrospectivo y exploratorio. La fuente primaria es la tesis de Cerna Loyola (2020), tablas 11–18 y 23–30, páginas 70–77 y 83–90. La primera campaña se fecha el 06/10/2018 en la página 60. La segunda se identifica como 13 de octubre en la página 83; ese párrafo omite el año, por lo que el CSV conserva la descripción literal y deja vacía su fecha normalizada.

Cada campaña contiene cuatro movimientos y dos franjas, 12:00–13:00 y 18:00–19:00, divididas en cuatro intervalos. Se sumaron las trece categorías publicadas, incluidas bicicletas, para conservar la definición de volumen de la fuente. Las 16 tablas producen 64 registros agregados y 208 filas de categorías. La unidad estadística es un movimiento en un intervalo de 15 minutos.

Se detectaron discrepancias aritméticas en la fuente. En la tabla 24, las columnas de camionetas suman 113 y el total impreso es 112; el total agregado calculado es 525 frente a 524 publicado. En la tabla 30, las columnas de camionetas suman 57 frente a 47 y el agregado es 358 frente a 348. Se conservaron los valores por intervalo y se documentaron ambas diferencias. Las imágenes de esas páginas se incluyen como evidencia.

Las variables explicativas son volumen del intervalo anterior, franja (12 o 18), posición del intervalo objetivo y cuatro indicadores de movimiento. El objetivo es el volumen del siguiente intervalo. Se excluye el primer intervalo de cada bloque porque carece de rezago. La fecha, el total de la hora y las categorías del intervalo objetivo quedan fuera de las entradas. Tampoco se enlaza el final del mediodía con el inicio de la tarde.

La campaña de ajuste aporta 24 pares. Se realiza LeaveOneGroupOut sobre ocho bloques definidos por movimiento y franja. Cada partición reserva un bloque completo. Esta validación interna evalúa transferencia entre bloques de la campaña y no estima por sí sola generalización temporal. La campaña de prueba aporta otros 24 pares, que se utilizan después de fijar el ganador. En prueba se conoce el aforo anterior a cada predicción, como en un uso secuencial a un paso.

### 3.2 Proceso de desarrollo
La secuencia fue identificar y descargar la fuente, verificar su ubicación, extraer las tablas, comprobar sumas, construir rezagos, validar por bloques, ajustar cada modelo, evaluar la campaña reservada y exportar los parámetros de inferencia. La exportación permite usar la interfaz sin servidor ni entrenamiento adicional.

Se fijaron los siguientes hiperparámetros antes de comparar las métricas de prueba. Ridge usa alpha=10 y StandardScaler. El árbol usa profundidad máxima 2 y mínimo 3 ejemplos por hoja. Random Forest utiliza 100 árboles, profundidad 3, mínimo 3 ejemplos por hoja y semilla 42. KNN utiliza cinco vecinos, pesos uniformes y StandardScaler. Se mantuvo una configuración por familia para reducir el ajuste excesivo de una muestra pequeña. Estos cuatro modelos sustituyen el alcance previo de LSTM y XGBoost en esta propuesta acotada.

La ejecución usa CPU en Windows y bibliotecas versionadas en requisitos.txt. Los scripts se pueden ejecutar en otra plataforma con Python. Los tiempos registrados incluyen validación interna y ajuste, y dependen del equipo. El notebook usa el mismo código y el CSV incorporado.

## IV Resultados
### 4.1 Recuperación y calidad de los datos
Se recuperaron 64 registros agregados y 208 filas de categorías. Catorce tablas coinciden con los totales publicados y dos presentan las discrepancias ya descritas. La trazabilidad se conserva mediante número de tabla, página, URL y hash del PDF. El conjunto queda limitado a dos campañas y ocho bloques por campaña.

### 4.2 Comparación experimental
La tabla siguiente resume el experimento. CV es el MAE obtenido reservando bloques de la primera campaña. Entrenamiento es el MAE del modelo ajustado sobre esa campaña. Las cuatro métricas restantes corresponden exclusivamente a prueba.

{{METRICAS}}

Ridge obtuvo el menor MAE de validación interna, 18,50. Su MAE de entrenamiento fue 11,44 y el de prueba 22,23. Esta diferencia indica que el ajuste dentro de la muestra resulta más favorable que la transferencia a la otra campaña. KNN obtuvo el menor MAE de prueba, 17,29, aunque el criterio de selección previamente definido mantuvo Ridge como modelo predeterminado de la interfaz. La interfaz permite inspeccionar las cuatro alternativas.

Frente a persistencia, Ridge reduce RMSE de 28,19 a 25,42, aproximadamente 9,84 %, mientras su MAE aumenta de 21,58 a 22,23. Por tanto, el balance depende de la medida utilizada. Ridge no alcanza las metas de R² ≥ 0,85 ni MAPE ≤ 15 %. La evaluación identifica una limitación concreta de precisión que debe abordarse con mayor cobertura y nuevas variables.

{{FIGURA_PREDICCIONES}}

Figura 1. Pronósticos del modelo seleccionado y observaciones de prueba por movimiento. Cada punto usa exclusivamente el conteo anterior disponible dentro de su bloque. El salto entre franjas representa un periodo sin observación.

{{FIGURA_ERRORES}}

Figura 2. MAE por modelo en la campaña reservada. La selección predeterminada se realizó previamente mediante validación interna.

### 4.3 Implementación y pruebas
El archivo ABRIR_INTERFAZ.html carga modelos.js y calcula el pronóstico para el movimiento, intervalo y aforo anterior elegidos. El usuario puede cargar cualquiera de los 24 casos de prueba, observar el valor real y su error absoluto. Los parámetros proceden del entrenamiento sobre aforos de la tesis. La interfaz rechaza conteos vacíos, negativos, fraccionarios o superiores a 1000 y advierte cuando la entrada supera el rango observado en entrenamiento.

La evidencia verificable comprende scripts de extracción y entrenamiento, CSV originales y agregados, registro de discrepancias, parámetros exportados, predicciones individuales, notebook y figuras. El control de concordancia compara los resultados de JavaScript y Python con tolerancia numérica sobre todos los casos de prueba.

## V Discusión y conclusiones
La microsimulación de Cerna Loyola caracteriza una intersección concreta, mientras este avance reutiliza sus aforos para pronóstico supervisado. Los errores de ambas tareas requieren métricas distintas y no se comparan numéricamente como si midieran el mismo resultado. La evaluación con referencia simple sigue la práctica de contrastar el valor añadido del modelo con un procedimiento reproducible.

Respecto al OE1, se completó la recuperación trazable de 64 conteos y se identificaron dos discrepancias de suma. Respecto al OE2, se construyeron 48 pares repartidos por campaña y se entrenaron cuatro modelos. Respecto al OE3, se obtuvieron las cuatro métricas de prueba y una interfaz de inferencia. La evidencia permite demostrar un avance funcional y reconocer el incumplimiento de las metas aspiracionales de precisión.

La cobertura de dos campañas, la antigüedad del registro, la elección original de horas de alta demanda y los errores de suma reducen representatividad. Se recomienda recolectar varias semanas en intervalos continuos y validar por días completos. También se recomienda ampliar variables con eventos y condiciones climáticas emparejadas por fecha comprobada, y evaluar regularización y rezagos mediante validación temporal interna. Cualquier nuevo ajuste deberá disponer de otra prueba reservada para evitar seleccionar repetidamente sobre la misma campaña.

## VI Aprendizaje y uso responsable de IA
El desarrollo requirió extracción de tablas, comprobación de procedencia, diseño de rezagos, validación agrupada y exportación de modelos. Las evidencias son los scripts y los registros de ejecución. La asistencia de IA intervino en búsqueda, programación y redacción. La verificación incluyó descarga del PDF, inspección de páginas, sumas aritméticas y contraste entre implementaciones. Los integrantes deben comprender y poder explicar el protocolo y los resultados durante la sustentación.

Se corrigió una atribución previa de datos de la avenida Mesones Muro de Jaén a Hoyos Rubio. Esa serie se excluyó de la presente entrega. Este hallazgo refuerza la necesidad de comprobar el título y el ámbito geográfico antes de reutilizar un resultado recuperado por búsqueda.

## VII Consideraciones éticas
Los datos son conteos agregados sin identificadores personales. Los riesgos principales son la transferencia de datos históricos a condiciones actuales, sesgo por selección de franjas, fuga de información entre entrenamiento y prueba, errores de transcripción e interpretación operativa excesiva. Las medidas adoptadas son mostrar cobertura y fuente, reservar una campaña, construir rezagos dentro de bloques, conservar discrepancias y presentar el sistema como prototipo académico. Una intervención sobre semáforos requeriría evaluación vial y autorización institucional.

## Referencias
Cerna Loyola, M. S. (2020). Microanálisis de los desplazamientos peatonales y vehiculares entre la av. Hoyos Rubio y la av. Andrés Zevallos, Cajamarca – 2020 [Tesis de licenciatura, Universidad Privada del Norte]. Repositorio Institucional UPN. https://repositorio.upn.edu.pe/handle/11537/25244

Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32. https://doi.org/10.1023/A:1010933404324

Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting principles and practice (3.ª ed.). OTexts. https://otexts.com/fpp3/

Scikit-learn developers. (s. f.). Ridge. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

Scikit-learn developers. (s. f.). RandomForestRegressor. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

Scikit-learn developers. (s. f.). KNeighborsRegressor. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
