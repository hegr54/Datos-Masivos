### Algoritmos de:

#### Clasificador de árbol de decisión
### Que es el algoritmo Clasificador de árbol de decisión?

Los árboles de decisión son una familia popular de métodos de clasificación y regresión. spark.mlPuede encontrar más información sobre la implementación en la sección sobre árboles de decisión .

###Como funciona: 
Importamos las bibliotecas y paquetes necesarios para cargar el programa.
* import org.apache.spark.ml.Pipeline
* import org.apache.spark.ml.classification.DecisionTreeClassificationModel
* import org.apache.spark.ml.classification.DecisionTreeClassifier
* import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
* import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
* import org.apache.spark.sql.SparkSession

Creamos una instancia de la sesion de spark
* val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()

cargan un conjunto de datos en formato LibSVM, lo dividen en conjuntos de entrenamiento y prueba, entrenan en el primer conjunto de datos y luego evalúan en el conjunto de prueba extendido. Utilizamos dos transformadores de características para preparar los datos; estas categorías de índice de ayuda para la etiqueta y las características categóricas, agregan metadatos a los DataFrameque el algoritmo del Árbol de decisión puede reconocer.

* val data = spark.read.format("libsvm").load("spark/data/mllib/sample_libsvm_data.txt")

Índice de etiquetas, agregando metadatos a la columna de etiquetas asi tambien Se ajusta a todo el conjunto de datos para incluir todas las etiquetas en el índice.
* val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

Identifica automáticamente las características categóricas e indízalas.
* val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

Posteriormente realizamos nuestro entrenamiento con nuestros datos de la siguiente mandera:
 divida los datos en conjuntos de entrenamiento y prueba por medio de un arreglo (30% para pruebas).
* val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

Entrenamos el modelo de arboles de desicion el cual contendra las etiquetas del indice y las caracteristicas indexadas.
* val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

Se crea la convercion de las etiquetas indexadas a etiquetas originales.
* val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

Cadena de indexadores y árbol en una tubería.
* val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

Pipelines proporciona un conjunto uniforme de API de alto nivel creadas sobre DataFrames que ayudan a los usuarios a crear y ajustar tuberías prácticas de aprendizaje automático.
MLlib estandariza las API para algoritmos de aprendizaje automático para facilitar la combinación de múltiples algoritmos en una sola tubería o flujo de trabajo. Esta sección cubre los conceptos clave introducidos por la API de Pipelines, donde el concepto de tubería se inspira principalmente en el proyecto scikit-learn .

En el aprendizaje automático, es común ejecutar una secuencia de algoritmos para procesar y aprender de los datos. Por ejemplo, un flujo de trabajo de procesamiento de documentos de texto simple puede incluir varias etapas:

* Divide el texto de cada documento en palabras.
* Convierta las palabras de cada documento en un vector de características numéricas.
* Aprenda un modelo de predicción utilizando los vectores de características y las etiquetas.
Modelo de entrenamiento Esto también ejecuta los indexadores.
* val model = pipeline.fit(trainingData)

Generamos la predicion con la siguiente variable
* val predictions = model.transform(testData)

Seleccione filas de ejemplo para mostrar
* predictions.select("predictedLabel", "label", "features").show(5)

creacion del evaluador de clasificacion que contendra las etiquetas indexadas la predicion , Seleccionar (predicción, etiqueta verdadera) y calcular error de prueba.
* val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
* val accuracy = evaluator.evaluate(predictions) -----> nombre de la metrica de exactitud
* println(s"Test Error = ${(1.0 - accuracy)}") -----------> test del error en este algorito entre mas cerca este de 1 mas precision. 
#### Authors: HERNANDEZ GARCIA RIGOBERTO
#### Titule: Clasificador de árbol de decisión
