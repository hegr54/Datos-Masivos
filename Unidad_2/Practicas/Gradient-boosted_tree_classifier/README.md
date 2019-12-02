### Algoritmos de:

#### árbol impulsado por gradiente
### Que es el algoritmo Clasificador de árbol impulsado por gradiente?

Los árboles impulsados ​​por gradientes (GBT) son un método popular de clasificación y regresión que utiliza conjuntos de árboles de decisión. spark.ml

###Como funciona: 
Importamos las bibliotecas y paquetes necesarios para cargar el programa.
* import org.apache.spark.ml.Pipeline 
* import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
* import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
* import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
* import org.apache.spark.sql.SparkSession

Creamos una instancia de la sesion de spark
* val spark = SparkSession.builder.appName("GradientBoostedTreeClassifierExample").getOrCreate()

cargan un conjunto de datos en formato LibSVM, lo dividen en conjuntos de entrenamiento y prueba, entrenan en el primer conjunto de datos y luego evalúan en el conjunto de prueba extendido. Utilizamos dos transformadores de características para preparar los datos; estas categorías de índice de ayuda para la etiqueta y las características categóricas, agregan metadatos a los DataFrameque el algoritmo del Árbol de decisión puede reconocer.

* val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")

Creacion del indexador de etiquetas el Índice de etiquetas, agregando metadatos a la columna de etiquetas y Se ajusta a todo el conjunto de datos para incluir todas las etiquetas en el índice.
*    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

Identifique automáticamente características categóricas e indícelas y Establezca maxCategories para que las entidades con> 4 valores distintos se traten como continuas.
* val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

Posteriormente realizamos nuestro entrenamiento con nuestros datos de la siguiente mandera:
 divida los datos en conjuntos de entrenamiento y prueba por medio de un arreglo (30% para pruebas y 70% de entrenamiento).
* val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

Entrenamos el modelo de Clasificador de árbol impulsado por gradiente el cual contendra las etiquetas del indice y las caracteristicas indexadas.
* val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")

Se crea la convercion de las etiquetas indexadas a etiquetas originales.
* val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

Cadena de indexadores y árbol en una tubería.
* val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

Pipelines proporciona un conjunto uniforme de API de alto nivel creadas sobre DataFrames que ayudan a los usuarios a crear y ajustar tuberías prácticas de aprendizaje automático.
MLlib estandariza las API para algoritmos de aprendizaje automático para facilitar la combinación de múltiples algoritmos en una sola tubería o flujo de trabajo. Esta sección cubre los conceptos clave introducidos por la API de Pipelines, donde el concepto de tubería se inspira principalmente en el proyecto scikit-learn .

En el aprendizaje automático, es común ejecutar una secuencia de algoritmos para procesar y aprender de los datos. Por ejemplo, un flujo de trabajo de procesamiento de documentos de texto simple puede incluir varias etapas:

* Divide el texto de cada documento en palabras.
* Convierta las palabras de cada documento en un vector de características numéricas.
* Aprenda un modelo de predicción utilizando los vectores de características y las etiquetas.
Cómo funciona
A Pipelinese especifica como una secuencia de etapas, y cada etapa es una Transformero una Estimator. Estas etapas se ejecutan en orden, y la entrada DataFramese transforma a medida que pasa por cada etapa. Para Transformeretapas, el transform()método se llama en DataFrame. Por Estimatoretapas, el fit()método se llama para producir una Transformer(que se convierte en parte de la PipelineModel, o equipada Pipeline), y que Transformer's transform()método se llama en el DataFrame.

Modelo de entrenamiento Esto también ejecuta los indexadores.
* val model = pipeline.fit(trainingData)

Generamos la predicion con la siguiente variable donde trasformamos la prueba de los datos
* val predictions = model.transform(testData)

Seleccione filas de ejemplo para mostrar
* predictions.select("predictedLabel", "label", "features").show(5)

creacion del evaluador de clasificacion que contendra las etiquetas indexadas la predicion , Seleccionar (predicción, etiqueta verdadera) y calcular error de prueba.
* val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
* val accuracy = evaluator.evaluate(predictions) -----> nombre de la metrica de exactitud
* println(s"Test Error = ${(1.0 - accuracy)}") -----------> test del error en este algorito entre mas cerca este de 1 mas precision. 

* val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
* println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
# Modelo de árbol de clasificación aprendido:
         GBTClassificationModel (uid=gbtc_59f66cfb58b0) with 10 trees
        Tree 0 (weight 1.0):
            If (feature 406 <= 12.0)
                Predict: 1.0
            Else (feature 406 > 12.0)
                Predict: -1.0
        Tree 1 (weight 0.1):
            If (feature 406 <= 12.0)
                If (feature 154 <= 165.5)
                    Predict: 0.4768116880884702
            Else (feature 154 > 165.5)
                Predict: 0.47681168808847024
            Else (feature 406 > 12.0)
                Predict: -0.4768116880884702
        Tree 2 (weight 0.1):
            If (feature 406 <= 12.0)
                If (feature 241 <= 91.0)
                    Predict: 0.43819358104272055
            Else (feature 241 > 91.0)
                Predict: 0.43819358104272066
            Else (feature 406 > 12.0)
                If (feature 350 <= 82.5)
                Predict: -0.4381935810427206
            Else (feature 350 > 82.5)
                Predict: -0.43819358104272066
        Tree 3 (weight 0.1):
            If (feature 490 <= 43.0)
                If (feature 512 <= 39.0)
                    If (feature 153 <= 8.5)
                        Predict: 0.4051496802845983
            Else (feature 153 > 8.5)
                Predict: 0.4051496802845984
            Else (feature 512 > 39.0)
                If (feature 512 <= 210.0)
                    If (feature 208 <= 241.0)
                        If (feature 124 <= 9.5)
                        Predict: 0.4051496802845983
                Else (feature 124 > 9.5)
                    Predict: 0.4051496802845984
            Else (feature 208 > 241.0)
                If (feature 152 <= 2.5)
                    Predict: 0.4051496802845983
                Else (feature 152 > 2.5)
                    Predict: 0.40514968028459836
            Else (feature 512 > 210.0)
                Predict: 0.4051496802845984
            Else (feature 490 > 43.0)
                Predict: -0.40514968028459825
        Tree 4 (weight 0.1):
            If (feature 490 <= 43.0)
                If (feature 267 <= 48.5)
                    If (feature 238 <= 17.5)
                    Predict: 0.3765841318352991
            Else (feature 238 > 17.5)
                Predict: 0.37658413183529926
            Else (feature 267 > 48.5)
                Predict: 0.3765841318352994
            Else (feature 490 > 43.0)
                Predict: -0.3765841318352992
        Tree 5 (weight 0.1):
                If (feature 406 <= 12.0)
                    If (feature 272 <= 3.0)
                        Predict: 0.35166478958101005
            Else (feature 272 > 3.0)
                Predict: 0.3516647895810101
            Else (feature 406 > 12.0)
                Predict: -0.35166478958101005
        Tree 6 (weight 0.1):
            If (feature 462 <= 62.5)
                If (feature 241 <= 27.5)
                    Predict: 0.32974984655529926
            Else (feature 241 > 27.5)
                Predict: 0.3297498465552993
            Else (feature 462 > 62.5)
                If (feature 267 <= 82.0)
                    Predict: -0.32974984655529926
            Else (feature 267 > 82.0)
                Predict: -0.3297498465552993
        Tree 7 (weight 0.1):
            If (feature 406 <= 12.0)
                If (feature 272 <= 80.0)
                    Predict: 0.31033724551979563
            Else (feature 272 > 80.0)
                    Predict: 0.3103372455197957
            Else (feature 406 > 12.0)
                If (feature 239 <= 28.0)
                    f (feature 377 <= 237.5)
                        Predict: -0.3103372455197956
            Else (feature 377 > 237.5)
                Predict: -0.3103372455197957
            Else (feature 239 > 28.0)
                Predict: -0.3103372455197957
        Tree 8 (weight 0.1):
            If (feature 406 <= 12.0)
                If (feature 183 <= 163.0)
                    Predict: 0.2930291649125433
            Else (feature 183 > 163.0)
                Predict: 0.2930291649125434
            Else (feature 406 > 12.0)
                If (feature 351 <= 102.0)
                    Predict: -0.2930291649125433
            Else (feature 351 > 102.0)
                Predict: -0.2930291649125434
        Tree 9 (weight 0.1):
            If (feature 406 <= 12.0)
                If (feature 127 <= 63.5)
                    If (feature 241 <= 27.5)
                        Predict: 0.27750666438358246
            Else (feature 241 > 27.5)
                Predict: 0.2775066643835825
            Else (feature 127 > 63.5)
                Predict: 0.27750666438358257
            Else (feature 406 > 12.0)
                If (feature 266 <= 42.5)
                    Predict: -0.2775066643835825
            Else (feature 266 > 42.5)
                If (feature 433 <= 173.5)
                    If (feature 155 <= 1.0)
                        Predict: -0.27750666438358246
            Else (feature 155 > 1.0)
                Predict: -0.2775066643835825
            Else (feature 433 > 173.5)
                Predict: -0.27750666438358257 

       
#### Authors: HERNANDEZ GARCIA RIGOBERTO
#### Titule: Clasificador de árbol impulsado por gradiente
