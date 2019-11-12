### Algoritmos de:

#### Clasificador Random Forest
### Que es el algoritmo Clasificador Random Forest?

Consiste en una gran cantidad de árboles de decisión individuales que operan como un conjunto. Cada árbol individual en el bosque aleatorio escupe una predicción de clase y la clase con más votos se convierte en la predicción de nuestro modelo.

###Como funciona: 
Importamos las bibliotecas y paquetes necesarios para cargar el programa.
* import org.apache.spark.ml.Pipeline
* import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
* import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
* import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
* import org.apache.spark.sql.SparkSession

Creamos una instancia de la sesion de spark
* val spark = SparkSession.builder.appName("RandomForestClassifierExample").getOrCreate()

Carga de nuestro dataset 

* val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
Data.show()

Índice de etiquetas, agregando metadatos a la columna de etiquetas y Se ajusta a todo el conjunto de datos para incluir todas las etiquetas en el índice. 
val  labelIndexer  =  new  StringIndexer () .setInputCol ( " etiqueta " ) .setOutputCol ( " indexedLabel " ) .fit (datos)

Identifica automáticamente las características categóricas y las indexa, Se establecen las categorías máximas para que las entidades con> 4 valores distintos se traten como continuas. 
* val  featureIndexer  =  new  VectorIndexer () .setInputCol ( " features " ) .setOutputCol ( " indexedFeatures " ) .setMaxCategories ( 4 ) .fit (datos)

Posteriormente realizamos nuestro entrenamiento con nuestros datos de la siguiente mandera:
 divida los datos en conjuntos de entrenamiento y prueba por medio de un arreglo (30% para pruebas y 70% de entrenamiento). 
* Val  Array (trainingData, testData) = data.randomSplit ( Array ( 0.7 , 0.3 ))

Entrena un modelo RandomForest. 
* val  rf  =  new  RandomForestClassifier () .setLabelCol ( " indexedLabel " ) .setFeaturesCol ( " indexedFeatures " ) .setNumTrees ( 10 )

Convierte las etiquetas indexadas de nuevo a etiquetas originales. 
* val  labelConverter  =  nuevo  IndexToString () .setInputCol ( " predicción " ) .setOutputCol ( " predictedLabel " ) .setLabels (labelIndexer.labels)

Indicadores de cadena y bosque en una tubería. 
* val  pipeline  =  new  Pipeline () .setStages ( Array (labelIndexer, featureIndexer, rf, labelConverter))

Modelo de tren. Esto también ejecuta los indexadores. 
* val model = pipeline.fit(trainingData)

Pipelines proporciona un conjunto uniforme de API de alto nivel creadas sobre DataFrames que ayudan a los usuarios a crear y ajustar tuberías prácticas de aprendizaje automático.
MLlib estandariza las API para algoritmos de aprendizaje automático para facilitar la combinación de múltiples algoritmos en una sola tubería o flujo de trabajo. Esta sección cubre los conceptos clave introducidos por la API de Pipelines, donde el concepto de tubería se inspira principalmente en el proyecto scikit-learn .

En el aprendizaje automático, es común ejecutar una secuencia de algoritmos para procesar y aprender de los datos. Por ejemplo, un flujo de trabajo de procesamiento de documentos de texto simple puede incluir varias etapas:

* Divide el texto de cada documento en palabras.
* Convierta las palabras de cada documento en un vector de características numéricas.
* Aprenda un modelo de predicción utilizando los vectores de características y las etiquetas.
Cómo funciona
A Pipelinese especifica como una secuencia de etapas, y cada etapa es una Transformero una Estimator. Estas etapas se ejecutan en orden, y la entrada DataFramese transforma a medida que pasa por cada etapa. Para Transformeretapas, el transform()método se llama en DataFrame. Por Estimatoretapas, el fit()método se llama para producir una Transformer(que se convierte en parte de la PipelineModel, o equipada Pipeline), y que Transformer's transform()método se llama en el DataFrame.

Hacer predicciones. 
* val predictions = model.transform(testData)

Seleccione filas de ejemplo para mostrar. 
* predictions.select ( " predictedLabel " , " label " , " features " ) .show ( 5 )

Seleccione (predicción, etiqueta verdadera) y calcule el error de prueba. 
* val  evaluator  =  new  MulticlassClassificationEvaluator () .setLabelCol ( " indexedLabel " ) .setPredictionCol ( " prediction " ) .setMetricName ( " precision " )
*   val  precision  = evaluator.evaluate (predictions)
println ( s " Error de prueba = $ {( 1.0  - precisión)} " )

* val  rfModel  = model.stages ( 2 ). asInstanceOf [ RandomForestClassificationModel ]
println ( s " Modelo de bosque de clasificación aprendida: \ n  $ {rfModel.toDebugString} " )


                Learned classification forest model:
                RandomForestClassificationModel (uid=rfc_a746584832a2) with 10 trees
                Tree 0 (weight 1.0):
                    If (feature 412 <= 8.0)
                        If (feature 454 <= 12.5)
                            redict: 0.0
                    Else (feature 454 > 12.5)
                        Predict: 1.0
                    Else (feature 412 > 8.0)
                    P   redict: 1.0
                Tree 1 (weight 1.0):
                    If (feature 463 <= 2.0)
                        If (feature 317 <= 8.0)
                            If (feature 216 <= 44.0)
                                Predict: 0.0
                    Else (feature 216 > 44.0)
                        Predict: 1.0
                    Else (feature 317 > 8.0)
                        Predict: 1.0
                    Else (feature 463 > 2.0)
                        Predict: 0.0
                Tree 2 (weight 1.0):
                    If (feature 540 <= 87.0)
                        If (feature 578 <= 9.0)
                            Predict: 0.0
                    Else (feature 578 > 9.0)
                        If (feature 550 <= 170.0)
                        Predict: 1.0
                    Else (feature 550 > 170.0)
                        Predict: 0.0
                    Else (feature 540 > 87.0)
                        Predict: 1.0
                Tree 3 (weight 1.0):
                    If (feature 518 <= 21.0)
                        If (feature 601 <= 27.0)
                            If (feature 605 <= 4.0)
                                Predict: 0.0
                    Else (feature 605 > 4.0)
                        Predict: 1.0
                    Else (feature 601 > 27.0)
                        Predict: 1.0
                    Else (feature 518 > 21.0)
                        If (feature 261 <= 1.0)
                            Predict: 0.0
                    Else (feature 261 > 1.0)
                        Predict: 1.0
                Tree 4 (weight 1.0):
                    If (feature 429 <= 7.0)
                        If (feature 358 <= 10.5)
                            Predict: 0.0
                    Else (feature 358 > 10.5)
                        Predict: 1.0
                    Else (feature 429 > 7.0)
                        Predict: 1.0
                Tree 5 (weight 1.0):
                    If (feature 462 <= 62.5)
                        Predict: 1.0
                    Else (feature 462 > 62.5)
                        Predict: 0.0
                Tree 6 (weight 1.0):
                    If (feature 385 <= 4.0)
                        If (feature 545 <= 3.0)
                        If (feature 600 <= 2.5)
                            Predict: 0.0
                    Else (feature 600 > 2.5)
                        Predict: 1.0
                    Else (feature 545 > 3.0)
                        Predict: 0.0
                    Else (feature 385 > 4.0)
                        Predict: 1.0
                Tree 7 (weight 1.0):
                    If (feature 512 <= 1.5)
                        If (feature 510 <= 2.5)
                            Predict: 0.0
                    Else (feature 510 > 2.5)
                        Predict: 1.0
                    Else (feature 512 > 1.5)
                        Predict: 1.0
                Tree 8 (weight 1.0):
                    If (feature 462 <= 62.5)
                        Predict: 1.0
                    Else (feature 462 > 62.5)
                        Predict: 0.0
                Tree 9 (weight 1.0):
                    If (feature 301 <= 27.0)
                        If (feature 517 <= 22.5)
                            If (feature 183 <= 3.0)
                                Predict: 0.0
                    Else (feature 183 > 3.0)
                        Predict: 1.0
                    Else (feature 517 > 22.5)
                        Predict: 0.0
                    Else (feature 301 > 27.0)
                        Predict: 1.0
#### Authors: HERNANDEZ GARCIA RIGOBERTO
#### Titule: Clasificador Random Forest
