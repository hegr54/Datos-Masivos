# ÍNDICE


* [INTRODUCCIÓN](#INTRODUCCIÓN)
*   [MARCO TEÓRICO](#MARCO_TEÓRICO)
*   [Decision_tree_classifier](#Decision_tree_classifier)
*   [Ventajas](#Ventajas)	
*   [Limitaciones](#Limitaciones)	
*   [Multilayer Perceptron Classifier](#Multilayer_Perceptron_Classifier)	
*   [Historia](#Historia)	
*   [Arquitectura de capa en MLPC[2]](#Arquitectura_de_capa_en_MLPC[2])	
*   [Características](#Características)
*   [Limitaciones_de_MLPC](#Limitaciones_de_MLPC)	
*   [Naive Bayes](#Naive_Bayes)	
*  [Principio del clasificador Bayes:](#Principio_del_clasificador_Bayes:)	
*   [Teorema de Bayes:](#Teorema_de_Bayes:)	
*   [Apache-Spark](#Apache-Spark)	
*   [Velocidad](#Velocidad)	
*   [Scala](#Scala)	
*   [IMPLEMENTACIÓN](#IMPLEMENTACIÓN)	
*   [Decision tree classifier](#Decision_tree_classifier)	
*   [Código](#Código)	
*  [ Multilayer Perceptron Classifier](#Multilayer_Perceptron_Classifier)	
*   [Código](#Codigo)
*   [Naive Bayes](#Naive_Bayes)	
*   [Código](#Código)
*   [RESULTADOS](#RESULTADOS)
*   [CONCLUSIÓN](#CONCLUSIÓN)	
*   [REFERENCIAS](#REFERENCIAS)	


### INTRODUCCIÓN
En el presente documento abordaremos tres de los algoritmos existentes de clasificación, Se trata de los algoritmos  Clasificación Decision tree classifier, Multilayer perceptron classifier y Naive Bayes. Estos algoritmos tienen como objetivo la comparación del rendimiento en machine learning.

También se hablará sobre cada uno de estos algoritmos de clasificación de cómo funcionan, ventajas que pueden tener a la hora de encontrarse en uso para la clasificación de una gran cantidad de datos y sus desventajas.

Las tecnologías utilizadas para poder realizar las comparaciones de rentabilidad y eficiencia de los algoritmos se abordará el tema de una breve descripción de cada una de estas herramientas. 

Se generarán unas pruebas a cada uno de los algoritmos antes mencionados para ver sus comportamiento a la hora de procesar información. 

En la implementación de cada uno de los algoritmos se llevará un control dentro del mismo código como es su comportamiento.

Al concluir el documento se dará una breve conclusión de lo analizado en este documento. Respondiendo las siguientes preguntas:
¿cual algoritmo presenta mejor su exactitud?
¿Cual algoritmo presenta un margen de error más amplio?
¿Cual algoritmo tarda menos tiempo en la clasificación de la información?
¿Por qué utilizamos esta herramienta?
Spark es un framework que te permite procesar big data y permite el cálculo en tiempo real que no se puede lograr con otras herramientas.

### MARCO_TEÓRICO
¿Qué es un clasificador?
Un clasificador es un modelo de aprendizaje automático que se utiliza para discriminar diferentes objetos en función de ciertas características.
#### Decision_tree_classifier

Árbol de decisión o Decision Tree Classification es un tipo de algoritmo de aprendizaje supervisado que se utiliza principalmente en problemas de clasificación, aunque funciona para variables de entrada y salida categóricas como continuas.

En esta técnica, dividimos la data en dos o más conjuntos homogéneos basados en el diferenciador más significativos en las variables de entrada. El árbol de decisión identifica la variable más significativa y su valor que proporciona los mejores conjuntos homogéneos de población. Todas las variables de entrada y todos los puntos de división posibles se evalúan y se elige la que tenga mejor resultado.[4]

Los algoritmos de aprendizaje basados en árbol se consideran uno de los mejores y más utilizados métodos de aprendizaje supervisado. Los métodos basados en árboles potencian modelos predictivos con alta precisión, estabilidad y facilidad de interpretación. A diferencia de los modelos lineales, mapean bastante bien las relaciones no lineales.[4]
##### Ventajas 
Fácil de entender. La salida del árbol de decisión es muy fácil de entender, incluso para personas con antecedentes no analíticos, no se requiere ningún conocimiento estadístico para leerlos e interpretarlos.
Útil en la exploración de datos. El árbol de decisión es una de las formas más rápidas para identificar las variables más significativas y la relación entre dos o más. Con la ayuda de los árboles de decisión podemos crear nuevas variables o características que tengan mejor poder para predecir la variable objetivo.
Se requiere menos limpieza de datos. Requiere menos limpieza de datos en comparación con algunas otras técnicas de modelado. A su vez, no está influenciado por los valores atípicos y faltantes en la data.
El tipo de datos no es una restricción. Puede manejar variables numéricas y categóricas.
Método no paramétrico. Es considerado un método no paramétrico, esto significa que los árboles de decisión no tienen suposiciones sobre la distribución del espacio y la estructura del clasificador.[4]
##### Limitaciones 

Sobreajuste. Es una de las dificultades más comunes que tiene este algoritmo, este problema se resuelve colocando restricciones en los parámetros del modelo y eliminando ramas en el análisis.
Los modelos basados en árboles no están diseñados para funcionar con características muy dispersas. Cuando se trata de datos de entrada dispersos (por ejemplo, características categóricas con una gran dimensión), podemos preprocesar las características dispersas para generar estadísticas numéricas, o cambiar a un modelo lineal, que es más adecuado para dichos escenarios.[4]

#### Multilayer_Perceptron_Classifier
##### Historia
En 1965 fue creado el algoritmo de  Multilayer Perceptron Classifier (MLPC) es una «aplicación» del percepción de una neurona a más de una. Además aparece el concepto de capas de entrada, oculta y salida. Pero con valores de entrada y salida binarios. No olvidemos que tanto el valor de los pesos como el de umbral de cada neurona lo asignar manualmente el científico. Cuantos más perceptrones en las capas, mucho más difícil conseguir los pesos para obtener salidas deseadas.[1]

Las redes de avance profundo, también llamadas a menudo redes neuronales de avance, o perceptrones  multicapa (MLP), son los modelos de aprendizaje profundo por excelencia.[2]

Figura 1: Un ejemplo de arquitectura de perceptrón multicapa

Para entrenar un clasificador de perceptrón multicapa basado en Spark, se deben establecer los siguientes parámetros:
Capa
* Tolerancia de iteración
* Tamaño de bloque del aprendizaje
* Tamaño de la semilla
* Número de iteración máx.

Tenga en cuenta que un valor menor de tolerancia de convergencia conducirá a una mayor precisión con el costo de más iteraciones. El parámetro de tamaño de bloque predeterminado es 128 y el número máximo de iteraciones se establece en 100 como valor predeterminado.[2]
Arquitectura de capa en MLPC[2]
Como se describe en la Figura 1, MLPC consta de múltiples capas de nodos, incluida la capa de entrada, las capas ocultas (también llamadas capas intermedias) y las capas de salida. Cada capa está completamente conectada a la siguiente capa en la red. Donde la capa de entrada, las capas intermedias y la capa de salida se pueden definir de la siguiente manera: 
La capa de entrada  consta de neuronas que aceptan los valores de entrada. La salida de estas neuronas es la misma que la de los predictores de entrada. Los nodos en la capa de entrada representan los datos de entrada. Todos los demás nodos asignan entradas a salidas mediante una combinación lineal de las entradas con los pesos w y sesgos del nodo b y aplicando una función de activación. Esto puede escribirse en forma de matriz para MLPC con  capas de la siguiente manera [4]:K+1
Las capas ocultas se  encuentran entre las capas de entrada y salida. Por lo general, el número de capas ocultas varía de uno a muchos. Es la capa central de computación que tiene las funciones que asignan la entrada a la salida de un nodo. Los nodos en las capas intermedias utilizan la función sigmoidea (logística), como sigue:

La  capa de salida  es la capa final de una red neuronal que devuelve el resultado al entorno del usuario. Basado en el diseño de una red neuronal, también señala las capas anteriores sobre cómo se han desempeñado en el aprendizaje de la información y, en consecuencia, mejoraron sus funciones. Los nodos en la capa de salida  utilizan la función softmax.
##### Características
*   Se trata de una estructura altamente no lineal.
*   Presenta tolerancia a fallos.
*   El sistema es capaz de establecer una relación entre dos conjuntos de datos.
*   Existe la posibilidad de realizar una implementación hardware. 

Es el algoritmo más popular de aprendizaje para el perceptrón multicapa es el backpropagation, el cual consiste en utilizar el error generado por la red y propagarlo hacia atrás, es decir, reproducirlo hacia las neuronas de las capas anteriores.[3]

El algoritmo backpropagation para el MLPC presenta ciertas desventajas, como son: lentitud de convergencia, precio a pagar por disponer de un método general de ajuste funcional, puede incurrir en sobre aprendizaje, fenómeno directamente relacionado con la capacidad de generalización de la red. Y no garantiza el mínimo global de la función de error, tan solo un mínimo local. [3]

##### Limitaciones_de_MLPC

El Perceptrón Multicapa no extrapola bien, es decir, si la red se entrena mal o de manera insuficiente, las salidas pueden ser imprecisas.
La existencia de mínimos locales en la función de error dificulta considerablemente el entrenamiento, pues una vez alcanzado un mínimo el entrenamiento se detiene aunque no se haya alcanzado la tasa de convergencia fijada.

#### Naive_Bayes
##### Principio_del_clasificador_Bayes:
Un clasificador Naive Bayes es un modelo probabilístico de aprendizaje automático que se utiliza para la tarea de clasificación. El quid del clasificador se basa en el teorema de Bayes.

Naive Bayes es un clasificador de aprendizaje automático simple, pero efectivo y de uso común. Es un clasificador probabilístico que realiza clasificaciones utilizando la regla de decisión Máximo A posteriori en un entorno bayesiano. También se puede representar usando una red bayesiana muy simple. Los clasificadores ingenuos de Bayes han sido especialmente populares para la clasificación de texto, y son una solución tradicional para problemas como la detección de spam.[5]

##### Teorema_de_Bayes:

Usando el teorema de Bayes, podemos encontrar la probabilidad de que ocurra A , dado que B ha ocurrido. Aquí, B es la evidencia y A es la hipótesis. La suposición hecha aquí es que los predictores / características son independientes. Es decir, la presencia de una característica particular no afecta a la otra. Por eso se llama ingenuo.[6]
#### Apache-Spark
Apache Spark es un sistema de computación en clúster rápido y de propósito general. Proporciona API de alto nivel en Java, Scala, Python y R, y un motor optimizado que admite gráficos de ejecución generales. También es compatible con un amplio conjunto de herramientas de alto nivel que incluyen Spark SQL para SQL y procesamiento de datos estructurado, MLlib para aprendizaje automático, GraphX ​​para procesamiento de gráficos y Spark Streaming.

Apache Spark ™ es un motor de análisis unificado para el procesamiento de datos a gran escala.[7]

##### Velocidad
Ejecute cargas de trabajo 100 veces más rápido.
Apache Spark logra un alto rendimiento tanto para datos de lote como de transmisión, utilizando un programador DAG de última generación, un optimizador de consultas y un motor de ejecución física.[7]
#### Scala
Scala combina programación orientada a objetos y funcional en un lenguaje conciso de alto nivel. Los tipos estáticos de Scala ayudan a evitar errores en aplicaciones complejas, y sus tiempos de ejecución de JVM y JavaScript le permiten construir sistemas de alto rendimiento con fácil acceso a enormes ecosistemas de bibliotecas.[8]

### IMPLEMENTACIÓN
#### Decision_tree_classifier
##### Código
```scala
                import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
                import org.apache.spark.mllib.util.MLUtils
                import org.apache.spark.sql.SparkSession
                import org.apache.spark.sql.types.DateType
                import org.apache.spark.sql.{SparkSession}
                import org.apache.spark.ml.feature.VectorIndexer
                import org.apache.spark.ml.feature.VectorAssembler
                import org.apache.spark.ml.Transformer
                import org.apache.spark.mllib.tree.model.DecisionTreeModel
                import org.apache.spark.ml.{Pipeline, PipelineModel}
                import org.apache.spark.ml.feature.StringIndexer
                import org.apache.spark.mllib.tree.DecisionTree
                import org.apache.spark.mllib.tree.model.DecisionTreeModel
                import org.apache.spark.mllib.util.MLUtils
                import org.apache.spark.ml.classification.DecisionTreeClassifier
                import org.apache.spark.ml.feature.IndexToString
                import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
                import org.apache.spark.ml.classification.DecisionTreeClassificationModel
                import org.apache.log4j._
                import java.time._
                Logger.getLogger("org").setLevel(Level.ERROR)
                // $example off$
                import org.apache.spark.sql.SparkSession

                val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()
                    // $example on$
                    // Load the data stored in LIBSVM format as a DataFrame.
                val data = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")
                data.show()
                data.printSchema()
                val Si = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
                val No = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
                val newcolumn = no.withColumn("y",'y.cast("Int"))
                //Desplegamos la nueva columna
                newcolumn.show(1)
                //Generamos la tabla features
                val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
                val fea = assembler.transform(newcolumn)
                //Mostramos la nueva columna
                fea.show(1)

                //Cambiamos la columna y a la columna label
                val cambio = fea.withColumnRenamed("y", "label")
                val feat = cambio.select("label","features")
                feat.show(1)
                    // Index labels, adding metadata to the label column.
                    // Fit on whole dataset to include all labels in index.
                val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(feat)
                // Automatically identify categorical features, and index them.
                val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)

                // Split the data into training and test sets (30% held out for testing).
                val Array(trainingData, testData) = feat.randomSplit(Array(0.7, 0.3))

                // Train a DecisionTree model.
                val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

                // Convert indexed labels back to original labels.
                val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

                // Chain indexers and tree in a Pipeline.
                val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

                // Train model. This also runs the indexers.
                val model = pipeline.fit(trainingData)

                // Make predictions.
                val predictions = model.transform(testData)

                // Select example rows to display.
                predictions.select("predictedLabel", "label", "features").show(5)

                // Select (prediction, true label) and compute test error.
                val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
                val accuracy = evaluator.evaluate(predictions)
                println(s"Test Error = ${(1.0 - accuracy)}")

                val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
                println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
                // $example off$
                val t1 = System.currentTimeMillis
```


#### Multilayer_Perceptron_Classifier
##### Codigo
```scala
            //Importación del algoritmo Multilayer Perception Classifier y importamos las librerías importantes para generar 
            import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
            import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
            import org.apache.spark.mllib.util.MLUtils
            import org.apache.spark.sql.SparkSession
            import org.apache.spark.sql.types.DateType
            import org.apache.spark.sql.{SparkSession, SQLContext}
            import org.apache.spark.ml.feature.VectorIndexer
            import java.time._
            import org.apache.spark.ml.feature.VectorAssembler
            import org.apache.spark.ml.Transformer
            import org.apache.spark.ml.classification.LinearSVC
            import org.apache.spark.ml.classification.LogisticRegression
            import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
            import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
            import org.apache.log4j._
            //Creación de la sección de spark.
            val Spark= SparkSession.builder().getOrCreate()
            //carga y lectura de nuestro dataset en un formato csv.
            val dataset = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").csv("bank-full.csv")
            //visualización de nuestro data set.
            dataset.show()
            //Visualización del esquema o como se encuentran nuestros datos.
            dataset.printSchema()
            // Convertimos valores en string a valores enteros ya que scala no maneja valores string.
            //hacer la limpieza de datos necesaria para poder ser procesado con el siguiente algoritmo
            creamos una variable la cual será llamada YES de la columna Y en la cual se sustituirá el string YES a un valor entero 1, esto también va aplicar para el No por un 2.
            val yes = dataset.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
            val no = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
            val newcolumn = no.withColumn("y",'y.cast("Int"))
            //Desplegamos la nueva columna
            newcolumn.show(1)
            //Generamos la tabla features
            val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
            val fea = assembler.transform(newcolumn)
            //Mostramos la nueva columna
            fea.show(1)

            //Cambiamos o renombramos la columna Y a label
            val cambio = fea.withColumnRenamed("y", "label")
            val feat = cambio.select("label","features")
            //Creación de una nueva tabla la cual contendrá las características con su respectiva etiqueta.
            feat.show(1) 
            //función de entrenamiento el cual será 60% de entrenamiento y 40% de prueba con una profundidad o semilla de 1234L.
            val splits = feat.randomSplit(Array(0.6, 0.4), seed = 1234L)
                val train = splits(0)
                val test = splits(1)

            // Especificación de nuestra capa de red neuronal de la siguiente manera:
            //la capa de entrada será de tamaño 4 características, dos capas oculta de tama;o 5 y 4, así tendremos una salida de tama;o de 3 clases.
            val layers = Array[Int](5, 5, 4, 3)
            //creamos el entrenador multilayerPercptronClassifier y establecimiento de sus parámetros.
            val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
            //creamos el modelo de entrenamiento
            val model = trainer.fit(train)
            //calculamos la predicción en el conjunto de prueba.
            val result = model.transform(test)
            val predictionAndLabels = result.select("prediction", "label")
            val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
            //Imprimimos el accuracy, error y tiempo de ejecución en milisegundos.
            println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
            println(s"Test Error = ${(1.0 - accuracy)}")
            val t1 = System.currentTimeMillis
```
#### Naive_Bayes
##### Código
``` scala
            import org.apache.spark.ml.classification.NaiveBayes
            import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
            import org.apache.spark.ml.feature.VectorIndexer
            import org.apache.spark.ml.feature.VectorAssembler
            import java.time._
            // Importación de apis y librerías necesarias para este algoritmo.
            import org.apache.spark.sql.SparkSession
            val spark = SparkSession.builder.appName("NaiveBayes").getOrCreate()

                //lectura de nuestro dataset en un formato csv.
            val dataset = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").csv("bank-full.csv")
            //visualización de nuestro data set.
            dataset.show()
            //Visualización del esquema o como se encuentran nuestros datos.
            dataset.printSchema()
            //hacer la limpieza de datos necesaria para poder ser procesado con el siguiente algoritmo creamos una variable la cual será llamada YES de la columna Y en la cual se sustituirá el string YES a un valor entero 1, esto también va aplicar para el No por un 2.
            val yes = dataset.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
            val no = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
            val newcolumn = no.withColumn("y",'y.cast("Int"))
            //Desplegamos la nueva columna
            newcolumn.show(1)
            //Generamos la tabla features con un arreglo de características que nosotros definimos.
            val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
            val fea = assembler.transform(newcolumn)
            //Mostramos la nueva columna
            fea.show(1)

            //Cambiamos o renombramos la columna Y a Label
            val cambio = fea.withColumnRenamed("y", "label")
            val feat = cambio.select("label","features")
            //Visualización de nuestra nueva tabla que contiene las características y etiquetas necesarias para este algoritmo.
            feat.show(1)  
                // Dividir los datos en conjuntos de entrenamiento y prueba (30% para pruebas)
            val Array(trainingData, testData) = feat.randomSplit(Array(0.7, 0.3), seed = 1234L)

                // Entrena un modelo Naive Bayes.
            val model = new NaiveBayes().fit(trainingData)

                //Seleccione filas de ejemplo para mostrar
            val predictions = model.transform(testData)

                //Seleccionar (predicción, etiqueta verdadera) y calcular error de prueba
            val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
            val accuracy = evaluator.evaluate(predictions)
            //Imprimimos el accuracy, error y tiempo de ejecución en milisegundos.
            println(s"Test set accuracy = $accuracy")
            println(s"Test Error = ${(1.0 - accuracy)}")
            val t1 = System.currentTimeMillis

### RESULTADOS
            En los próximas pruebas se tomarán en cuenta los comportamiento de cada uno de los algoritmos con la misma base de datos para ver como es la exactitud, error de cada prueba y cuanto es el tiempo de ejecución de cada uno.

            Algoritmos
            Split(Array(0.7, 0.3), seed = 1234L) y tiempo de ejecucion
            Decision tree classifier
            Test set accuracy = 0.8955091807388835                                           
            Test Error = 0.10449081926111647
            Time: 1575612711481
            Split(Array(0.6, 0.4), seed = 1234L) y tiempo de ejecución
            Test set accuracy = 0.894921960870521                                            
            Test Error = 0.105078039129479
            Time: 1575613276945

            Split(Array(0.7, 0.3), seed = 1234L) y tiempo de ejecucion
            Multilayer Perceptron Classifier
            Test set accuracy = 0.8862768145753747
            Test Error = 0.11372318542462534
            Time: 1575612952354
            Split(Array(0.6, 0.4), seed = 1234L) y tiempo de ejecución
            Test set accuracy = 0.8848956335944776                                          
            Test Error = 0.11510436640552235
            Time: 1575613369107

            Split(Array(0.7, 0.3), seed = 1234L) y tiempo de ejecucion
            Naive Bayes
            Test set accuracy = 0.8862768145753747
            Test Error = 0.11372318542462534
            Time: 1575613063496
            Split(Array(0.6, 0.4), seed = 1234L) y tiempo de ejecución
            Test set accuracy = 0.8848956335944776
            Test Error = 0.11510436640552235
            Time: 1575613419712
    tabla 1    
En la tabla 1 puesta en marcha se muestra el comportamiento de Decision tree Classifier con una división de datos con un 70% de entrenamiento y 30% de pueda en el cual nos demuestra que tienen una exactitud 0.8955, tiene un error de 0.1044 en un tiempo de ejecución de 1575612711481 milisegundos.
En el análisis del funcionamiento de MLPC nos da un resultado de exactitud de 0.8862 con un error de 0.1137 en un tiempo de 1575612952354 y por último en nuesta última prueba en naive bayes nos da la exactitud de 0.8862 con un error de 0.1137 en un tiempo de 1575613063496.
En ambas pruebas no se da un cambio significativo ya que los valores cambian muy poco.

            Algoritmos
            Split(Array(0.8, 0.2), seed = 1234L) y tiempo de ejecucion
            Decision tree classifier
            Test set accuracy = 0.8880894398937348                                           
            Test Error = 0.11191056010626521
            Time: 1575613556126
            Split(Array(0.75, 0.25), seed = 1234L) y tiempo de ejecución
            Test set accuracy = 0.8912827909492472                                           
            Test Error = 0.10871720905075277
            Time: 1575613802467
            
            Split(Array(0.8, 0.2), seed = 1234L) y tiempo de ejecucion
            Multilayer Perceptron Classifier
            Test set accuracy = 0.8874070835155226
            Test Error = 0.11259291648447745
            Time= 1575613685235
            Split(Array(0.75, 0.25), seed = 1234L) y tiempo de ejecución
            Test set accuracy = 0.8865416776692192                                          
            Test Error = 0.11345832233078079
            Time: 1575614046487

            Split(Array(0.8, 0.2), seed = 1234L) y tiempo de ejecucion
            Naive Bayes
            Test set accuracy = 0.026781810231744644
            Test Error = 0.9732181897682554
            Time: 1575613836298
            Split(Array(0.75, 0.25), seed = 1234L) y tiempo de ejecución
            Test set accuracy = 0.8865416776692192
            Test Error = 0.11345832233078079
            Time: 1575614083143
    TABLA 2

En la tabla de los algoritmos se puede notar que existe poca diferencia de exactitud, error y tiempo de cada algoritmo a la tabla 1 ya que los datos divididos fueron 80% de entrenamiento y 20% de tabla a lo que los algoritmos no presentaron cambios a la hora de realizar sus diferentes tareas.

            Algoritmos
            Split(Array(0.5, 0.5), seed = 1234L) y tiempo de ejecucion
            
            Decision tree classifier
            Test set accuracy = 0.8901357225228422
            Test Error = 0.10986427747715777
            Time = 1575614355827
            Split(Array(0.65, 0.35), seed = 1234L) y tiempo de ejecución
            Test set accuracy = 0.8951306112097388                                           
            Test Error = 0.1048693887902612
            Time: 1575614431408

            Split(Array(0.5, 0.5), seed = 1234L) y tiempo de ejecucion
            Multilayer Perceptron Classifier
            Test set accuracy = 0.884087912087912                                           
            Test Error = 0.11591208791208796
            Time: 1575614479715
            Split(Array(0.65, 0.35), seed = 1234L) y tiempo de ejecución
            Test set accuracy = 0.8854597917450759                                          
            Test Error = 0.11454020825492406
            Time: 1575614536528

            Split(Array(0.5, 0.5), seed = 1234L) y tiempo de ejecucion
            Naive Bayes
            accuracy: Double = 0.884087912087912
            Test set accuracy = 0.884087912087912
            Test Error = 0.11591208791208796
            Time: 1575614584288
            Split(Array(0.65, 0.35), seed = 1234L) y tiempo de ejecución
            accuracy: Double = 0.8854597917450759
            Test set accuracy = 0.8854597917450759
            Test Error = 0.11454020825492406
            Time: 1575614617784
    TABLA 3
En la tabla 3 se puede observar que cuando seguimos trabajando con nuestros algoritmos de clasificación nos generan los mismos resultados ya que nuestros algoritmos se encuentran entrenando para una mejor respuesta a cada una de las peticiones que nosotros les pedimos.

### CONCLUSIÓN
Todos estos modelos que presentan son muy buenos ya que pueden facilitar el hacer patrones en la data para que el data scientist realice su tarea .
Se analizó en las tablas 1,2 y 3 que los resultados de cada algoritmo son muy diferentes, los comportamientos es el siguiente:

En la tabla 1 los algoritmos presentan sus resultados, nos damos cuenta que el algoritmo Decision tree classifier es más exacto en el tratado de los datos llevandole una ventaja significativa a MLPC, pero sin embargo MLPC tiene una mejor predicción de error en un tiempo más corto, Naive bayes es un algoritmo que pretende ir a la par que Decision tree y MLPC pero tiene una exactitud,  error y tiempo es muy por debajo de de ellos.
En la tabla 2 los resultados siguen siendo los mismos que la tabla 1 y decision tree sigue llevando la batuta en cuanto a su exactitud, error y tiempo contra los dos algoritmos. En esta prueba Decision tree género sus resultados en menos tiempo ganándole a MLPC y a Naive Bayes con gran facilidad.
Por último en el análisis de la tabla 3 se percató que efectivamente el algoritmo Decision tree sigue generando los mismos datos y llevando una relevante ventaja a los demás algoritmos de clasificación.
Con los datos analizados en las tablas anteriores se puede llegar a concluir que el algoritmo Decision tree es ganador en esta comparación ya que se vio superior a MLPC y a Naive Bayes en todas las pruebas que se les implementaron. Probablemente es por el tiempo de análisis y tratado de los datos, porque si hablamos de rapidez de ejecución MLPC se lleva la de ganar ya que es el algoritmo que tarda menos en generar un resultado. Naive Bayes es un algoritmo que te da una exactitud y un error bueno pero no óptimo y rápido como los demás algoritmos.
### REFERENCIAS
1. NA. (Breve Historia de las Redes Neuronales Artificiales). septiembre 12, 2018 . 5/12/2019, de aprende ML Sitio web: https://www.aprendemachinelearning.com/breve-historia-de-las-redes-neuronales-artificiales/
2. Md. Rezaul Karim. (12 y 16 de diciembre). Aprendizaje profundo a través del clasificador perceptrón multicapa. 2019, de Zona de Big Data Sitio web: https://dzone.com/articles/deep-learning-via-multilayer-perceptron-classifier
3. Darwin Mercado Polo1 , Luis Pedraza Caballero2 , Edinson Martínez Gómez3. ( 13/11/14 ). Comparación de Redes Neuronales aplicadas a la predicción de Series de Tiempo. 2019, de e Tiempo Completo, Universidad de la Costa - CUC, Sitio web: http://www.scielo.org.co/pdf/prosp/v13n2/v13n2a11.pdf
4. ligdi Gonzales. (23 Marzo, 2018). Aprendizaje Supervisado: Decision Tree Classification. 2019, de LIGDI Sitio web: http://ligdigonzalez.com/aprendizaje-supervisado-decision-tree-classification/
5. Devin Soni. (May 16, 2018). Introduction to Naive Bayes Classification 2019, de Data science Sitio web: https://towardsdatascience.com/introduction-to-naive-bayes-classification-4cffabb1ae54
6. Rohith Gandhi. (5 de mayo de 2018). Clasificador ingenuo de Bayes. 2019, de Data science Sitio web: https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
7. apachespark. ((Nov 06, 2019)). Lightning-fast unified analytics engine. 2019, de apache-spark Sitio web: http://spark.apache.org/
8. scala. (2019). scala. 2019, de Lausanne (EPFL) Lausanne, Switzerland Sitio web: https://www.scala-lang.org/

#### Authors: HERNANDEZ GARCIA RIGOBERTO
#### Titule: PROYECTO FINAL