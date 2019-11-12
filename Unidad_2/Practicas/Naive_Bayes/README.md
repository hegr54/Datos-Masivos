### Algoritmos de:

#### Naive Bayes
### Que es el algoritmo Naive Bayes?

Los clasificadores de Naive Bayes son una familia de clasificadores probabilísticos simples y multiclase basados ​​en la aplicación del teorema de Bayes con fuertes supuestos de independencia (ingenuos) entre cada par de características.

Los Naive Bayes pueden ser entrenados de manera muy eficiente. Con un solo paso sobre los datos de entrenamiento, calcula la distribución de probabilidad condicional de cada característica dada cada etiqueta. Para la predicción, aplica el teorema de Bayes para calcular la distribución de probabilidad condicional de cada etiqueta dada una observación.

###Como funciona: 
Importamos las bibliotecas y paquetes necesarios para cargar el programa.
* import  org.apache.spark.ml.classification.NaiveBayes 
* import  org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
* import org.apache.spark.sql.SparkSession

Creamos una instancia de la sesion de spark
* val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()

caraga de nuestro dataset
* val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")

Posteriormente realizamos nuestro entrenamiento con nuestros datos de la siguiente mandera:
 divida los datos en conjuntos de entrenamiento y prueba por medio de un arreglo (30% para pruebas y 70% de entrenamiento).
* val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

Entrena un modelo NaiveBayes asiendo un ajuste a los mismos. 
* val model = new NaiveBayes().fit(trainingData)

Seleccion de filas de ejemplo para mostrar las cuales fueron transformadas del dataset.
* val predictions = model.transform(testData)
* predictions.show() -------------> Muestra de las prediciones 

generamos el evaluador de multiclassclassficationevaluator el cual contendra las columnas de las etiquetas, la predicion de las columnas y el nombre de la metrica.
* val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")

creacion de la variable predicion 
* val accuracy = evaluator.evaluate(predictions)
Impresion de los resultados 
* println(s"Test set accuracy = $accuracy") 

                accuracy: Double = 0.8
                Test set accuracy = 0.8
#### Authors: HERNANDEZ GARCIA RIGOBERTO
#### Titule: Naive Bayes
