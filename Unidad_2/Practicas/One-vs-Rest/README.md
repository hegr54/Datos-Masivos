### Algoritmos de:

#### Clasificador One-vs-Rest
### Que es el algoritmo Clasificador One-vs-Rest?

OneVsRest es un ejemplo de una reducción de aprendizaje automático para realizar una clasificación multiclase dado un clasificador base que puede realizar la clasificación binaria de manera eficiente. También se conoce como "Uno contra todos".
La clasificación uno contra todos es un método que involucra entrenamiento con N clasificadores binarios distintos, cada uno diseñado para reconocer una clase particular. One-vs-All es derivado de una reducción de aprendizaje automatizada para poder realizar una clasificación multiclase dado un clasificador base que puede realizar la clasificación binaria de manera eficiente.
OneVsRestse implementa como un Estimator. Para el clasificador base, toma instancias de Classifiery crea un problema de clasificación binaria para cada una de las k clases. El clasificador para la clase i está entrenado para predecir si la etiqueta es i o no, distinguiendo la clase i de todas las demás clases.

Las predicciones se realizan evaluando cada clasificador binario y el índice del clasificador más seguro se genera como etiqueta.

###Como funciona: 
Importamos las bibliotecas y paquetes necesarios para cargar el programa.
* importar  org.apache.spark.ml.classification. { LogisticRegression ,  OneVsRest } 
* import  org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
* import org.apache.spark.sql.SparkSession

Creamos una instancia de la sesion de spark
* val spark = SparkSession.builder.appName("OneVsRestExample").getOrCreate()

Carga de nuestro dataset 

* val Data = spark.read.format("libsvm").load("sample_multiclass_classification_data.txt")
Data.show()

Posteriormente realizamos nuestro entrenamiento con nuestros datos de la siguiente mandera:
 divida los datos en conjuntos de entrenamiento y prueba por medio de un arreglo (20% para pruebas y 80% de entrenamiento).
* val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

Instanciamos la base del clasificador el cual contenbra la maxima de interacciones la tolerania y el ajuste de intercepciones.
*  val classifier = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)

Generamos las instancias de OneVsRest el cual nos va traer el clasificador
* val ovr = new OneVsRest().setClassifier(classifier)

Entrenamiento del modelo multiclases generando un ajuste a los datos de entrenamiento
* val ovrModel = ovr.fit(train)

Transformamos los datos de prueba en el metodo de prediccion
* val predictions = ovrModel.transform(test)

generamos el evaluador el cual traeremos la instancia nombre de la metrica
* val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

calcula el error de clasificación en los datos de prueba.
* val accuracy = evaluator.evaluate(predictions)
* println(s"Test Error = ${1 - accuracy}")
                accuracy: Double = 0.9642857142857143
                Test Error = 0.0357142857142857 //el error de prediccion es muy bajo por lo que se puede decir que la prediccion que el algoritmo hace es muy buena.
#### Authors: HERNANDEZ GARCIA RIGOBERTO
#### Titule: Clasificador One-vs-Rest
