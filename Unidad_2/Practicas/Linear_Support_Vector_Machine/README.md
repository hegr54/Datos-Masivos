### Algoritmos de:

#### Máquina lineal de vectores de soporte
### Que es el algoritmo Máquina lineal de vectores de soporte?

son modelos de aprendizaje supervisados con algoritmos de aprendizaje asociados que analizan los datos utilizados para el análisis de clasificación y regresión 

###Como funciona: 
Importamos las bibliotecas y paquetes necesarios para cargar el programa.
* import org.apache.spark.ml.classification.LinearSVC 
* import org.apache.spark.sql.SparkSession

Creamos una instancia de la sesion de spark
* val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()

Una máquina de vectores de soporte construye un hiperplano o un conjunto de hiperplanos en un espacio de dimensión alta o infinita, que puede usarse para clasificación, regresión u otras tareas. Intuitivamente, se logra una buena separación mediante el hiperplano que tiene la mayor distancia a los puntos de datos de entrenamiento más cercanos de cualquier clase (denominado margen funcional), ya que en general cuanto mayor es el margen, menor es el error de generalización del clasificador.

* val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")

LinearSVC en Spark ML admite la clasificación binaria con SVM lineal. Internamente, optimiza la pérdida de la bisagra utilizando el optimizador OWLQN.Donde se va traer el maximo de interacciones las cuales seran 10 interaciones y un parametro de 1.
*   val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

Modelo de ajuste de entrenamiento.
* val lsvcModel = lsvc.fit(training)

Imprimimos los coeficientes e intercepta para svc. 
* println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

#### Authors: HERNANDEZ GARCIA RIGOBERTO
#### Titule: Máquina lineal de vectores de soporte
