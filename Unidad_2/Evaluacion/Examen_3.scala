//Importación del algoritmo Multilayer Perception Classifier y importamos las librerías importantes para generar 
//1.- hacer la limpieza de datos necesaria para poder ser procesado con el siguiente algoritmo
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types._
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
//Creación de la sección de spark.
val Spark= SparkSession.builder().getOrCreate()
//carga y lectura de nuestro dataset en un formato csv.
val dataset = spark.read.option("header","true").option("inferSchema", "true")csv("Iris.csv")
//visualización de nuestro data set.
dataset.show()
//Visualización del esquema o como se encuentran nuestros datos.
dataset.printSchema()
// se crea una variable que contendrá el objeto struct type el cual puede tener uno o más structfields se puede extraer por nombres, en caso de extraer varios
// structfield este se volvera un objeto structtype. Si uno de estos nombres que nosotros proporcionamos no contiene un campo
//por coincidente se ignorará. para este caso de extraer un solo Structfield se devolverá un valor nulo.

val datasetSchema =  /// estructura de un StructType
StructType(
StructField("Cinco/uno", DoubleType, true) :: /// el nombre del campo, tipo de dato en el que corresponda e indicar si los valores de este capo serán valores nulos.
StructField("Tres/tres", DoubleType, true) ::///los metadatos de este campo deben de conservarse durante la transformación si el contenido de la columna no se modifican.
StructField("Uno/cuatro", DoubleType, true) ::
StructField("Cero/dos",DoubleType, true) ::
StructField("Iris-setosa", StringType, true) :: Nil)
/// carga y análisis de Nuestro dataset ya completamente listo para trabajar sobre el.
val dataset2 = spark.read.option("header", "false").schema(datasetSchema)csv("Iris.csv")
dataset2.columns
//lectura de nuestro dataset ya limpio.
// creación de nuestra etiqueta con la que se trabajara en este algoritmo.
 val labelIndexer = new StringIndexer().setInputCol("Iris-setosa").setOutputCol("label").fit(dataset2)
 //características con las que cuenta nuestra dataset en cual será un nuevo vector Assembler el cual contendrá nuestra entrada de datos con un arreglo el cual contendrá nuestras características y tendremos una salida de "factures" características.
 val featureIndexer = new VectorAssembler().setInputCols(Array("Cinco/uno", "Tres/tres", "Uno/cuatro", "Cero/dos")).setOutputCol("features")

//función de entrenamiento el cual será 60% de entrenamiento y 40% de prueba con una profundidad o semilla de 1234L.
val splits = dataset2.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

// Especificación de nuestra capa de red neuronal de la siguiente manera:
//la capa de entrada será de tamaño 4 características, dos capas oculta de tama;o 5 y 4, así tendremos una salida de tama;o de 3 clases.
val layers = Array[Int](4, 5, 4, 3)
//creamos el entrenador multilayerPercptronClassifier y establecimiento de sus parámetros.
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setBlockSize(128).setSeed(1234L).setMaxIter(100)
//e.- explique la función de error que utilizó para el resultado final
val pipeline = new Pipeline().setStages(Array(labelIndexer,featureIndexer,trainer))
//entrenamos el modelo de clasificación de perceptrón multicapa utilizando el estimador anterior.
val model = pipeline.fit(train)
//calculamos la predicción en el conjunto de prueba.
val result = model.transform(test)
//impresión de resultados del cálculo del conjunto de prueba.
result.show()
//Evaluación del modelo para el rendimiento de predicción.
val predictionAndLabels = result.select("prediction", "label")
predictionAndLabels.show()
// Evaluador de nuestra predicción.
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
//Salida de la prueba de predicción.
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")