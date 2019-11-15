//Importacion del algoritmo Multilayer Perceotion Classifier y importamos las librerias importantes para generar 
//1.- hacer la limpieza de datos necesaria para poder ser procesadocon el siguiente algoritmo
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
//Creacion de la secion de spark.
val Spark= SparkSession.builder().getOrCreate()
//caraga y lectura de nuestro data set en un formato csv.
val dataset = spark.read.option("header","true").option("inferSchema", "true")csv("Iris.csv")
//visualizacion de nuestro data set.
dataset.show()
//Visualizacion del esque o como se encuentran nuestros datos.
dataset.printSchema()
// se crea una variabje que contenbra el objeto structtype el cual puede tener uno o mas structfields se puede extraer por nombres, en caso de extraer varios
// structfield este se volvera un objeto structtype. Si uno de estos nombres que nosotros proporcionamos no contiene un campo
//por coincidente se ignorara. para este caso de ectraer un solo Structfield se devolvera un valor nulo.

val datasetSchema =  /// estructura de un StructType
StructType(
StructField("Cinco/uno", DoubleType, true) :: /// el nombre del campo, tipo de dato en el que corresponde e incar si los valores de este capo seran valores nulos.
StructField("Tres/tres", DoubleType, true) ::///los metadatos de este campo deben de conservase durante la trasformacion si el contenido de la columna no se modifican.
StructField("Uno/cuatro", DoubleType, true) ::
StructField("Cero/dos",DoubleType, true) ::
StructField("Iris-setosa", StringType, true) :: Nil)
/// carga y amalisis de Nuestro data set ya completamente listo para trabajar sobre el.
val dataset2 = spark.read.option("header", "false").schema(datasetSchema)csv("Iris.csv")
dataset2.columns
//lectura de nuestro data set ya limpio.
// creacion de nuestra etique con la que se trabajara en este algoritno.
 val labelIndexer = new StringIndexer().setInputCol("Iris-setosa").setOutputCol("label").fit(dataset2)
 //caracteristicas con las que cuenta nuestra data set en cual sera un nuevo vectorAssember el cual contenbra nuestra entrada de datos con un arreglo el cual contendra nuestras caracteristicas y tendremos una salida de "factures" caracteristicas.
 val featureIndexer = new VectorAssembler().setInputCols(Array("Cinco/uno", "Tres/tres", "Uno/cuatro", "Cero/dos")).setOutputCol("features")

//funcion de entrenamiento el cual sera 60% de entrenamiento y 40% de prueba con una profundidad o semilla de 1234L.
val splits = dataset2.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

// Especificacion de nuestra capa de red neuronal de la siguiente manera:
//la capa de entrada sera de tama;o 4 caracteristicas, dos capas oculta de tama;o 5 y 4, asi tendremos una salida de tama;o de 3 clases.
val layers = Array[Int](4, 5, 4, 3)
//creamos el entrenador multilayerPercptronClassifier y establecimiento de sus parametros.
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setBlockSize(128).setSeed(1234L).setMaxIter(100)
//e.- explique la funcion de error que utilizo para el resultado final
val pipeline = new Pipeline().setStages(Array(labelIndexer,featureIndexer,trainer))
//entrenamos el modelo de clasificacion de perceptron multicapa utilizando el estimador anterior.
val model = pipeline.fit(train)
//calculamos la predicion en el conjunto de prueba.
val result = model.transform(test)
//impresion de resultados del calculo del conjunto de prueba.
result.show()
//Evaluacion del modelo para el rendimiento de predicion.
val predictionAndLabels = result.select("prediction", "label")
predictionAndLabels.show()
// Evaluador de nuestra predicion.
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
//Salida de la prueba de predicion.
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

