//Importacion del algoritmo Multilayer Perceotion Classifier y importamos las librerias importantes para generar 
//la limpieza de los datos para poder procesarlos.
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

val Spark= SparkSession.builder().getOrCreate()

val dataset = spark.read.option("header","true").option("inferSchema", "true")csv("Iris.csv")

dataset.show()

dataset.printSchema()

val datasetSchema =
StructType(
StructField("Cinco/uno", DoubleType, true) ::
StructField("Tres/tres", DoubleType, true) ::
StructField("Uno/cuatro", DoubleType, true) ::
StructField("Cero/dos",DoubleType, true) ::
StructField("Iris-setosa", StringType, true) :: Nil)

val dataset2 = spark.read.option("header", "false").schema(datasetSchema)csv("Iris.csv")
dataset2.columns

 val labelIndexer = new StringIndexer().setInputCol("Iris-setosa").setOutputCol("label").fit(dataset2)
 val featureIndexer = new VectorAssembler().setInputCols(Array("Cinco/uno", "Tres/tres", "Uno/cuatro", "Cero/dos")).setOutputCol("features")


val splits = dataset2.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

val layers = Array[Int](4, 5, 4, 3)
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setBlockSize(128).setSeed(1234L).setMaxIter(100)

val pipeline = new Pipeline().setStages(Array(labelIndexer,featureIndexer,trainer))
val model = pipeline.fit(train)
val result = model.transform(test)
result.show()
val predictionAndLabels = result.select("prediction", "label")
predictionAndLabels.show()

val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

