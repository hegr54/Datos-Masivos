//Importación del algoritmo Multilayer Perception Classifier y importamos las librerías importantes para generar 
//1.- hacer la limpieza de datos necesaria para poder ser procesado con el siguiente algoritmo
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
// se crea una variable que contendrá el objeto struct type el cual puede tener uno o más structfields se puede extraer por nombres, en caso de extraer varios
// structfield este se volvera un objeto structtype. Si uno de estos nombres que nosotros proporcionamos no contiene un campo
//por coincidente se ignorará. para este caso de extraer un solo Structfield se devolverá un valor nulo.

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

//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1) 
//función de entrenamiento el cual será 60% de entrenamiento y 40% de prueba con una profundidad o semilla de 1234L.
val splits = feat.randomSplit(Array(0.65, 0.35), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

// Especificación de nuestra capa de red neuronal de la siguiente manera:
//la capa de entrada será de tamaño 4 características, dos capas oculta de tama;o 5 y 4, así tendremos una salida de tama;o de 3 clases.
val layers = Array[Int](5, 5, 4, 3)
//creamos el entrenador multilayerPercptronClassifier y establecimiento de sus parámetros.
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
//e.- explique la función de error que utilizó para el resultado final
val model = trainer.fit(train)
//calculamos la predicción en el conjunto de prueba.

//Imprimimos la exactitud
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
println(s"Test Error = ${(1.0 - evaluator.evaluate(predictionAndLabels))}")
val t1 = System.currentTimeMillis
println(s"Time: $t1")