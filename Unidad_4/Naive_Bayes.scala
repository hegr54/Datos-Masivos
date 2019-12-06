import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import java.time._
// $example off$
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("NaiveBayes").getOrCreate()

    // $example on$
    // Load the data stored in LIBSVM format as a DataFrame.
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
val data = assembler.transform(newcolumn)
//Mostramos la nueva columna
data.show(1)

//Cambiamos la columna y a la columna label
val Renamed = data.withColumnRenamed("y", "label")
val data = Renamed.select("label","features")
data.show(1)  
    // Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = data.randomSplit(Array(0.65, 0.35), seed = 1234L)

    // Train a NaiveBayes model.
val model = new NaiveBayes().fit(trainingData)

    // Select example rows to display.

val predictions = model.transform(testData)

    // Select (prediction, true label) and compute test error
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test set accuracy = $accuracy")
println(s"Test Error = ${(1.0 - accuracy)}")
val t1 = System.currentTimeMillis
println(s"Time: $t1")