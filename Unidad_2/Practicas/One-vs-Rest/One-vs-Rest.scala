import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * An example of Multiclass to Binary Reduction with One Vs Rest,
 * using Logistic Regression as the base classifier.
 * Run with
 * {{{
 * ./bin/run-example ml.OneVsRestExample
 * }}}
 */

val spark = SparkSession.builder.appName(s"OneVsRestExample").getOrCreate()

    // $example on$
    // load data file.
    val inputData = spark.read.format("libsvm").load("sample_multiclass_classification_data.txt")
inputData.show()
    // generate the train/test split.
    val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

    // instantiate the base classifier
    val classifier = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)

    // instantiate the One Vs Rest Classifier.
    val ovr = new OneVsRest().setClassifier(classifier)

    // train the multiclass model.
    val ovrModel = ovr.fit(train)

    // score the model on test data.
    val predictions = ovrModel.transform(test)

    // obtain evaluator.
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    // compute the classification error on test data.
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1 - accuracy}")
    // $example off$
