import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{LBFGS, LogisticGradient, SquaredL2Updater}
import org.apache.spark.mllib.util.MLUtils

val data = MLUtils.loadLibSVMFile(sc, "dataset.libsvm")
val numFeatures = data.take(1)(0).features.size

val splits = data.randomSplit(Array(0.8, 0.2))
val training = splits(0).map(x => (x.label, MLUtils.appendBias(x.features))).cache()

val test = splits(1)

val numCorrections = 10
val convergenceTol = 1e-4
val maxNumIterations = 20
val regParam = 0.1
val initialWeightsWithIntercept = Vectors.dense(new Array[Double](numFeatures + 1))

val (weightsWithIntercept, loss) = LBFGS.runLBFGS(
  training,
  new LogisticGradient(),
  new SquaredL2Updater(),
  numCorrections,
  convergenceTol,
  maxNumIterations,
  regParam,
  initialWeightsWithIntercept)

val model = new SVMModel(Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)),weightsWithIntercept(weightsWithIntercept.size - 1))


model.clearThreshold()
val scoreAndLabels = test.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}


val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val auROC = metrics.areaUnderROC()

println("Loss of each step in training process")
loss.foreach(println)
println(s"Area under ROC = $auROC")

val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")