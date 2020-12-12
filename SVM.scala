import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

val data = MLUtils.loadLibSVMFile(sc, "dataset.libsvm")

val splits = data.randomSplit(Array(0.8, 0.2))
val training = splits(0).cache()
val test = splits(1)

val numIterations = 500
val model = SVMWithSGD.train(training, numIterations)

model.clearThreshold()
val scoreAndLabels = test.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}

val metrics = new BinaryClassificationMetrics(scoreAndLabels)

val auROC = metrics.areaUnderROC()
println(s"Area under ROC = $auROC")

val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")