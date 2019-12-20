/**
 * Acknowledgement: the implementation is based on Spark native tutorial for ML:
 * --- https://spark.apache.org/docs/2.2.0/ml-features.html
 * --- https://spark.apache.org/docs/2.2.0/ml-classification-regression.html#linear-support-vector-machine
 * --- https://spark.apache.org/docs/latest/ml-tuning.html
 */

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LinearSVC, LogisticRegression, NaiveBayes}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover, Tokenizer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.{DataFrame, SparkSession}

object spam {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.master("local").appName("Classification").getOrCreate()
    import spark.implicits._

    // =======================================================
    // Preparation - Load data as spark-datasets
    // =======================================================

    val spam_training = spark.read.textFile("src/main/resources/spam_training.txt")
    val spam_testing = spark.read.textFile("src/main/resources/spam_testing.txt")
    val nospam_training = spark.read.text("src/main/resources/nospam_training.txt")
    val nospam_testing = spark.read.text("src/main/resources/nospam_testing.txt")

    // ======================================================
    // Preparation - Convert datasets into dataframes
    // =======================================================

    // Label data as 0 (= no spam) and insert "train" column so that it can be distinguished from test data
    val noSpam_train_records = nospam_training.map(t => (t.toString(), 0, "train"))
    val spam_train_records = spam_training.map(t => (t.toString(), 1, "train"))
    //Union all training data (spam & no spam) into one dataset
    val training_dataset = noSpam_train_records.union(spam_train_records)
    //Convert training dataset into DataFrame with corresponding columns
    val training_df = training_dataset.toDF("sentences", "label", "dataset")

    // Label data as 1 (= spam) and insert "test" column so that it can be distinguished from train data
    val noSpam_test_records = nospam_testing.map(t => (t.toString(), 0, "test"))
    val spam_test_records = spam_testing.map(t => (t.toString(), 1, "test"))
    //Union all testing data (spam & no spam) into one dataset
    val testing_dataset = noSpam_test_records.union(spam_test_records)
    //Convert testing dataset into DataFrame with corresponding columns
    val testing_df = testing_dataset.toDF("sentences", "label", "dataset")

    // =======================================================
    // Feature engineering: Bag of Words, URLs, TF-IDF
    // =======================================================

    import org.apache.spark.sql.functions._

    // set a UDF to check whether a sentence contains a url
    val isUrl: String => Boolean = { t => t.contains("www.") || t.contains("http") }
    val isUrlUDF = udf(isUrl)

    val training_url = training_df.withColumn("url", isUrlUDF(training_df.col("sentences")))
    val testing_url = testing_df.withColumn("url", isUrlUDF(testing_df.col("sentences")))

    // set a UDF to check whether a sentence contains phone number
    //    val regexStr: String = "^[0-9]{10}$"
    //    val pattern = Pattern.compile(regexStr)
    //    val hasNumber: String => Boolean = { t => { val matcher = pattern.matcher(t.toString)
    //                                                return matcher.matches()}
    //                                        }
    //    val isNumber = udf(hasNumber)
    //
    //    val training_number = training_url.withColumn("number", isNumber(training_df.col("sentences")))
    //    val testing_number = testing_url.withColumn("number", isNumber(testing_df.col("sentences")))

    val tokenizer = new Tokenizer().setInputCol("sentences").setOutputCol("words")

    // stopwords elimination
    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("words_noStopWords")

    // stemming
    val stemmer = new Stemmer()
      .setInputCol("words_noStopWords")
      .setOutputCol("words_filtered")
      .setLanguage("English")

    // applying Hashing TF to convert input words into feature vectors
    val hashingTF = new HashingTF().setInputCol("words_filtered").setOutputCol("rawFeatures").setNumFeatures(2000)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("idf_features")

    val assembler = new VectorAssembler()
      .setInputCols(Array("idf_features", "url"))
      .setOutputCol("features")

    // =======================================================
    // Classification methods: Naive Bayes, SVM, Decision Tree
    // - Training set is used to train model
    // - Testing set is used to test the model's predictions
    // =======================================================

    // - Naive Bayes
    val bayes = new NaiveBayes()
    //    val bayes_model = new NaiveBayes().fit(training_data)
    //    val bayes_predictions = bayes_model.transform(testing_data)

    // - SVM
    val lsvc = new LinearSVC()
    //    val svm = lsvc.setMaxIter(10).setRegParam(0.1)

    // - Logistic Regression
    val lr = new LogisticRegression()
    //    val lr_model = lr
    //      .setMaxIter(10)
    //      .setRegParam(0.3)
    //      .setElasticNetParam(0.8)
    //      .setFamily("multinomial")

    // - Decision tree
    val dt = new DecisionTreeClassifier()
    //    val dt_model = dt
    //      .setLabelCol("label")
    //      .setFeaturesCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, stemmer, hashingTF, idf, assembler, lsvc))

    //    val model = pipeline.fit(training_url)
    //    val test = model.transform(testing_url)

    // =======================================================
    // Cross validation
    // =======================================================

    // lr
    //    val paramGrid = new ParamGridBuilder()
    //      .addGrid(hashingTF.numFeatures, Array(5000))
    //      .addGrid(lr.maxIter, Array(1000))
    //      .addGrid(lr.regParam, Array(0.2, 0.5, 0.7))
    //      .build()

    // svm
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(5000))
      .addGrid(lsvc.maxIter, Array(2000))
      .addGrid(lsvc.regParam, Array(0.0, 0.1, 0.2, 0, 3, 0.5, 0.7))
      .build()

    // bayes / decision tree
    //    val paramGrid = new ParamGridBuilder()
    //      .addGrid(hashingTF.numFeatures, Array(5000))
    //      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    val cvModel = cv.fit(training_url)
    val result = cvModel.transform(testing_url)

    //    printMetric("accuracy", result, "Linear Regression")
    //    printMetric("f1", result, "Linear Regression")

    //    printMetric("accuracy", result, "Naive Bayes")
    //    printMetric("f1", result, "Naive Bayes")

    printMetric("accuracy", result, "SVM")
    printMetric("f1", result, "SVM")

  }

  def printMetric(metric: String, model_predictions: DataFrame, model: String): Unit = {
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName(metric)
    val result = evaluator.evaluate(model_predictions)
    println(model + " classifier | " + metric + "=\t" + result)
  }

}
