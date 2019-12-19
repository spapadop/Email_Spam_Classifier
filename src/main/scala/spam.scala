/*
Acknowledgement: the implementation is based on Spark native tutorial for ML:
- https://spark.apache.org/docs/2.2.0/ml-features.html
- https://spark.apache.org/docs/2.2.0/ml-classification-regression.html#linear-support-vector-machine
 */

import org.apache.spark.ml.classification.{DecisionTreeClassifier, LinearSVC, LogisticRegression, NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, HashingTF, IDF, NGram, StopWordsRemover, Tokenizer, VectorAssembler}
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object spam {


  def main(args: Array[String]): Unit = {

    // start spark session
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
    // Feature representation: Bag of Words
    // =======================================================

    val tokenizer = new Tokenizer().setInputCol("sentences").setOutputCol("words")
    val training_words = tokenizer.transform(training_df) // tokenize the training set into words
    val testing_words = tokenizer.transform(testing_df) // tokenize the testing set into words

    // stopwords elimination
    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("words_noStopWords")

    val training_words_noStopWords = remover.transform(training_words) //remove stopwords from training set
    val testing_words_noStopWords = remover.transform(testing_words) //remove stopwords from test set

    // stemming
    val stemmer = new Stemmer()
      .setInputCol("words_noStopWords")
      .setOutputCol("words_filtered")
      .setLanguage("English")

    val training_words_filtered = stemmer.transform(training_words_noStopWords)
    val testing_words_filtered = stemmer.transform(testing_words_noStopWords)

    // applying Hashing TF to convert input words into feature vectors
    val hashingTF = new HashingTF().setInputCol("words_filtered").setOutputCol("rawFeatures").setNumFeatures(20)

    // =======================================================
    // Feature weighting: TF-IDF
    // =======================================================

    val training_featured_data = hashingTF.transform(training_words_filtered) // calculate TF for training set words
    val testing_featured_data = hashingTF.transform(testing_words_filtered) // calculate TF for testing set words

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val training_IDF_model = idf.fit(training_featured_data) // calculate IDF for training featured data
    val testing_IDF_model = idf.fit(testing_featured_data) // calculate IDF for testing featured data

    val training_data = training_IDF_model.transform(training_featured_data)
    val testing_data = testing_IDF_model.transform(testing_featured_data)
    //    training_data.select("label", "features").show()
    //    testing_data.select("label", "features").show()

    // =======================================================
    // Classification methods: Naive Bayes, SVM, Decision Tree
    // - Training set is used to train model
    // - Testing set is used to test the model's predictions
    // =======================================================

    // - Naive Bayes
    val bayes_model = new NaiveBayes().fit(training_data)
    val bayes_predictions = bayes_model.transform(testing_data)

    // - SVM
    val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
    val svm_model = lsvc.fit(training_data)
    val svm_predictions = svm_model.transform(testing_data)

    // - Logistic Regression
    //    val mlr = new LogisticRegression()
    //      .setMaxIter(10)
    //      .setRegParam(0.3)
    //      .setElasticNetParam(0.8)
    //      .setFamily("multinomial")
    //
    //    val mlrModel = mlr.fit(training_data)
    //    val lr_predictions = mlrModel.transform(testing_data)

    // - Decision tree
    val dt = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
    //      .setNumTrees(10)

    val dt_model = dt.fit(training_data)
    val dt_predictions = dt_model.transform(testing_data)

    // =======================================================
    // Classifier Evaluation:
    // - Precision
    // - Recall
    // - Accuracy
    // - F1-score
    // =======================================================

    printMetric("accuracy", bayes_predictions, "Bayes")
    printMetric("weightedPrecision", bayes_predictions, "Bayes")
    printMetric("weightedRecall", bayes_predictions, "Bayes")
    printMetric("f1", bayes_predictions, "Bayes")

    printMetric("accuracy", svm_predictions, "SVM")
    printMetric("weightedPrecision", svm_predictions, "SVM")
    printMetric("weightedRecall", svm_predictions, "SVM")
    printMetric("f1", svm_predictions, "SVM")

    printMetric("accuracy", dt_predictions, "Random Forest")
    printMetric("weightedPrecision", dt_predictions, "Random Forest")
    printMetric("weightedRecall", dt_predictions, "Random Forest")
    printMetric("f1", dt_predictions, "Random Forest")

    //    printMetric("accuracy",lr_predictions, "Linear Regression")
    //    printMetric("weightedPrecision",lr_predictions, "Linear Regression")
    //    printMetric("weightedRecall",lr_predictions, "Linear Regression")
    //    printMetric("f1",lr_predictions, "Linear Regression")

  }

  def printMetric(metric: String, model_predictions: DataFrame, model: String): Unit = {
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName(metric)
    val result = evaluator.evaluate(model_predictions)
    println(model + " classifier | " + metric + "=\t" + result)

  }

  def buildNgrams(inputCol: String = "word_filtered",
                  outputCol: String = "features", n: Int = 3) = {

    val ngrams = (1 to n).map(i =>
      new NGram().setN(i)
        .setInputCol(inputCol).setOutputCol(s"${i}_grams")
    )

    val vectorizers = (1 to n).map(i =>
      new CountVectorizer()
        .setInputCol(s"${i}_grams")
        .setOutputCol(s"${i}_counts")
    )

    val assembler = new VectorAssembler()
      .setInputCols(vectorizers.map(_.getOutputCol).toArray)
      .setOutputCol(outputCol)

//    new Pipeline().setStages((ngrams ++ vectorizers :+ assembler).toArray)

  }

  val df = Seq((1, Seq("a", "b", "c", "d"))).toDF("id", "tokens")
}
