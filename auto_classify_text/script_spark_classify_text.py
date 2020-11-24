#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:26:47 2020

@author: sparkuser
"""

# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import pyspark
import pyspark.ml.feature
import pyspark.ml.classification

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer,HashingTF,IDF,Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification  import NaiveBayes
from pyspark.ml.feature import StringIndexer

sc = SparkContext()

spark = SparkSession.builder. \
      master("local") \
      .appName("spark session example") \
      .getOrCreate()

# telecharger les données de train et test 20ng all terms depuis http://ana.cachopo.org/datasets-for-single-label-text-categorization/

from pyspark.sql import Row

def load_dataframe(path):
    rdd = sc.textFile(path)\
        .map(lambda line : line.split())\
        .map(lambda words : Row(label=words[0], words=words[1:]))
    return spark.createDataFrame(rdd)


def load_dataframe_text_unsplitted(path):
    rdd = sc.textFile(path)\
        .map(lambda line : line.split())\
        .map(lambda words : Row(label=words[0], sentence=' '.join(words[1:])))
    return spark.createDataFrame(rdd)

        
train_data= load_dataframe("20ng-train-all-terms.txt")
test_data= load_dataframe("20ng-test-all-terms.txt")
    

#vectorizer = CountVectorizer(inputCol="words", outputCol="bag_of_words")
#vectorizer_transformer= vectorizer.fit(train_data)
#
#train_bag_of_words = vectorizer_transformer.transform(train_data)
#test_bag_of_words = vectorizer_transformer.transform(test_data)
#
#train_data.select("label").distinct().sort("label").show(truncate=False)
#
#
#label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
#label_indexer_transformer = label_indexer.fit(train_bag_of_words)
#
#from pyspark.ml.feature import StringIndexer
#label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
#label_indexer_transformer = label_indexer.fit(train_bag_of_words)
#
#train_bag_of_words = label_indexer_transformer.transform(train_bag_of_words)
#test_bag_of_words = label_indexer_transformer.transform(test_bag_of_words)
#
#classifier_transformer = pyspark.ml.classification.NaiveBayes(labelCol="label_index", featuresCol="bag_of_words",predictionCol="label_index_predicted").fit(train_bag_of_words)
#test_predicted = classifier_transformer.transform(test_bag_of_words)
#
#test_predicted.select("label_index", "label_index_predicted").limit(10).show()
#
#accuracy = test_predicted.filter(test_predicted.label_index_predicted == test_predicted.label_index).count() / float(test_predicted.count())
#print("accuracy {}".format(accuracy))


#utilisation d'une pipeline avec bag of words pour simplifier et fluidifier l'ecriture


vectorizer = CountVectorizer(inputCol="words", outputCol="bag_of_words")
label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
classifier = NaiveBayes(
    labelCol="label_index", featuresCol="bag_of_words", predictionCol="label_index_predicted",
)
pipeline = Pipeline(stages=[vectorizer, label_indexer, classifier])
pipeline_model = pipeline.fit(train_data)

test_predicted = pipeline_model.transform(test_data)



#utilisation d'une pipeline avec tf-idf

train_data_2= load_dataframe_text_unsplitted("20ng-train-all-terms.txt")
test_data_2= load_dataframe_text_unsplitted("20ng-test-all-terms.txt")


vectorizer = CountVectorizer(inputCol="words", outputCol="bag_of_words")

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")

idf = IDF(inputCol="rawFeatures", outputCol="bag_of_words")


#pipeline_tf_idf = Pipeline(stages=[label_indexer, HashingTF, IDF,  classifier])
pipeline_tf_idf = Pipeline(stages=[label_indexer,tokenizer,hashingTF,idf,classifier])
pipeline_model_tf_idf = pipeline_tf_idf.fit(train_data_2)
test_predicted_tf_idf = pipeline_model_tf_idf.transform(test_data_2)


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="label_index_predicted", metricName="accuracy")
accuracy = evaluator.evaluate(test_predicted)
print("Accuracy = {:.2f}".format(accuracy))

accuracy_tf_idf = evaluator.evaluate(test_predicted_tf_idf)
print("Accuracy = {:.2f} avec tf-idf".format(accuracy_tf_idf))


# PYSPARK_PYTHON=python3 $SPARK_HOME/bin/spark-submit --master local[4] ./script_spark_classify_text.py 


