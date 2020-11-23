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

sc = SparkContext()

spark = SparkSession.builder. \
      master("local") \
      .appName("spark session example") \
      .getOrCreate()
      
iliad = sc.textFile("iliad.mb.txt"). \
map(lambda line : line.split()). \
map(lambda words: [w.strip(',.;:?!-"') for w in words])

for line in iliad.takeSample(False,10):
    print(line)
    
odyssey = sc.textFile("odyssey.mb.txt"). \
map(lambda line : line.split()). \
map(lambda words: [w.strip(',.;:?!-"') for w in words])

for line in iliad.takeSample(False,10):
    print(line)
    
iliad = iliad.map(lambda words: pyspark.sql.Row(label=0, words=words))
odyssey= odyssey.map(lambda words: pyspark.sql.Row(label=1, words=words))

data = spark.createDataFrame(iliad.union(odyssey))


vectorizer = pyspark.ml.feature.CountVectorizer(inputCol="words",outputCol="bag_of_words").fit(data)
features = vectorizer.transform(data)
train, test= features.randomSplit([0.75,0.25])

classifier = pyspark.ml.classification.NaiveBayes(labelCol="label", featuresCol="bag_of_words",predictionCol="label_predicted").fit(train)
predicted = classifier.transform(test)
accuracy = predicted.filter(predicted.label_predicted == predicted.label).count() / float(predicted.count())
print("accuracy {}".format(accuracy))

#PYSPARK_PYTHON=python3 $SPARK_HOME/bin/spark-submit --master local[4] ./iliad_detector.py 
