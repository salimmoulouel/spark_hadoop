#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:05:32 2020

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

sc = SparkContext()

spark = SparkSession.builder. \
      master("local") \
      .appName("spark session example") \
      .getOrCreate()
      
agents = spark.read.json("agents.json")

frenchagents=agents.filter(agents.country_name=="France")

agent = frenchagents.first()
print(agent.country_name, agent.id)

agents.filter((agents.country_name == "France") & (agents.latitude < 0)).count()

agents.createTempView("agents_table")
spark.sql("SELECT * FROM agents_table ORDER BY id DESC LIMIT 10").show()

agents.persist()

agents.rdd.filter(lambda row: row.country_name == "France").count()


from pyspark.sql import Row
rdd = sc.parallelize([Row(name="Alice"), Row(name="Bob")])
spark.createDataFrame(rdd)



#download the agent db
#wget https://s3-eu-west-1.amazonaws.com/course.oc-static.com/courses/4297166/agents.json
#PYSPARK_PYTHON=python3 $SPARK_HOME/bin/spark-submit --master local[4] ./word_population_queries.py 
