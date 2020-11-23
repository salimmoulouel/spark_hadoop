import sys
from pyspark import SparkContext

sc = SparkContext()
lines = sc.textFile(sys.argv[1])
word_counts = lines.flatMap(lambda line: line.split(' ')) \
                   .map(lambda word: (word, 1)) \
                   .reduceByKey(lambda count1, count2: count1 + count2) \
                   .collect()

for (word, count) in word_counts:
    print(word, count)

#apply the following command on spark and replace 4 with the number of nodes you want to be parallelised
# spark-3.0.1-bin-hadoop2.7/bin/spark-submit --master local[4] ./wordcount.py ./iliad100.txt  

