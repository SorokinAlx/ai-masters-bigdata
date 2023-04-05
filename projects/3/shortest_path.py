import os
import sys

SPARK_HOME = "/usr/lib/spark3"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark import SparkConf
from pyspark.sql import SparkSession

conf = SparkConf()

spark = SparkSession.builder.config(conf=conf).appName("PageRank_alsor").getOrCreate()

from pyspark.sql.types import *
schema = StructType(fields=[
    StructField("user_id", IntegerType()),
    StructField("follower_id", IntegerType()),
])

df = spark.read\
          .schema(schema)\
          .format("csv")\
          .option("sep", "\t")\
          .load(sys.argv[3])

def bf_search(df, start, stop):
    queue = []
    parents = {}
    cur_el = start
    success = 0
    queue.append(cur_el)
    while queue:
        cur_el = queue.pop(0)
        children = df.filter(df.follower_id == cur_el).select(df.user_id).rdd.flatMap(lambda x: x).collect()
        if stop in children:
            success = 1
            break
        for el in children:
            if parents.get(el, -1) == -1:
                parents[el] = cur_el
                queue.append(el)
    if (success == 1):
        res = []
        while cur_el != start:
            res.append(cur_el)
            cur_el = parents[cur_el]
    return res

start = int(sys.argv[1])
stop = int(sys.argv[2])

res = bf_search(df, start, stop)

res = [stop] + res + [start]

print(res[::-1])

df_res = spark.createDataFrame(data=res[::-1])
df_res.write.csv(sys.argv[4], sep=',')