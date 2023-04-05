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

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

conf = SparkConf()

sc = SparkContext(appName="PageRank", conf=conf)
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
    queue.append((start,0))
    stop_crit = -1
    while queue:
        cur_el, level = queue.pop(0)
        children = df.filter(df.follower_id == cur_el).select(df.user_id).rdd.flatMap(lambda x: x).collect()
        if stop in children and stop_crit==-1:
            stop_crit = level
        if stop_crit > -1 and  level > stop_crit:
            break
        for el in children:
            if parents.get(el, [(1000,1000)])[0][1] >= level:
                if parents.get(el, [(-1,-5)])[0][1] == -5:
                    parents[el] = [(cur_el, level)]
                    queue.append((el,level+1))
                else:
                    parents[el].append((cur_el, level))
    return parents

def get_res(current, target, parents):
    if current == target:
        return [[target]]
    lst = []
    for parent in parents[current]:
        cur = get_res(parent[0],target, parents)
        lst.extend(cur)
    for el in lst:
        el.append(current)
    return lst

start = int(sys.argv[1])
stop = int(sys.argv[2])

parents = bf_search(df, start, stop)

res = get_res(stop, start, parents)

result = spark.createDataFrame(res1)
result.write.csv(sys.argv[4], sep=',')
