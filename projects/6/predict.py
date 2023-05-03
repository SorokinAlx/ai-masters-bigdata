#!/opt/conda/envs/dsenv/bin/python
from joblib import load
import pandas as pd
import pyspark.sql.functions as f
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, BooleanType, TimestampType
from pyspark.ml.linalg import VectorUDT


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
from pyspark.sql.types import *

conf = SparkConf()

spark = SparkSession.builder.config(conf=conf).appName("SparkML_hw").getOrCreate()

model = load(sys.argv[6])

model = spark.sparkContext.broadcast(model)

@f.pandas_udf(FloatType())
def predict(vector):
    pred = model.value.predict(vector.tolist())
    return pd.Series(pred)



schema = StructType([
    StructField("vote", StringType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("id", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", TimestampType()),
    StructField("words", VectorUDT()),
    StructField("words_without_stop", VectorUDT()),
    StructField("words_hashed", VectorUDT()),
    StructField("words_final", VectorUDT())
])

df = spark.read.json(sys.argv[2], schema=schema)

df = df.withColumn('label_pred', predict(vector_to_array('words_final')))
df = df.withColumn("id",f.monotonically_increasing_id())
print(df.select("id", "label_pred").show(10))
df.select("id", "label_pred").write.mode('overwrite').csv(sys.argv[4], header='false')
spark.stop()