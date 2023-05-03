#!/opt/conda/envs/dsenv/bin/python

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
import pyspark.sql.functions as f

from pyspark.ml.feature import StopWordsRemover, Tokenizer, VectorAssembler, HashingTF
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

conf = SparkConf()

spark = SparkSession.builder.config(conf=conf).appName("SparkML_hw").getOrCreate()

tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
stop_words = StopWordsRemover(inputCol="words",outputCol="words_without_stop", stopWords=StopWordsRemover.loadDefaultStopWords("english"))
hasher = HashingTF(numFeatures=110, binary=True, inputCol="words_without_stop", outputCol="words_hashed")
assembler = VectorAssembler(inputCols=['words_hashed'], outputCol="words_final")

pipeline = Pipeline(stages=[
    tokenizer,
    stop_words,
    hasher,
    assembler,
])

schema = StructType(fields=[
    StructField("label", FloatType()),
    StructField("vote", StringType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", TimestampType())])


df = spark.read.schema(schema).format("json").load(sys.argv[2]).fillna({"reviewText": "missingreview"})

mdl = pipeline.fit(df)
mdl = mdl.transform(df)
mdl.write.format("json").save(sys.argv[4])
spark.stop()
