#!/opt/conda/envs/dsenv/bin/python

import os
import sys

from pyspark import SparkConf
from pyspark.sql import SparkSession

conf = SparkConf()
spark = SparkSession.builder.config(conf=conf).appName("SparkML_hw").getOrCreate()
spark.sparkContext.setLogLevel('INFO')

from pyspark.ml import Pipeline, PipelineModel

model = PipelineModel.load(sys.argv[1])

schema = StructType(fields=[
    StructField("overall", FloatType()),
    StructField("vote", StringType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", TimestampType())])

df = spark.read.schema(schema).format("json").load(sys.argv[2]).fillna( {"reviewText": "missingreview"})

pred = model.transform(df)
pred.select("id","prediction").write.save(sys.argv[3])

spark.stop()