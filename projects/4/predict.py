#!/opt/conda/envs/dsenv/bin/python

import os
import sys

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

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

df = spark.read.schema(schema).format("json").load(sys.argv[2])
df = df.fillna({"verified" : True}, {"reviewText" : "missingreview"})
df = df.withColumn("verified", f.when(f.col("verified"), f.lit(1)).otherwise(0))

pred = model.transform(df)
pred.write.json(sys.argv[3])