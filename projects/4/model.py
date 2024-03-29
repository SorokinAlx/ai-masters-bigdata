#!/opt/conda/envs/dsenv/bin/python

from pyspark.ml.feature import StopWordsRemover, Tokenizer, VectorAssembler, HashingTF
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
stop_words = StopWordsRemover(inputCol="words",outputCol="words_without_stop", stopWords=StopWordsRemover.loadDefaultStopWords("english"))
hasher = HashingTF(numFeatures=110, binary=True, inputCol="words_without_stop", outputCol="words_hashed")
assembler = VectorAssembler(inputCols=['words_hashed'], outputCol="words_final")
log_reg = GBTRegressor(featuresCol="words_final", labelCol="overall", maxIter=12)

pipeline = Pipeline(stages=[
    tokenizer,
    stop_words,
    hasher,
    assembler,
    log_reg
])