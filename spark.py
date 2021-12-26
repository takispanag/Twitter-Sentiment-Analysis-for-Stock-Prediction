import pandas as pd
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.getOrCreate()

df = spark.read.csv("data/training_data/AAPL_data.csv", inferSchema=True, header=True)

df_va = VectorAssembler(inputCols = ["Sentiment", "Previous_Close"], outputCol = "features")
df = df_va.transform(df)
df = df.withColumnRenamed("Open", "label")
df = df.select(["features", "label"])

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(df)
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)