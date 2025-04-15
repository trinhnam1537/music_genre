# -*- coding: utf-8 -*-

from pyspark.sql.functions import when, col,sqrt, cbrt,log, expr
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
import numpy as np
spark = SparkSession.builder.appName("TrainModel").master("local").getOrCreate()
df = spark.read.csv("hdfs://namenode:9000/user/root/train.csv", header=True, inferSchema=True)

df.show(5)
def transfrom_data(df):
    # chinh ten cot 
    new_columns = [col.replace(" ", "_").lower() for col in df.columns]
    df = df.toDF(*new_columns)
    df = df.withColumn(
    "duration_in_min/ms",
    when(col("duration_in_min/ms") < 30, col("duration_in_min/ms") * 60000)
    .otherwise(col("duration_in_min/ms"))
    )
    # xoa trung lap
    columns_to_check=[col for col in df.columns if col != "class"]
    df = df.dropDuplicates(subset=columns_to_check)
    # thay null = median 
    df = df.withColumn("popularity", col("popularity").cast("float"))
    popularity_median = df.approxQuantile("popularity", [0.5], 0.001)[0]
    instrumentalness_median = df.approxQuantile("instrumentalness", [0.5], 0.001)[0]
    df = df.fillna({
    "popularity": popularity_median,
    "instrumentalness": instrumentalness_median
    })
    df = df.fillna({"key": -1})


    return df

def data_fea(df):
    df = df.drop("energy")
    epsilon = 1e-6

     # Biến đổi các cột để giảm skew
    df =    df.withColumn("duration_in_ms_trans", sqrt(col("duration_in_min/ms")))
    df =    df.withColumn("loudness_trans", cbrt(col("loudness")))
    df =    df.withColumn("speechiness_trans", expr(f"1 / (speechiness + {epsilon})"))
    df =    df.withColumn("acousticness_trans", cbrt(col("acousticness")))
    df =    df.withColumn("instrumentalness_trans", log(col("instrumentalness") + epsilon))
    df =    df.withColumn("liveness_trans", log(col("liveness") + epsilon))
    df =    df.withColumn("tempo_trans", cbrt(col("tempo")))
    columns = ["artist_name", "track_name"]
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in columns]
    pipeline = Pipeline(stages=indexers)
    df= pipeline.fit(df).transform(df)
    # df=df.drop("artist_name","track_name")
    return df
df=transfrom_data(df)
df=data_fea(df)

print("done")
df.coalesce(1).write.mode("overwrite").option("header", "true").csv("hdfs://namenode:9000/data/data_clean_train.csv")
