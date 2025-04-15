from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
# Khởi tạo SparkSession
spark=SparkSession.builder.appName("music").master("spark://spark-master:7077").config("spark.jars.packages","org.elasticsearch:elasticsearch-spark-30_2.12:7.17.4") \
                .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
                .config("spark.driver.extraClassPath","elasticsearch-hadoop-7.17.4.jar")\
                .config("spark.es.nodes","elasticsearch") \
                .config("spark.es.port","9200") \
                .config("spark.es.nodes.wan.only","true") \
                .getOrCreate()
spark.sparkContext.setLogLevsel("ERROR")
# Đọc file CSV từ HDFS
df = spark.read.csv("hdfs://namenode:9000/data/prediction_genre.csv", header=True, inferSchema=True)

elasticsearch_conf = {
            'es.nodes': "elasticsearch",
            'es.port': "9200",
            "es.nodes.wan.only": "true"
        }



# Hiển thị 5 dòng đầu tiên
df.show(5)
df_final = df.withColumn("prediction",col("prediction").cast("int"))
genres_labels=["Acoustic","Alt_music","Blues","Bollywood","Country","HipHop","Indie_Alt","Instrument","Metal","Pop","Rock"]
def map_genre(index):
    if index is not None and 0 <= int(index) < len(genres_labels):
        return genres_labels[index]
    else:
        return "unknown"
map_genre_udf=udf(map_genre,StringType())
df_with_genre = df_final.withColumn("genre", map_genre_udf(col("prediction")))
df_with_genre.groupBy("genre").count().orderBy("genre").show()

df_genre_number = df_with_genre.groupBy("release_year","release_month","genre").count().orderBy("release_year","release_month","genre")
df_genre_number.show()
df_genre_year = df_genre_number.withColumn("release_year",col("release_year").cast("string"))

df_to_elas = (df_genre_year,df_with_genre)
df_indexs= ("trend_music","full_music")
for dataframe, index in zip(df_to_elas,df_indexs):
    print("write index : ", index)
    dataframe.write \
        .format("org.elasticsearch.spark.sql") \
        .option("es.resource", f"{index}") \
        .options(**elasticsearch_conf) \
        .mode("overwrite") \
        .save()

spark.stop()