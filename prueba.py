from pyspark.sql import SparkSession, Row
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import lit
from pyspark.ml.feature import Normalizer
from pyspark.sql.types import ArrayType, StringType
from collections import OrderedDict
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
# - *- coding: utf- 8 - *-
import codecs;


sc = SparkContext(appName="KMeans")

spark = SparkSession \
                .builder \
                .appName("PythonSQL") \
                .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
                .getOrCreate()
                
        

sentenceData = spark.createDataFrame([
    (1, "I wish Java could use case classes 0"),
    (2, "I wish Java could use case classes"),
    (3, "I wish Java could use case classes 2"),
    (4, "Esto es un texto de prueba"),
    (5, "Esto es un texto de prueba 2"),
    (6, "I wish Java could use case classes 3"),
    (7, "I wish Java could use case classes 4"),
    (8, "Esto es un texto de prueba 3"),
    (9, "Esto es un texto de prueba 4"),
    (10, "I wish Java could use case classes 5"),
    (11, "I wish Java could use case classes fndjf djcdkanjan djfnadkf djfakf akf"),
    (12, "hablando en espaniol aunque hay diversas formas del espaniol solo digo eso esto es una prueba"),
    (13, "I only wish than this test be ok on all"),
    (14, "Spark es un framework de analisis distribuido en memoria, el cual fue desarrolado en la universidad de California < > . ' '.")
], ["label", "sentence"])

        
es_conf = {"es.resource": 'prueba2', "es.nodes": '127.0.0.1'}                
sentenceRdd = sentenceData.rdd
sentenceRdd.saveAsNewAPIHadoopFile(
    path='-',
    outputFormatClass="org.elasticsearch.hadoop.mr.EsOutputFormat",
    keyClass="org.apache.hadoop.io.NullWritable",
    valueClass="org.elasticsearch.hadoop.mr.LinkedMapWritable",
    conf=es_conf)
