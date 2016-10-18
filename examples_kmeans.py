from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer,StopWordsRemover,CountVectorizer
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.feature import Word2Vec	
from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession, Row	
#from texto import texto
from pyspark.ml.clustering import LDA
from pandas.tools.plotting import parallel_coordinates
from pyspark.sql.functions import lit
from pyspark.ml.feature import Normalizer
from pyspark.sql.functions import exp
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import random 
import numpy 
from elasticsearch import Elasticsearch
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.feature import DCT
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import DCT
from pyspark.sql.types import ArrayType, StringType
from collections import OrderedDict
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
from pyspark.ml.feature import MaxAbsScaler
from pyspark.ml.feature import MinMaxScaler
import Stemmer
# -*- coding: utf-8 -*-
import codecs;
sc = SparkContext(appName="KMeans")	
#~ 
spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .config("spark.sql.inMemoryColumnarStorage.compressed", "true")\
    .getOrCreate()
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
es = Elasticsearch("127.0.0.1")
#




from pyspark.sql.functions import udf
def stemming(palabras):
	stemmer = Stemmer.Stemmer('english')
	return stemmer.stemWords(palabras)

def words (indices):
	palabras = []
	for index in indices:
		palabras.append (vocabulario[index])
	return palabras
 	
def convert_to_row(d):
    text =  d[1]['content'] 
    nombre = d[1] ['name']	
    return Row(label = nombre, sentence = text)

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])
    
def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex
    
conf = {"es.resource" : "prueba", "es.nodes" : "127.0.0.1", "es.query" : "?q=name:alt.atheism name:misc.forsale" }


rdd = sc.newAPIHadoopRDD("org.elasticsearch.hadoop.mr.EsInputFormat",
"org.apache.hadoop.io.NullWritable", 
"org.elasticsearch.hadoop.mr.LinkedMapWritable", conf=conf).persist(StorageLevel.DISK_ONLY)


#print rdd.getStorageLevel()
rowData = rdd.map(convert_to_row)
sentenceData = spark.createDataFrame(rowData).persist(StorageLevel.DISK_ONLY)
print sentenceData.count()

print "tokenizer"
tokenizer = RegexTokenizer(inputCol="sentence", outputCol="words_complete", pattern="\\W")
wordsData = tokenizer.transform(sentenceData).persist(StorageLevel.DISK_ONLY)
sentenceData.unpersist()

print "stemming"
udfStemming=udf(stemming, ArrayType( StringType() ))
dataStemm = wordsData.withColumn("stemm", udfStemming("words_complete")).persist(StorageLevel.DISK_ONLY)
wordsData.unpersist()

print "words remover"
remover = StopWordsRemover(inputCol="stemm", outputCol="words",stopWords =  StopWordsRemover.loadDefaultStopWords('english'))
dataCleaned = remover.transform(dataStemm).persist(StorageLevel.DISK_ONLY)
dataCleaned.select ("words").show()
dataStemm.unpersist()


#usado para el clustering de topics
cv = CountVectorizer(inputCol="words", outputCol="featuresCount", minTF=1.0, minDF=10.0)
model = cv.fit(dataCleaned)
featurizedData = model.transform(dataCleaned)
vocabulario =  model.vocabulary


idf = IDF(inputCol="featuresCount", outputCol="featuresInv",minDocFreq=10.0)	
idfModel = idf.fit(featurizedData)
tfIdf = idfModel.transform(featurizedData)
#~ rescaledData = idfModel.transform(featurizedData)

