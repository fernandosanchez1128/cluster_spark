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

#~ 
 #~ 
#~ sentenceData = spark.createDataFrame([
    #~ (1, "I wish Java could use case classes 0"),
    #~ (2, "I wish Java could use case classes"),
    #~ (3, "I wish Java could use case classes 2"),
    #~ (4, "Esto es un texto de prueba"),
    #~ (5, "Esto es un texto de prueba 2"),
    #~ (6, "I wish Java could use case classes 3"),
    #~ (7, "I wish Java could use case classes 4"),
    #~ (8, "Esto es un texto de prueba 3"),
    #~ (9, "Esto es un texto de prueba 4"),
    #~ (10, "I wish Java could use case classes 5"),
    #~ (11, "I wish Java could use case classes fndjf djcdkanjan djfnadkf djfakf akf"),
    #~ (12, "hablando en espaniol aunque hay diversas formas del espaniol solo digo eso esto es una prueba"),
    #~ (13, "I only wish than this test be ok on all"),
    #~ (14, "Spark es un framework de analisis distribuido en memoria, el cual fue desarrolado en la universidad de California < > . ' '.")
#~ ], ["label", "sentence"])


#sentenceData = spark.createDataFrame(data, ["label", "sentence"])
conf = {"es.resource" : "prueba", "es.nodes" : "127.0.0.1",
        "es.query" : "?q=name:alt.atheism name:misc.forsale" }


rdd = sc.newAPIHadoopRDD("org.elasticsearch.hadoop.mr.EsInputFormat",
"org.apache.hadoop.io.NullWritable", 
"org.elasticsearch.hadoop.mr.LinkedMapWritable", conf=conf).persist(StorageLevel.DISK_ONLY)


#print rdd.getStorageLevel()
rowData = rdd.map(convert_to_row)
sentenceData = spark.createDataFrame(rowData).persist(StorageLevel.DISK_ONLY)
print sentenceData.count()

print "tokenizer"
#tokenizer = RegexTokenizer(inputCol="sentence", outputCol="words_complete", pattern="[^a-zA-Z]")
tokenizer = RegexTokenizer(inputCol="sentence", outputCol="words_complete", pattern="[^a-zA-Z]")
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

#---------------------------------------------------------------------------------------
#usado para el clustering con kmeans
#~ 
#~ word2Vec = Word2Vec(inputCol="words", outputCol="features")
#~ print "vectorizer1"
#~ model = word2Vec.fit(dataCleaned)
#~ print "vectorizer2"
#~ rescaledData = model.transform(dataCleaned).persist(StorageLevel.DISK_ONLY)
#~ dataCleaned.unpersist()
#~ rescaledData.show()
#print rescaledData.getStorageLevel()
#---------------------------------------------------------------------------------------

#~ 
hashingTF = HashingTF(inputCol="words", outputCol="featuresCount", numFeatures=1000)#numero de palabras
featurizedData = hashingTF.transform(dataCleaned)

#---------------------------------------------------------------------------------------

#usado para el clustering de topics
#~ cv = CountVectorizer(inputCol="words", outputCol="featuresCount")#, minTF=4.0, minDF=20)
#~ model = cv.fit(dataCleaned)
#~ featurizedData = model.transform(dataCleaned)
#~ vocabulario =  model.vocabulary



idf = IDF(inputCol="featuresCount", outputCol="featuresInv")	
idfModel = idf.fit(featurizedData)
tfIdf = idfModel.transform(featurizedData)
#~ 
normalizer = Normalizer(inputCol="featuresInv", outputCol="features",p=2.0)
rescaledData = normalizer.transform(tfIdf)

#---------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------
#inicio de clustering usando kmeans

print "kmeans"
kmeans = KMeans(k=2)
model = kmeans.fit(rescaledData)
wssse = model.computeCost(rescaledData)
#~ 
centers = model.clusterCenters()
transformed = model.transform(rescaledData) 
#~ #transformed.select ("features").show()   


instancias = transformed.groupBy("prediction").count()
instancias.show()
#~ 
#~ for center in centers:
	#~ print center
instancias = transformed.groupBy("prediction","label").count().orderBy("count",ascending=False)
instancias.show(truncate=False)
#~ p = instancias.collect()
#~ print p
    

#----------------------------------------------------------------------
#~ #inicio de clustering usando latent
#~ lda = LDA(k=2)
#~ model = lda.fit(rescaledData)
#~ 
#~ ll = model.logLikelihood(rescaledData)
#~ lp = model.logPerplexity(rescaledData)
#~ 
#~ print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
#~ print("The upper bound bound on perplexity: " + str(lp))
#~ 
#~ # Describe topics.
#~ topics = model.describeTopics(10)
#~ print("The topics described by their top-weighted terms:")
#~ 
#~ udfWords=udf(words, ArrayType( StringType() ))
#~ topics_words = topics.withColumn("words", udfWords("termIndices"))
#~ topics_words.select ("words").show(truncate=False)

#----------------------------------------------------------------------
#bisecting kmeans
#bkm = BisectingKMeans().setK(2).setSeed(1)
#model = bkm.fit(dataset)
