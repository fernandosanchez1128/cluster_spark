from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer,StopWordsRemover,CountVectorizer
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.feature import Word2Vec	
from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession, Row	
from texto import texto
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

#~ page = es.search(
  #~ index = 'prueba',
  #~ scroll = '2m',
  #~ search_type = 'scan',
  #~ size = 1000,
  #~ body = {
#~ })

from pyspark.sql.functions import udf
def stemming(palabras):
	stemmer = Stemmer.Stemmer('english')
	return stemmer.stemWords(palabras)

def words (indices):
	palabras = []
	for index in indices:
		palabras.append (vocabulario[index])
	return palabras
 	
def search2():
    query = es.search(index="prueba",scroll='2m',size =2000,
    body={"query": {"match_all": {}}})
    sid = query['_scroll_id']
    scroll_size = query['hits']['total']
    #response = [ (str (hit["_source"] ["name"]), str (hit["_source"] ["content"])) for hit in query['hits']['hits']]
    print "inicio"
    response = []
    for hit in query['hits']['hits']:
			val = hit["_source"] ["content"]
			response.append((str (hit["_source"] ["name"]), val ))
    while (scroll_size > 0):
	    query = es.scroll(scroll_id = sid, scroll = '2m')
	    # Update the scroll ID
	    sid = query['_scroll_id']
	    # Get the number of results that we returned in the last scroll
	    for hit in query['hits']['hits']:
			val = hit["_source"] ["content"]
			response.append((str (hit["_source"] ["name"]), val))
	    scroll_size = len(query['hits']['hits'])
	    #print "scroll size: " + str(scroll_size)
    print "fin"
    return response

def convert_to_row(d):
    text =  d[1]['content']
    return Row(label = '2', sentence = text)

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
    #~ (13, "I only wish than this test be ok on all")
#~ ], ["label", "sentence"])

#~ data = search2()
#~ print len(data)
#~ sentenceData = spark.createDataFrame(data, ["label", "sentence"])

conf = {"es.resource" : "prueba", "es.nodes" : "127.0.0.1"}


rdd = sc.newAPIHadoopRDD("org.elasticsearch.hadoop.mr.EsInputFormat",
"org.apache.hadoop.io.NullWritable", 
"org.elasticsearch.hadoop.mr.LinkedMapWritable", conf=conf)

#sentenceData = sc.parallelize([{"arg1": "", "arg2": ""},{"arg1": "", "arg2": ""},{"arg1": "", "arg2": ""}]).toDF()
rowData = rdd.map(convert_to_row)
sentenceData = spark.createDataFrame(rowData)
sentenceData.show()


tokenizer = RegexTokenizer(inputCol="sentence", outputCol="words_complete", pattern="\\W")
wordsData = tokenizer.transform(sentenceData)
remover = StopWordsRemover(inputCol="words_complete", outputCol="words")
dataCleaned = remover.transform(wordsData)
udfStemming=udf(stemming, ArrayType( StringType() ))
dataCleaned = dataCleaned.withColumn("stemm", udfStemming("words"))



#---------------------------------------------------------------------------------------
#usado para el clustering con kmeans
word2Vec = Word2Vec(inputCol="stemm", outputCol="features")#vectorSize=100
model = word2Vec.fit(dataCleaned)
rescaledData = model.transform(dataCleaned)
rescaledData.show()
#---------------------------------------------------------------------------------------


#rescaledData.select('stemm','features').show(truncate=False)

#~ 

#~ hashingTF = HashingTF(inputCol="stemm", outputCol="features2", numFeatures=100) #numero de palabras
#~ featurizedData = hashingTF.transform(dataCleaned)
#~ hashingTF.vocabulary
# alternatively, CountVectorizer can also be used to get term frequency vectors

#---------------------------------------------------------------------------------------
#usado para el clustering de topics
cv = CountVectorizer(inputCol="stemm", outputCol="features2")
model = cv.fit(dataCleaned)
featurizedData = model.transform(dataCleaned)
vocabulario =  model.vocabulary

idf = IDF(inputCol="features2", outputCol="features")	
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
#---------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------
#inicio de clustering usando kmeans
#~ r = rescaledData.cache()
#~ kmeans = KMeans(k=2, seed=10,initMode = "k-means||")
#~ model = kmeans.fit(r)
#~ wssse = model.computeCost(r)

#~ centers = model.clusterCenters()
#~ transformed = model.transform(rescaledData) 
#~ transformed.select ("features").show(truncate=True)   
#~ transformed.show(truncate=True)
#~ 
#~ instancias = transformed.groupBy("prediction").count()
#~ instancias.show()
	#~ 
#~ instancias = transformed.groupBy("prediction","label").count().orderBy("count",ascending=False)
#~ instancias.show()

#~ 
#~ print("Within Set Sum of Squared Errors = " + str(wssse))
#a = instancias.withColumn('algo', lit (model.computeCost(transformed.filter(transformed.prediction == instancias['prediction']))))


#j1 =  instancias.toJSON().collect()
#print j1

#~ grupos = transformed.select('prediction','features')
#~ j2 = grupos.toJSON().first()
#~ 
#~ solve = {"1":j1,"2":j2}
#~ print solve["1"]


#grupos = transformed.select('prediction','features').collect() 
#arreglo = [group['features'].toArray() for group in groups]
#labels =  [group['prediction'] for group in groups]


#df = pd.DataFrame(arreglo)
#df['grupo']= pd.Series(labels, index=df.index)
#print labels
#parallel_coordinates(df,'grupo')    
#plt.show()
    

#----------------------------------------------------------------------
#~ #inicio de clustering usando latent
lda = LDA(k=2)
model = lda.fit(rescaledData)

ll = model.logLikelihood(rescaledData)
lp = model.logPerplexity(rescaledData)

print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound bound on perplexity: " + str(lp))



# Describe topics.
topics = model.describeTopics(10)
print("The topics described by their top-weighted terms:")

udfWords=udf(words, ArrayType( StringType() ))
topics_words = topics.withColumn("words", udfWords("termIndices"))
topics_words.show(truncate=False)


# Shows the result
transformed = model.transform(rescaledData)
transformed	.show()

#----------------------------------------------------------------------	

#~ #gaussian clustering
#~ gmm = GaussianMixture().setK(2)
#~ model = gmm.fit(rescaledData)
#~ 
#~ print("Gaussians: ")
#~ transformed = model.transform(rescaledData)
#~ transformed.show(truncate=True)
#~ instancias = transformed.groupBy("prediction").count()
#~ instancias.show()
#~ instancias = transformed.groupBy("prediction","label").count().orderBy("count",ascending=False)
#~ instancias.show()
#~ 

#----------------------------------------------------------------------	

#bisecting kmeans
#~ gmm = GaussianMixture().setK(2).setSeed(1)
#~ model = gmm.fit(rescaledData)
#~ transformed = model.transform(rescaledData)
#~ transformed.show(truncate=True)
#~ instancias = transformed.groupBy("prediction").count()	
#~ instancias.show()
#~ 
#~ instancias = transformed.groupBy("prediction","label").count().orderBy("count",ascending=False)
#~ instancias.show()
#~ wssse = model.computeCost(rescaledData)
#~ print("Within Set Sum of Squared Errors = " + str(wssse))
