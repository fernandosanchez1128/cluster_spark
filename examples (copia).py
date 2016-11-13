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
from pyspark.sql.types import ArrayType, StringType, NumericType, IntegralType
from collections import OrderedDict
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel

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

from pyspark.sql.functions import udf
def stemming(palabras):
	stemmer = Stemmer.Stemmer('english')
	return stemmer.stemWords(palabras)
	
def clean(palabras):
	cleaned = []
	for palabra in palabras:
		if len (palabra) > 2:
			cleaned.append(palabra)
	return cleaned


def words (indices):
	palabras = []
	for index in indices:
		palabras.append (vocabulario[index])
	return palabras
 	
def convert_to_row(d):
	text = d[1]['content']
	nombre = d[1]['name']
	return Row(label=nombre, sentence=text)

def group(distri):
    maximo = 0
    group=0
    i = 0
    for val in distri:
		if val > maximo:
			group  = i
			maximo = val
		i = i+1
    return str (group)
			
conf = {"es.resource" : "prueba", "es.nodes" : "127.0.0.1", "es.query" : "?q=name:alt.atheism name:misc.forsale" }


rdd = sc.newAPIHadoopRDD("org.elasticsearch.hadoop.mr.EsInputFormat",
"org.apache.hadoop.io.NullWritable", 
"org.elasticsearch.hadoop.mr.LinkedMapWritable", conf=conf).persist(StorageLevel.DISK_ONLY)


#print rdd.getStorageLevel()
rowData = rdd.map(convert_to_row)
sentenceData = spark.createDataFrame(rowData).persist(StorageLevel.DISK_ONLY)
print sentenceData.count()

tokenizer = RegexTokenizer(inputCol="sentence", outputCol="words_complete", pattern="[^a-zA-Z]")
wordsData = tokenizer.transform(sentenceData)

print "limpieza"
udfClean=udf(clean, ArrayType( StringType() ))
dataClean = wordsData.withColumn("cleaned", udfClean("words_complete")).persist(StorageLevel.DISK_ONLY)
wordsData.unpersist()

remover = StopWordsRemover(inputCol="cleaned", outputCol="words",stopWords =  StopWordsRemover.loadDefaultStopWords('english'))
dataCleaned = remover.transform(dataClean)
dataCleaned.select ("words").show(truncate = False)


#---------------------------------------------------------------------------------------
#usado para el clustering con kmeans
#~ word2Vec = Word2Vec(inputCol="words", outputCol="features")#vectorSize=100
#~ model = word2Vec.fit(dataCleaned)
#~ rescaledData = model.transform(dataCleaned)
#~ rescaledData.show()
#---------------------------------------------------------------------------------------


#~ hashingTF = HashingTF(inputCol="stemm", outputCol="features2" ) # numFeatures=100numero de palabras
#~ featurizedData = hashingTF.transform(dataCleaned)

#---------------------------------------------------------------------------------------

#usado para el clustering de topics
cv = CountVectorizer(inputCol="words", outputCol="features2", minDF=4.0)
model = cv.fit(dataCleaned)
featurizedData = model.transform(dataCleaned)
vocabulario =  model.vocabulary



idf = IDF(inputCol="features2", outputCol="features",minDocFreq=10)	
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

#~ normalizer = Normalizer(inputCol="featuresL", outputCol="features",p=2.0)
#~ rescaledData = normalizer.transform(rescaledData)

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
#~ transformed.show(truncate=False)
#~ 
#~ instancias = transformed.groupBy("prediction").count()
#~ instancias.show()
	#~ 
#~ instancias = transformed.groupBy("prediction","label").count().orderBy("count",ascending=False)
#~ instancias.show()

#~ 
#~ print("Within Set Sum of Squared Errors = " + str(wssse))
#a = instancias.withColumn('algo', lit (model.computeCost(transformed.filter(transformed.prediction == instancias['prediction']))))



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

#print model.topicsMatrix()

r=model.transform(rescaledData)

r.select('topicDistribution').show(truncate=False)
udfWords=udf(group, StringType())
topics_words = r.withColumn("grupo", udfWords("topicDistribution"))
instancias = topics_words.groupBy("grupo","label").count().orderBy("count",ascending=False)
instancias.show()

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


#----------------------------------------------------------------------	

#bisecting kmeans
