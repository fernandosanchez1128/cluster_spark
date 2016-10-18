from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer,StopWordsRemover,CountVectorizer
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.feature import Word2Vec	
from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession, Row	
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



sc = SparkContext(appName="KMeans")	
#~ 
spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .config("spark.sql.inMemoryColumnarStorage.compressed", "true")\
    .getOrCreate()

from pyspark.ml.feature import Tokenizer, RegexTokenizer

from pyspark.ml.feature import DCT
from pyspark.ml.linalg import Vectors

df = spark.createDataFrame([
    (Vectors.dense([0.0, 1.0, -2.0, 3.0]),),
    (Vectors.dense([-1.0, 2.0, 4.5, -7.0]),),
    (Vectors.dense([14.0, -2.0, -5.0, 1.0]),)], ["features"])

dct = DCT(inverse=False, inputCol="features", outputCol="featuresDCT")

dctDf = dct.transform(df)

for dcts in dctDf.select("featuresDCT").take(3):
    print(dcts)
