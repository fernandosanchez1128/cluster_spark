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

sentenceDataFrame = spark.createDataFrame([
    (0, "Spark es un framework de analisis distribuido en memoria, el cual fue desarrolado en la universidad de California < > . ' '."),
    (1, "I wish Java could use case classes"),
    (2, "Logistic,regression,models,are,neat")
], ["label", "sentence"])
#~ tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
#~ wordsDataFrame = tokenizer.transform(sentenceDataFrame)
regexTokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="//W")
wordsDataFrame = regexTokenizer.transform(sentenceDataFrame)
wordsDataFrame.select("words").show(truncate=False)
