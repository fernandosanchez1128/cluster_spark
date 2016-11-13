from statsmodels.iolib.tests.test_table_econpy import row0data

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer,StopWordsRemover,CountVectorizer
from pyspark.ml.clustering import BisectingKMeans,KMeans
from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import Word2Vec
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import lit
from pyspark.sql.types import ArrayType, StringType
from collections import OrderedDict
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel

# - *- coding: utf- 8 - *-
import codecs;
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import Normalizer
from pyspark.sql.functions import udf
import logging
from docs.config import *
import Stemmer

from elasticsearch import Elasticsearch

class Clustering:
    def __init__(self, algoritmo="kmeans", features ="tfidf", num_clusters=2):

        logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                            format='%(levelname)s[%(asctime)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                            filemode='a')
        logging.info("Archvo de log creado")

        try:
            logging.info("conectando Spark con ElasticSearch")
            sc = SparkContext(appName="KMeans")
            self.k = num_clusters
            self.algoritmo = algoritmo
            self.features = features
            spark = SparkSession \
                .builder \
                .appName("PythonSQL") \
                .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
                .getOrCreate()

            conf = {"es.resource": ELASTIC_SEARCH_INDEX, "es.nodes": ELASTIC_SEARCH_HOST}
                    #"es.query": "?q=name:rec.sport.hockey name:rec.autos"} #name:rec.sport.baseball"}
                    #"es.query": "?q=name:sci.space name:sci.crypt name:sci.med name:sci.electronics"}
                    #"es.query": "?q=name:comp.graphics name:rec.motorcycles name:sci.space name:talk.politics.misc"}
            rdd = sc.newAPIHadoopRDD("org.elasticsearch.hadoop.mr.EsInputFormat",
                                     "org.apache.hadoop.io.NullWritable",
                                     "org.elasticsearch.hadoop.mr.LinkedMapWritable", conf=conf).persist(
                StorageLevel.DISK_ONLY )

            #print rdd.collect()
            rowData = rdd.map(self.convert_to_row)

            self.sentenceData = spark.createDataFrame(rowData).persist(StorageLevel.DISK_ONLY )
            print self.sentenceData.count()
        except:

            logging.warning("Ha fallado la comunicacion entre ElasticSearch y Spark")

    def convert_to_row(self, d):
        logging.info("convirtiendo a DataFrame la data de ElasticSearch")
        text = d[1]['content']
        nombre = d[1]['name']
        id = d[0]
        return Row(label=nombre, sentence=text,id=id)

    def proceso_principal(self):
        if self.algoritmo == 'lda':
            self.lda_preprocessing()
            self.lda(self.vocabulario)
        else:
            self.preprocess()
            self.extract_features()
            self.clustering()
            #~ self.actualizar_indice()

    def preprocess(self):
        logging.info("Iniciando el preproceamiento del texto")

        def stemming(palabras):
            stemmer = Stemmer.Stemmer('english')
            return stemmer.stemWords(palabras)

        def clean(palabras):
            cleaned = []
            for palabra in palabras:
                if len(palabra) > 2:
                    cleaned.append(palabra)
            return cleaned

        try:
            logging.info("Tokenizacion del texto")
            tokenizer = RegexTokenizer(inputCol="sentence", outputCol="words_complete", pattern="[^a-zA-Z]")
            wordsData = tokenizer.transform(self.sentenceData).persist(StorageLevel.DISK_ONLY )
            self.sentenceData.unpersist()
            wordsData.select("id").show()
            print "limpieza"
            udfClean = udf(clean, ArrayType(StringType()))
            dataClean = wordsData.withColumn("cleaned", udfClean("words_complete")).persist(StorageLevel.DISK_ONLY )
            wordsData.unpersist()

            logging.info("stemming de los tokens")
            udfStemming = udf(stemming, ArrayType(StringType()))
            dataStemm = dataClean.withColumn("stemm", udfStemming("cleaned")).persist(StorageLevel.DISK_ONLY )
            wordsData.unpersist()

            logging.info("Eliminacion de las plabras vacias")
            remover = StopWordsRemover(inputCol="stemm", outputCol="words",
                                       stopWords=StopWordsRemover.loadDefaultStopWords('english'))
            self.dataCleaned = remover.transform(dataStemm).persist(StorageLevel.DISK_ONLY )
            self.dataCleaned.select("words").show()
            dataStemm.unpersist()
        except :
            logging.warning("Ha ocurrido un fallo en el preprocesamiento del texto")


    def extract_features(self):
        if self.features == "tfidf":
            self.tfidf()
        elif self.features == "vector":
            self.vector()
        else:
            self.counter()

    def tfidf(self):
        try:
            logging.info("Creando el TF-IDF")
            hashingTF = HashingTF(inputCol="words", outputCol="featuresCount", numFeatures=1000)  # numero de palabras
            featurizedData = hashingTF.transform(self.dataCleaned)
            idf = IDF(inputCol="featuresCount", outputCol="featuresInv")
            idfModel = idf.fit(featurizedData)
            tfIdf = idfModel.transform(featurizedData)

            normalizer = Normalizer(inputCol="featuresInv", outputCol="features", p=2.0)
            self.rescaledData = normalizer.transform(tfIdf)	
        except:
            logging.warning("Ha ocurrido un fallo creando el TF-IDF")

    def vector(self):
        try:
            logging.info("Creando el Vector")
            word2Vec = Word2Vec(inputCol="words", outputCol="featuresL")
            model = word2Vec.fit(self.dataCleaned)
            rescaledData = model.transform(self.dataCleaned).persist(StorageLevel.DISK_ONLY)
            self.dataCleaned.unpersist()
            normalizer = Normalizer(inputCol="featuresL", outputCol="features",p=2.0)
            self.rescaledData = normalizer.transform(rescaledData).persist(StorageLevel.DISK_ONLY)

        except:
            logging.warning("Ha ocurrido un fallo creando el vector")

    def counter(self):
        try:
            logging.info("Creando el Vector de conteo")
            cv = CountVectorizer(inputCol="words", outputCol="features2", minDF=20.0)
            model = cv.fit(self.dataCleaned)
            featurizedData = model.transform(self.dataCleaned).persist(StorageLevel.DISK_ONLY)
            vocabulario = model.vocabulary

            idf = IDF(inputCol="features2", outputCol="featuresL", minDocFreq=20.0)
            idfModel = idf.fit(featurizedData)
            rescaledData = idfModel.transform(featurizedData).persist(StorageLevel.DISK_ONLY)
            normalizer = Normalizer(inputCol="featuresL", outputCol="features",p=2.0)
            self.rescaledData = normalizer.transform(rescaledData).persist(StorageLevel.DISK_ONLY)
        except:
            logging.warning("Ha ocurrido un fallo creando el vector de conteo")




    def clustering(self):
        if self.algoritmo == 'kmeans':
            self.kmeans()
        else:
            self.bisectingKmeans()

    def bisectingKmeans (self):
        try:
            logging.info("Inicializando el algoritmo BisectingKmeans")
            bkm = BisectingKMeans().setK(self.k).setSeed(1)
            model = bkm.fit(self.rescaledData)
            self.transformed = model.transform(self.rescaledData)
            instancias = self.transformed.groupBy("prediction").count()
            instancias.show()
            instancias = self.transformed.groupBy("prediction", "label").count().orderBy("count", ascending=False)
            instancias.show(truncate=False)
            #self.transformed.show()
        except:
            logging.warning("algoritmo BisectingKmeans ha fallado")

    def kmeans(self):
        try:
            kmeans = KMeans(k=self.k, initMode="random")
            model = kmeans.fit(self.rescaledData)
            wssse = model.computeCost(self.rescaledData)
            # ~
            centers = model.clusterCenters()
            self.transformed = model.transform(self.rescaledData)
            # ~ #transformed.select ("features").show()


            instancias = self.transformed.groupBy("prediction").count()
            instancias.show()
            # ~
            # ~ for center in centers:
            # ~ print center
            instancias = self.transformed.groupBy("prediction", "label").count().orderBy("count", ascending=False)
            instancias.show(truncate=False)

        except:
            logging.warning("algoritmo Kmeans ha fallado")

    def lda_preprocessing (self):

        def clean(palabras):
            cleaned = []
            for palabra in palabras:
                if len(palabra) > 2:
                    cleaned.append(palabra)
            return cleaned

        try:
            logging.info("Tokenizacion del texto")
            tokenizer = RegexTokenizer(inputCol="sentence", outputCol="words_complete", pattern="[^a-zA-Z]")
            wordsData = tokenizer.transform(self.sentenceData).persist(StorageLevel.DISK_ONLY )
            self.sentenceData.unpersist()
            wordsData.select("id").show()
            print "limpieza"

            udfClean = udf(clean, ArrayType(StringType()))
            dataClean = wordsData.withColumn("cleaned", udfClean("words_complete")).persist(StorageLevel.DISK_ONLY )
            wordsData.unpersist()

            logging.info("Eliminacion de las plabras vacias")
            remover = StopWordsRemover(inputCol="cleaned", outputCol="words",
                                       stopWords=StopWordsRemover.loadDefaultStopWords('english'))
            self.dataCleaned = remover.transform(dataClean).persist(StorageLevel.DISK_ONLY )
            self.dataCleaned.select("words").show()
            dataClean.unpersist()
        except :
            logging.warning("Ha ocurrido un fallo en el preprocesamiento del texto")
        try:
            logging.info("Creando el Vector de conteo")
            cv = CountVectorizer(inputCol="words", outputCol="features2", minDF=20.0)
            model = cv.fit(self.dataCleaned)
            featurizedData = model.transform(self.dataCleaned).persist(StorageLevel.DISK_ONLY)
            self.vocabulario = model.vocabulary

            idf = IDF(inputCol="features2", outputCol="featuresL", minDocFreq=20.0)
            idfModel = idf.fit(featurizedData)
            rescaledData = idfModel.transform(featurizedData).persist(StorageLevel.DISK_ONLY)
            normalizer = Normalizer(inputCol="featuresL", outputCol="features",p=2.0)
            self.rescaledData = normalizer.transform(rescaledData).persist(StorageLevel.DISK_ONLY)
        except:
            logging.warning("Ha ocurrido un fallo creando el vector de conteo")

    def lda (self,vocabulario):
        def group(distri):
            maximo = 0
            group = 0
            i = 0
            for val in distri:
                if val > maximo:
                    group = i
                    maximo = val
                i = i + 1
            return str(group)

        def words(indices):
            palabras = []
            for index in indices:
                palabras.append(vocabulario[index])
            return palabras

        try :
            lda = LDA(k=self.k)
            model = lda.fit(self.rescaledData)
            # Describe topics.
            r = model.transform(self.rescaledData)
            topics = model.describeTopics(10)
            print("The topics described by their top-weighted terms:")
            udfWords = udf(words, ArrayType(StringType()))
            topics_words = topics.withColumn("words", udfWords("termIndices"))
            topics_words.select("words").show(truncate=False)


            #r.select('topicDistribution').show(truncate=False)
            udfGroup= udf(group, StringType())
            topics_words = r.withColumn("grupo", udfGroup("topicDistribution"))
            instancias = topics_words.groupBy("grupo").count()
            instancias.show()
            instancias = topics_words.groupBy("grupo", "label").count().orderBy("count", ascending=False)
            instancias.show()
        except:
            print "error realizando lda clustering"
            logging.warning("error realizando lda clustering")


         
    def actualizar_indice(self):

        def grupo(id, grupo):
            es = Elasticsearch("127.0.0.1")
            query = es.get(index="prueba", id=id)
            documento = query['_source']
            print documento
            documento['cluster'] = grupo
            response = es.update(index='prueba', doc_type="pdf", id=id, body={"doc": documento})
            es.indices.refresh(index="prueba")
            return "hola"

        udfWords = udf(grupo, StringType())
        topics_words = self.transformed.withColumn("grupo", udfWords("id", "prediction"))
        topics_words.show()

        return "hola"


clustering = Clustering("kmeans", "counter", 4)     
#clustering = Clustering("bkm", "counter", 4)
#clustering = Clustering("lda", "counter", 4)
clustering.proceso_principal()
