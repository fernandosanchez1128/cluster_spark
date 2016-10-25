#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function
from operator import add
from pyspark import SparkContext
from os import listdir
# $example on$
from pyspark.mllib.feature import Word2Vec
from elasticsearch import Elasticsearch

# $example off$
es = Elasticsearch("127.0.0.1")

if __name__ == "__main__":

    sc = SparkContext(appName="Word2VecExample")  # SparkContext
    # clases = ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",  "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale","rec.autos","rec.motorcycles","rec.sport.baseball", "rec.sport.hockey","sci.crypt","sci.electronics","sci.med","sci.space", "soc.religion.christian"]
    #clases = ["talk.politics.guns", "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc"]
    #clases = ["alt.atheism", "misc.forsale"]
    clases = ["comp.graphics","misc.forsale","rec.motorcycles","alt.atheism",]
    name = "prueba"
    es.indices.delete(index=name, ignore=[400, 404])

    # $example on$
    i=0
    for clase in clases:
        directorio = "textos/" + clase + "/"
        documentos = [doc for doc in listdir(directorio)]
        for doc in documentos:
            inp = sc.textFile(directorio + doc)
            texto = ""
            texto = inp.map(lambda line: line).reduce(lambda x, y: x + y)
            print(clase)
            i = i+1
            document = {
                'id': i,
                'name': clase,  # nombre del archivo o documento
                'author': "Fernando",  # author de este archivo o documento
                'content': texto,  # el contenido de este archivo o documento
                'owner': "propietario"  # propietario, usuario que crea el indice
            }
            response = es.index(index="prueba", doc_type="pdf", body=document)

    es.indices.refresh(index="prueba")

    # synonyms = model.findSynonyms('china', 5)

    # for word, cosine_distance in synonyms:
    #    print("{}: {}".format(word, cosine_distance))
    # $example off$

    sc.stop()
