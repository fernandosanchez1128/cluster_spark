from elasticsearch import Elasticsearch
es = Elasticsearch("127.0.0.1")

query = es.get(index="prueba", id='AVfy7Uvjs-K3wBQOT5Bb')
documento = query['_source']
print documento
#~ documento['cluster'] = 5
#~ response = es.update(index='prueba',doc_type = "pdf" ,id='AVfy7UrQs-K3wBQOT5BZ',body={"doc": documento })
#~ es.indices.refresh(index="prueba")

