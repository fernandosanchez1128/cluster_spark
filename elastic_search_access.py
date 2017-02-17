from elasticsearch import Elasticsearch
from docs import conf


def index_document(project_id, document):
    es = Elasticsearch([
        {'host': conf.ELASTIC_SEARCH_HOST, 'port': conf.ELASTIC_SEARCH_PORT}
    ])
    res = es.index(index=project_id, doc_type='paper', body=document, id=document['id'])
    es.indices.refresh(index=project_id)
