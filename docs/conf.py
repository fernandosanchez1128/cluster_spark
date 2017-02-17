import os

LOG_FILE = os.environ.get('LOG_FILE') or '/home/master/cluster_spark/docs/logs.log'
ELASTIC_SEARCH_HOST = '192.168.1.64'
ELASTIC_SEARCH_INDEX = 'prueba'
ELASTIC_SEARCH_PORT = '9200'
CROSS_REF_URL = 'http://search.crossref.org/dois'
CROSS_REF_URL_JSON = 'http://api.crossref.org/v1/works/http://dx.doi.org/'
