import textract
import logging
from docs import conf
import re
import requests
import json
import os
from elastic_search_access import index_document
import click
import sys
from tika import parser


def init():
    logging.basicConfig(filename=conf.LOG_FILE, level=logging.INFO,
                        format='%(levelname)s[%(asctime)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        filemode='a')


def get_text(file_path):
    """
    Converts PDF to plain text
    :param file_path: path to the file to convert
    :return: plain text
    """
    text = ''
    try:
        logging.info("Trying to get text from PDF file" + file_path)
        parsedPDF = parser.from_file(file_path)
        text = parsedPDF["content"]
    except:
        logging.info("Failed to get text from PDF file" + file_path)
        pass
    return text


def get_doi(text, file_path):
    """
    Gets the doi from a string
    :param text: plain text
    :return: first doi retrieved from the text
    """
    doi = ""
    parsedPDF = parser.from_file(file_path)
    metadata = parsedPDF["metadata"]
    try:
        doi = metadata["doi"]
    except:
        print "except doi"
        m = re.search('\\b(10[.][0-9]{4,}(?:[.][0-9]+)*/\\S+)', text)
        if m:
            doi = m.group(0)
        else:
            doi = None

    return doi


def get_metadata(file_path):
    """
    Get the plain text from PDF file and retrieves the doi and the metadata
    :param file_path: path to the file to process
    :return: dictionary representing document in elasticsearch model
    """
    text = get_text(file_path)
    doi = get_doi(text, file_path)
    # ~ print file_path.rpartition('/')[2].replace('.pdf', '')
    document = dict(doi=doi, id=file_path.rpartition('/')[2].replace('.pdf', ''), full_text=text)
    # ~ print doi
    # metadatos del documentos
    title = 'Generic'
    year = 'Generic'
    autor = 'Generic'
    subject = 'Generic'
    print file_path
    print doi

    if doi:
        # url used for getting title and year
        try:
            url = conf.CROSS_REF_URL
            resp = requests.get(url=url, params=dict(q=doi))
            data = json.loads(resp.text)

            # url used for getting author and subject
            url_json = conf.CROSS_REF_URL_JSON
            url = url_json + doi
            print url
            resp_json = requests.get(url=url)
            print resp_json
            data_2 = json.loads(resp_json.text)["message"]

            # mapping title
            try:
                title = data[0]['title']
            except:
                print "no title"

            # mapping year
            try:
                year = data[0]['year']
            except:
                print "no year"

            # mapping autor
            try:
                autor = data_2["author"][0]["family"] + " " + data_2["author"][0]["given"]
            except:
                print "no author"

            # mapping  subject
            try:
                subject = data_2["subject"][0]
            except:
                print "no subject"
        except:
            print "problems with doi"

    print "before"
    document['title'] = title
    document['year'] = year
    document['subject'] = subject
    document['author'] = autor

    return document


def process_file(file_path):
    """
    Gets the metadata for a document and index it to Elasticsearch
    :param file_path: path to file to process
    """
    document = get_metadata(file_path)
    print "documento"
    print document["title"]

    # ~ print document
    index_document('uvrs', document)


def process_path(path):
    """
    Iterate over a path tree to process each file
    :param path: path to the set of documents to process
    """
    for base, dirs, files in os.walk(path):
        for file_name in files:
            if file_name.startswith('.'):
                continue
            logging.info('Processing file: ' + base + file_name)
            process_file(base + "/" + file_name)


if __name__ == "__main__":
    try:
        process_path('/home/master/doc_pruebas/prueba')
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise
