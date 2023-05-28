import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from doxpy.models.knowledge_extraction.knowledge_graph_extractor import KnowledgeGraphExtractor

import confuse
from get_aspects import *

PHI = [
    'IF (I is similar to "as a developer id like to add support for dynamic classpath for modules so we can have the flexibility to load the right dependencies either based on module options 0 or via other properties such as including the dependencies from a specific location 1 0 code libjarlibdistrojar code 1 code xdhomelibhadoopdistrojar code example code http hdfs --distrophd22 http mycustommodule --classpathmyfunkydir http jpa --providereclipse jpa config libsomething-that-is-commonjar eclipseeclipse-link-32jar hibernatehibernate-core-50jar moduleclasspath libjarlibproviderjar code)"',
    "THEN '8'"
]

if __name__ == '__main__':
    config = confuse.Configuration('XAI', __name__)
    knowledge_builder_options = config["knowledge_builder_options"].get(dict)

    print('Building Graph..')

    explainable_information_graph = KnowledgeGraphExtractor(knowledge_builder_options).set_content_list(PHI,
                                                                                                         remove_stopwords=False,
                                                                                                         remove_numbers=False,
                                                                                                         avoid_jumps=True).build()

    # save_graphml(explainable_information_graph, 'knowledge_graph')
    print('Graph size:', len(explainable_information_graph))
    print("Graph's Clauses:", len(list(filter(lambda x: '{obj}' in x[1], explainable_information_graph))))
    print(get_aspects(explainable_information_graph))