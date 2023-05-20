from doxpy.misc.graph_builder import save_graphml, get_concept_description_dict
from doxpy.misc.jsonld_lib import HAS_LABEL_PREDICATE
from remove_duplicate_aspects import *


def get_aspects(knowledge_graph):
    aspects_list = list(get_concept_description_dict(graph=knowledge_graph, label_predicate=HAS_LABEL_PREDICATE,
                                                     valid_concept_filter_fn=lambda x: '{obj}' in x[1]).keys())
    return remove_duplicate_aspects(aspects_list)
