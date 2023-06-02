from doxpy.misc.graph_builder import get_concept_description_dict
from doxpy.misc.jsonld_lib import HAS_LABEL_PREDICATE
from remove_duplicate_aspects import *


def get_details_from(knowledge_graph) -> list:
    graph_clauses = list(filter(lambda x: '{obj}' in x[1], knowledge_graph))
    return graph_clauses


def get_aspects_from(knowledge_graph) -> list:
    aspects_list = list(get_concept_description_dict(graph=knowledge_graph, label_predicate=HAS_LABEL_PREDICATE,
                                                     valid_concept_filter_fn=lambda x: '{obj}' in x[1]).keys())
    return remove_duplicate_aspects(aspects_list)


question_template_list = [
    'What is {X}?',
    'How is {X}?',
    'Why {X}?',
]

"""'In what manner is {X}?',
    'What is the reason for {X}?',
    'What is the result of {X}?',
    'What is an example of {X}?',
    'After what is {X}?',
    'While what is {X}?',
    'In what case is {X}?',
    'Despite what is {X}?',
    'What is contrasted with {X}?',
    'Before what is {X}?',
    'What is similar to {X}?',
    'Instead of what is {X}?',
    'What is an alternative to {X}?',
    '{X}, unless what?',"""
