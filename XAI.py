from fuzzy_expert.inference import DecompositionalInference

from doxpy.models.knowledge_extraction.knowledge_graph_extractor import KnowledgeGraphExtractor
from doxpy.models.estimation.dox_estimator import DoXEstimator
from doxpy.models.knowledge_extraction.knowledge_graph_manager import KnowledgeGraphManager
from doxpy.models.reasoning.answer_retriever import AnswerRetriever
from doxpy.misc.doc_reader import load_or_create_cache, DocParser
from doxpy.misc.graph_builder import get_betweenness_centrality, save_graphml, get_concept_set, \
    get_concept_description_dict
from doxpy.misc.jsonld_lib import *
from doxpy.misc.utils import *
import math

import json
import os
import sys
import logging

from matplotlib import pyplot as plt
from fuzzy_expert.variable import FuzzyVariable

from get_aspects import get_aspects

logger = logging.getLogger('doxpy')
logger.setLevel(logging.INFO)

model_type = "fb"
answer_pertinence_threshold = 0.5
synonymity_threshold = 0.6
cache_path = "./cache"

answer_pertinence_threshold = float(answer_pertinence_threshold)
synonymity_threshold = float(synonymity_threshold)
if not os.path.exists(cache_path): os.mkdir(cache_path)


AVOID_JUMPS = True
# keep_the_n_most_similar_concepts = 2
# query_concept_similarity_threshold = 0.75,

ARCHETYPE_FITNESS_OPTIONS = {
    'one_answer_per_sentence': False,
    'answer_pertinence_threshold': answer_pertinence_threshold,
    'answer_to_question_max_similarity_threshold': None,
    'answer_to_answer_max_similarity_threshold': 0.85,
}

KG_MANAGER_OPTIONS = {
    'spacy_model': 'en_core_web_trf',
    'n_threads': 1,
    'use_cuda': True,
    'with_cache': False,
    'with_tqdm': False,

    # 'min_triplet_len': 0,
    # 'max_triplet_len': float('inf'),
    # 'min_sentence_len': 0,
    # 'max_sentence_len': float('inf'),
    # 'min_paragraph_len': 0,
    # 'max_paragraph_len': 0, # do not use paragraphs for computing DoX
}

GRAPH_EXTRACTION_OPTIONS = {
    'add_verbs': False,
    'add_predicates_label': False,
    'add_subclasses': True,
    'use_wordnet': False,
}

GRAPH_CLEANING_OPTIONS = {
    'remove_stopwords': False,
    'remove_numbers': False,
    'avoid_jumps': AVOID_JUMPS,
    'parallel_extraction': False,
}

GRAPH_BUILDER_OPTIONS = {
    'spacy_model': 'en_core_web_trf',
    'n_threads': 1,
    'use_cuda': True,

    'with_cache': False,
    'with_tqdm': False,

    'max_syntagma_length': None,
    'add_source': True,
    'add_label': True,
    'lemmatize_label': False,

    # 'default_similarity_threshold': 0.75,
    'default_similarity_threshold': 0,
    'tf_model': {
        'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5',  # Transformer
    },
}


CONCEPT_CLASSIFIER_OPTIONS = {
    'spacy_model': 'en_core_web_trf',
    'n_threads': 1,
    'use_cuda': True,

    'default_batch_size': 20,
    'with_tqdm': False,
    'with_cache': True,

    'tf_model': {
        'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5',
    },
    'default_similarity_threshold': synonymity_threshold,
    'default_tfidf_importance': 0,
}


SENTENCE_CLASSIFIER_OPTIONS = {
    'spacy_model': 'en_core_web_trf',
    'n_threads': 1,
    'use_cuda': True,

    'with_tqdm': False,
    'with_cache': False,

    'default_tfidf_importance': 0,
}

if model_type == 'tf':
    SENTENCE_CLASSIFIER_OPTIONS['tf_model'] = {
        'url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3',
        'with_cache': True,
    }
else:
    SENTENCE_CLASSIFIER_OPTIONS['sbert_model'] = {
        'url': 'multi-qa-MiniLM-L6-cos-v1',  # model for paraphrase identification
        'with_cache': True,
    }


def get_explicandum_graph(explicandum_path):

    explicandum_path_split = list(explicandum_path.split('/'))
    explicandum_name = explicandum_path_split[len(explicandum_path_split) - 1]

    explicandum_graph_cache = os.path.join(cache_path,
                                           f"cache_explicandum_graph_{explicandum_name}.pkl")
    print(f'Building Explicandum Graph for {explicandum_name}...')

    explicandum_graph = load_or_create_cache(
        explicandum_graph_cache,
        lambda: KnowledgeGraphExtractor(GRAPH_BUILDER_OPTIONS).set_documents_path(explicandum_path,
                                                                                  remove_stopwords=True,
                                                                                  remove_numbers=True,
                                                                                  avoid_jumps=True).build(
            **GRAPH_EXTRACTION_OPTIONS)
    )
    save_graphml(explicandum_graph, os.path.join(cache_path, 'explicandum_graph'))

    return explicandum_graph


def get_explainable_information_graph(explainable_information_path):
    explainable_information_graph_cache = os.path.join(cache_path,
                                                       f"cache_explainable_information_graph.pkl")

    explainable_information_graph = load_or_create_cache(
        explainable_information_graph_cache,
        lambda: KnowledgeGraphExtractor(GRAPH_BUILDER_OPTIONS).set_documents_path(explainable_information_path,
                                                                                  **GRAPH_CLEANING_OPTIONS).build(
            **GRAPH_EXTRACTION_OPTIONS)
    )
    save_graphml(explainable_information_graph, os.path.join(cache_path, 'explainable_information_graph'))
    return explainable_information_graph


def get_keys_with_highest_value(dictionary):
    # Find the maximum value in the dictionary
    max_value = max(dictionary.values())

    # Create a list to store the keys with the highest value
    highest_value_keys = []

    # Iterate over the dictionary and check if the value is equal to the maximum value
    for key, value in dictionary.items():
        if value == max_value:
            highest_value_keys.append(key)

    return highest_value_keys


if __name__ == "__main__":

    outsider_explicandum_graph = get_explicandum_graph("./data/explicanda/outsider")
    outsider_aspects_amount = len(get_aspects(outsider_explicandum_graph))

    practitioner_explicandum_graph = get_explicandum_graph("./data/explicanda/practitioner")
    practitioner_aspects_amount = len(get_aspects(practitioner_explicandum_graph))

    experts_explicandum_graph = get_explicandum_graph("./data/explicanda/expert")
    experts_aspects_amount = len(get_aspects(experts_explicandum_graph))

    aspects_variable = FuzzyVariable(
            universe_range=(0, 20),
            terms = {
                "SPARSE": ('zmf', outsider_aspects_amount, outsider_aspects_amount + 2),
                "PERTINENT": ("gbellmf", practitioner_aspects_amount, practitioner_aspects_amount / 2, 4),
                "EXTENSIVE": ("smf", experts_aspects_amount, experts_aspects_amount + 2)
            }
        )

    acceptability_variable = FuzzyVariable(
            universe_range=(0, 10),
            terms={
                "TOTALLY_UNACCEPTABLE": ('trapmf', 0, 0, 1, 1.5),
                "UNACCEPTABLE": ('trapmf', 1, 1.5, 2.5, 3),
                "SLIGHTLY_UNACCEPTABLE": ('trapmf', 2.5, 3, 4, 4.5),
                "NEUTRAL": ('trapmf', 4, 4.5, 5.5, 6),
                "SLIGHTLY_ACCEPTABLE": ('trapmf', 5.5, 6, 7, 7.5),
                "ACCEPTABLE": ('trapmf', 7, 7.5, 8.5, 9),
                "TOTALLY_ACCEPTABLE": ('trapmf', 8.5, 9, 10, 10)
            })

    fuzzy_variables = {
        "aspects": aspects_variable,
        "outsider_acceptability": acceptability_variable,
        "practitioner_acceptability": acceptability_variable,
        "expert_acceptability": acceptability_variable,
    }

    fuzzy_variables['aspects'].plot()
    plt.savefig('aspects.png')
    plt.clf()

    print(f"outsider_aspects_amount: {outsider_aspects_amount}")
    print(f"practitioner_aspects_amount: {practitioner_aspects_amount}")
    print(f"experts_aspects_amount: {experts_aspects_amount}")

    from fuzzy_expert.rule import FuzzyRule

    outsider_rules = [
        FuzzyRule(
            premise=[
                ("aspects", "SPARSE"),
            ],
            consequence=[("outsider_acceptability", "TOTALLY_ACCEPTABLE")],
        ),
        FuzzyRule(
            premise=[
                ("aspects", "PERTINENT"),
            ],
            consequence=[("outsider_acceptability", "SLIGHTLY_UNACCEPTABLE")],
        ),
        FuzzyRule(
            premise=[
                ("aspects", "EXTENSIVE"),
            ],
            consequence=[("outsider_acceptability", "TOTALLY_UNACCEPTABLE")],
        ),
    ]

    practitioner_rules = [
        FuzzyRule(
            premise=[
                ("aspects", "SPARSE"),
            ],
            consequence=[("practitioner_acceptability", "UNACCEPTABLE")],
        ),
        FuzzyRule(
            premise=[
                ("aspects", "PERTINENT"),
            ],
            consequence=[("practitioner_acceptability", "TOTALLY_ACCEPTABLE")],
        ),
        FuzzyRule(
            premise=[
                ("aspects", "EXTENSIVE"),
            ],
            consequence=[("practitioner_acceptability", "UNACCEPTABLE")],
        ),
    ]

    expert_rules = [
        FuzzyRule(
            premise=[
                ("aspects", "SPARSE"),
            ],
            consequence=[("expert_acceptability", "TOTALLY_UNACCEPTABLE")],
        ),
        FuzzyRule(
            premise=[
                ("aspects", "PERTINENT"),
            ],
            consequence=[("expert_acceptability", "NEUTRAL")],
        ),
        FuzzyRule(
            premise=[
                ("aspects", "EXTENSIVE"),
            ],
            consequence=[("expert_acceptability", "TOTALLY_ACCEPTABLE")],
        ),
    ]

    fuzzy_rules = []
    fuzzy_rules.extend(outsider_rules)
    fuzzy_rules.extend(practitioner_rules)
    fuzzy_rules.extend(expert_rules)

    model = DecompositionalInference(
        and_operator="min",
        or_operator="max",
        implication_operator="Rc",
        composition_operator="max-min",
        production_link="max",
        defuzzification_operator="cog",
    )

    model(
        variables=fuzzy_variables,
        rules=fuzzy_rules,
        aspects=6
    )

    result = model.defuzzificated_infered_memberships
    print(f'Highest score acceptability:')
    for key in get_keys_with_highest_value(result):
        print(f'{key} : {result[key]}')

    plt.savefig('model.png')
    plt.clf()


