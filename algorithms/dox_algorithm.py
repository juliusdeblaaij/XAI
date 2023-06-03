from multiprocessing import current_process

from dataset_cleaner import filter_allowed_words, filter_allowed_words_in_sentences
from doxpy.models.estimation.dox_estimator import DoXEstimator
from doxpy.models.knowledge_extraction.knowledge_graph_manager import KnowledgeGraphManager
from doxpy.models.reasoning.answer_retriever import AnswerRetriever
from nltk.tokenize import sent_tokenize

from dox_utils import get_details_from, get_aspects_from, question_template_list
from non_blocking_process import AbstractNonBlockingProcess
from doxpy.models.knowledge_extraction.knowledge_graph_extractor import KnowledgeGraphExtractor

import confuse

config = confuse.Configuration('XAI', __name__)


class DoXAlgorithm(AbstractNonBlockingProcess):

    def _do_work(self, cases=None, explanandum_aspects=None):

        knowledge_manager_options = config["knowledge_manager_options"].get(dict)
        concept_classifier_options = config["concept_classifier_options"].get(dict)
        sentence_classifier_options = config["sentence_classifier_options"].get(dict)
        archetype_fitness_options = config["archetype_fitness_options"].get(dict)
        graph_builder_options = config["graph_builder_options"].get(dict)

        average_dox_scores = []

        for i, case in enumerate(cases):
            sentences_in_case = sent_tokenize(case)
            filtered_sentences_in_case = filter_allowed_words_in_sentences(sentences_in_case, keep_stop_words=True) # Keeping stop words increases meaningfulness scores

            phi_sentences = filtered_sentences_in_case

            explainable_information_graph = KnowledgeGraphExtractor(graph_builder_options).set_content_list(phi_sentences,
                                                                                                            remove_stopwords=False,
                                                                                                            remove_numbers=False,
                                                                                                            avoid_jumps=True).build()

            kg_manager = KnowledgeGraphManager(knowledge_manager_options, explainable_information_graph)

            qa = AnswerRetriever(
                kg_manager=kg_manager,
                concept_classifier_options=concept_classifier_options,
                sentence_classifier_options=sentence_classifier_options,
            )

            explanandum_aspect_list = get_aspects_from(explainable_information_graph)

            question_generator = lambda question_template, concept_label: question_template.replace('{X}',
                                                                                                    concept_label)

            dox_estimator = DoXEstimator(qa)

            dox = dox_estimator.estimate(
                aspect_uri_iter=list(explanandum_aspect_list),
                query_template_list=question_template_list,
                question_generator=question_generator,
                **archetype_fitness_options,
            )

            average_dox = dox_estimator.get_weighted_degree_of_explainability(dox, archetype_weight_dict=None)

            average_dox_scores.append(average_dox)
            print(f"Calculated {i + 1} out of {len(cases)} DoX.")

        result = {
            "average_dox_scores": average_dox_scores,
            "pid": current_process().pid
        }

        return result
