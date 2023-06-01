import json
import re

from doxpy.models.estimation.dox_estimator import DoXEstimator
from doxpy.models.knowledge_extraction.knowledge_graph_manager import KnowledgeGraphManager
from doxpy.models.reasoning.answer_retriever import AnswerRetriever
from get_aspects import get_aspects

from myutils import extract_knowledge_graph, split_text_into_sentences
from non_blocking_process import AbstractNonBlockingProcess
from lime.lime_text import LimeTextExplainer
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

        for case in cases:

            PHI = split_text_into_sentences(case)

            print('Building Graph..')
            explainable_information_graph = KnowledgeGraphExtractor(graph_builder_options).set_content_list(PHI,
                                                                                                                 remove_stopwords=False,
                                                                                                                 remove_numbers=False,
                                                                                                                 avoid_jumps=True).build()

            PHI_aspects = get_aspects(explainable_information_graph)
            print(PHI_aspects)

            # save_graphml(explainable_information_graph, 'knowledge_graph')
            print('Graph size:', len(explainable_information_graph))
            print("Graph's Clauses:", len(list(filter(lambda x: '{obj}' in x[1], explainable_information_graph))))
            #############
            print('Building Question Answerer..')
            # betweenness_centrality = get_betweenness_centrality(filter(lambda x: '{obj}' in x[1], explainable_information_graph))
            kg_manager = KnowledgeGraphManager(knowledge_manager_options, explainable_information_graph)
            qa = AnswerRetriever(
                # Using qa_dict_list also for getting the archetype_fitness_dict might over-estimate the median pertinence of some archetypes (and in a different way for each), because the QA Extractor is set to prefer a higher recall to a higher precision.
                kg_manager=kg_manager,
                concept_classifier_options=concept_classifier_options,
                sentence_classifier_options=sentence_classifier_options,
                # betweenness_centrality= betweenness_centrality,
            )
            ########################################################

            ### Get explanandum aspects
            explanandum_aspect_list = explanandum_aspects
            print('Important explicandum aspects:', len(explanandum_aspect_list))
            print(json.dumps(explanandum_aspect_list, indent=4))

            ### Define archetypal questions
            question_template_list = [  # Q: the archetypal questions
                ##### AMR
                'What is {X}?',
                'Who is {X}?',
                'How is {X}?',
                'Where is {X}?',
                'When is {X}?',
                'Which {X}?',
                'Whose {X}?',
                'Why {X}?',
                ##### Discourse Relations
                'In what manner is {X}?',  # (25\%),
                'What is the reason for {X}?',  # (19\%),
                'What is the result of {X}?',  # (16\%),
                'What is an example of {X}?',  # (11\%),
                'After what is {X}?',  # (7\%),
                'While what is {X}?',  # (6\%),
                'In what case is {X}?',  # (3),
                'Despite what is {X}?',  # (3\%),
                'What is contrasted with {X}?',  # (2\%),
                'Before what is {X}?',  # (2\%),
                'Since when is {X}?',  # (2\%),
                'What is similar to {X}?',  # (1\%),
                'Until when is {X}?',  # (1\%),
                'Instead of what is {X}?',  # (1\%),
                'What is an alternative to {X}?',  # ($\leq 1\%$),
                'Except when it is {X}?',  # ($\leq 1\%$),
                '{X}, unless what?',  # ($\leq 1\%$).
            ]

            ### Define a question generator
            question_generator = lambda question_template, concept_label: question_template.replace('{X}',
                                                                                                    concept_label)

            ### Initialise the DoX estimator
            dox_estimator = DoXEstimator(qa)
            ### Estimate DoX
            dox = dox_estimator.estimate(
                aspect_uri_iter=list(explanandum_aspect_list),
                query_template_list=question_template_list,
                question_generator=question_generator,
                **archetype_fitness_options,
            )
            print(f'DoX:', json.dumps(dox, indent=4))
            ### Compute the average DoX
            average_dox = dox_estimator.get_weighted_degree_of_explainability(dox, archetype_weight_dict=None)
            print('Average DoX:', average_dox)

            average_dox_scores.append(average_dox)

        return average_dox_scores
