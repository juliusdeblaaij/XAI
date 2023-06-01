import numpy as np

from DataEvent import DataEvent
from EventsBroadcaster import broadcast_data
from algorithms.faithfulness_algorithm_adapter import FaithfulnessAlgorithmAdapter
from algorithms.xdnn_algorithm_adapter import xDNNAlgorithmAdapter
from d2v import doc2vec
from indicators.CompositeIndicator import CompositeIndicator
from myutils import pre_process_text, sort_with_indices, label_to_story_point


def generate_valid_user_story_explanation(case: str, predicted_label: int, similar_cases: list) -> str:

    if len(similar_cases) == 0:
        raise ValueError("The 'similar_cases' list must not be empty.")

    sp = label_to_story_point(predicted_label)

    explanation = f'"Introduction:\n' + \
                  f'XUSP is an algorithm that is designed to automatically determine the amount of story points for a given user story.\n' + \
                  f'A user story is a description of a feature that must be developed for a specific end-user ofa product, who wants the feature because it creates certain value.' + \
                  f'An example of a user story is: "As a new user, I would like to have a contact form, so that I can contact customer service for help with product setup."\n' + \
                  f'The prediction of the story point value is made using a specific process called xDNN, which classifies items based on the similarity between it and already learned items.\n' + \
                  f'Prediction:\n' + \
                  f'{case.capitalize()}" is a user story, and is worth {sp} story points; because it is most similar to:\n{similar_cases[0].capitalize()}'

    similar_cases_sentences = ""

    if len(similar_cases) > 1:

        for i, similar_case in enumerate(similar_cases):
            appendage = ""

            if i == 0:
                continue
            if i == 1 or i == len(similar_cases) - 1:
                appendage = f' and similar to "{similar_case.capitalize()}"'

                if i == len(similar_cases) - 1:
                    appendage += f' (which are all user stories with a worth of {sp} story points).'
            else:
                appendage = f', "{similar_case.capitalize()}"'

            similar_cases_sentences += appendage
    else:
        similar_cases_sentences = "."

    return explanation + similar_cases_sentences


def generate_valid_non_user_story_explanation(case: str, similar_cases: list) -> str:

    if len(similar_cases) == 0:
        raise ValueError('The "similar_cases" list must not be empty.')

    explanation = f'"Introduction:\n' + \
                  f'XUSP is an algorithm that is designed to automatically determine the amount of story points for a given user story.\n' + \
                  f'A user story is a description of a feature that must be developed for a specific end-user ofa product, who wants the feature because it creates certain value.' + \
                  f'An example of a user story is: "As a new user, I would like to have a contact form, so that I can contact customer service for help with product setup."\n' + \
                  f'The prediction of the story point value is made using a specific process called xDNN, which classifies items based on the similarity between it and already learned items.\n' + \
                  f'Prediction:\n' + \
                  f'"{case.capitalize()}" is a not user story; because it is most similar to:\n"{similar_cases[0].capitalize()}"'

    similar_cases_sentences = ""

    if len(similar_cases) > 1:

        for i, similar_case in enumerate(similar_cases):
            appendage = ""

            if i == 0:
                continue
            if i == 1 or i == len(similar_cases) - 1:
                appendage = f' and similar to "{similar_case.capitalize()}"'

                if i == len(similar_cases) - 1:
                    appendage += " (which are not user stories)."
            else:
                appendage = f', "{similar_case.capitalize()}"'

            similar_cases_sentences += appendage
    else:
        similar_cases_sentences = "."

    return explanation + similar_cases_sentences


def generate_outside_knowledge_limits_explanation(case: str, predicted_label: int, similar_cases: list, similarity_threshold_flag: int) -> str:

    if len(similar_cases) == 0:
        raise ValueError("The 'similar_cases' list must not be empty.")

    sp = label_to_story_point(predicted_label)

    explanation = f'"Introduction:\n' + \
                  f'XUSP is an algorithm that is designed to automatically determine the amount of story points for a given user story.\n' + \
                  f'A user story is a description of a feature that must be developed for a specific end-user ofa product, who wants the feature because it creates certain value.' + \
                  f'An example of a user story is: "As a new user, I would like to have a contact form, so that I can contact customer service for help with product setup."\n' + \
                  f'The prediction of the story point value is made using a specific process called xDNN, which classifies items based on the similarity between it and already learned items.\n' + \
                  f' XUSP could not provide an answer as the amount of similarity between "{case.capitalize()}" and "{similar_cases[0].capitalize()}" does not exceed the minimum threshold of 75%.'

    return explanation


def get_top_k_similar_cases(index: int, k: int, predicted_label: int, xdnn_training_results: dict,
                            xdnn_classification_results: dict) -> list:
    similarities = xdnn_classification_results.get("Similarities")

    in_label_similarities = list(similarities[index][predicted_label])

    training_parameters = xdnn_training_results.get("xDNNParms").get("Parameters")
    classes = list(training_parameters[predicted_label].get("Prototype").values())

    sorted_in_label_similarities, original_indices = sort_with_indices(in_label_similarities)

    sorted_in_label_similarities = np.flip(sorted_in_label_similarities)
    original_indices = np.flip(original_indices)

    classes_with_similarities = []
    for i in range(0, len(classes)):
        classes_with_similarities.append([classes[original_indices[i]], sorted_in_label_similarities[i]])

    sorted_cases = [item[0] for item in classes_with_similarities]
    sorted_similarities = [item[1] for item in classes_with_similarities]

    top_cases = []

    for i, case in enumerate(sorted_cases):
        if i >= k:
            break

        top_cases.append(case)

    return top_cases


class ExplanationsGenerator(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"testing_cases": [],
                "training_results": {},
                "classification_results": {},
                "adherence_to_knowledge_limits": [],
                "correct_outside_knowledge_domain_flags": [],
                "adherence_to_similarity_threshold_flags": [],
        }

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        predicted_labels = kwargs.get("classification_results").get("EstLabs").flatten()
        predicted_labels = predicted_labels.astype(int)
        xdnn_training_results = kwargs.get("training_results")
        xdnn_classification_results = kwargs.get("classification_results")

        adherence_to_knowledge_limits = kwargs.get("adherence_to_knowledge_limits")
        correct_outside_knowledge_domain_flags = kwargs.get("correct_outside_knowledge_domain_flags")
        adherence_to_similarity_threshold_flags = kwargs.get("adherence_to_similarity_threshold_flags")

        explanations = []

        for i, case in enumerate(kwargs["testing_cases"]):
            predicted_label = predicted_labels[i]
            explanation = ""

            top_3_similar_cases = get_top_k_similar_cases(index=i, k=3, predicted_label=predicted_label,
                                                          xdnn_training_results=xdnn_training_results,
                                                          xdnn_classification_results=xdnn_classification_results)

            if adherence_to_similarity_threshold_flags[i] == 1:

                if predicted_label > 0:

                    explanation = generate_valid_user_story_explanation(case=case, predicted_label=predicted_label, similar_cases=top_3_similar_cases)
                else:
                    explanation = generate_valid_non_user_story_explanation(case=case,
                                                                           similar_cases=top_3_similar_cases)
            else:

                similarity_threshold_flag = adherence_to_similarity_threshold_flags[i]

                explanation = generate_outside_knowledge_limits_explanation(case, predicted_label, top_3_similar_cases,
                                                                            similarity_threshold_flag)

            explanations.append(explanation)

        broadcast_data({"explanations": explanations})





    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())