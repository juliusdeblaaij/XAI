import numpy as np

from DataEvent import DataEvent
from EventsBroadcaster import broadcast_data
from indicators.CompositeIndicator import CompositeIndicator
from myutils import sort_with_indices


class AdherenceToKnowledgeLimitsIndicator(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"training_results": {}, "classification_results": {}, "testing_labels": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        testing_labels = kwargs.get("testing_labels")
        predicted_labels = kwargs.get("classification_results").get("EstLabs").flatten()
        predicted_labels = predicted_labels.astype(int)
        xdnn_training_results = kwargs.get("training_results")
        xdnn_classification_results = kwargs.get("classification_results")

        # The system should correctly flag responses that are
        # a) classified as outside of the knowledge domain

        correct_outside_knowledge_domain_flags = []
        adherence_to_similarity_threshold_flags = []
        adherence_to_similarity_distance_threshold_flags = []
        adherence_to_knowledge_limits_flags = []

        for i, testing_label in enumerate(testing_labels):
            predicted_label = predicted_labels[i]

            if testing_label == 0 and predicted_label != 0:
                correct_outside_knowledge_domain_flags.append(0)
            else:
                correct_outside_knowledge_domain_flags.append(1)

        minimum_similarity_threshold = 0.75
        minimum_similarity_distance_threshold = 0.05

        similarities = list(xdnn_classification_results.get("Similarities"))

        for i in range(0, len(testing_labels)):
            predicted_label = predicted_labels[i]

            all_case_labels_similarities = list(similarities[i])
            case_predicted_label_similarities = all_case_labels_similarities[predicted_label]
            del all_case_labels_similarities[predicted_label]

            max_case_label_similarity = 0

            for case_label_similarities in all_case_labels_similarities:

                case_label_max_similarity = np.max(np.asarray(case_label_similarities))

                if case_label_max_similarity > max_case_label_similarity:
                    max_case_label_similarity = case_label_max_similarity

            training_parameters = xdnn_training_results.get("xDNNParms").get("Parameters")
            classes = list(training_parameters[predicted_label].get("Prototype").values())

            sorted_in_label_similarities, original_indices = sort_with_indices(case_predicted_label_similarities)

            sorted_in_label_similarities = np.flip(sorted_in_label_similarities)
            original_indices = np.flip(original_indices)

            classes_with_similarities = []
            for i in range(0, len(classes)):
                classes_with_similarities.append([classes[original_indices[i]], sorted_in_label_similarities[i]])

            sorted_cases = [item[0] for item in classes_with_similarities]
            sorted_similarities = [item[1] for item in classes_with_similarities]

            # b) are too uncertain (not meeting a confidence threshold
            if sorted_similarities[0] <= minimum_similarity_threshold:
                adherence_to_similarity_threshold_flags.append(0)
            else:
                adherence_to_similarity_threshold_flags.append(1)

            """# c) or being too close to the runner up classification).
            if sorted_similarities[0] - max_case_label_similarity <= minimum_similarity_distance_threshold:
                adherence_to_similarity_distance_threshold_flags.append(0)
            else:
                adherence_to_similarity_distance_threshold_flags.append(1)"""

        for i in range(0, len(predicted_labels)):

            if correct_outside_knowledge_domain_flags[i] == 1 \
                    and adherence_to_similarity_threshold_flags[i] == 1:
                    # and adherence_to_similarity_distance_threshold_flags[i] == 1:

                adherence_to_knowledge_limits_flags.append(1)
            else:
                adherence_to_knowledge_limits_flags.append(0)

        broadcast_data({"adherence_to_knowledge_limits": adherence_to_knowledge_limits_flags,
                        "correct_outside_knowledge_domain_flags": correct_outside_knowledge_domain_flags,
                        "adherence_to_similarity_threshold_flags": adherence_to_similarity_threshold_flags,
                        "adherence_to_similarity_distance_threshold_flags": adherence_to_similarity_distance_threshold_flags
                        })

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
