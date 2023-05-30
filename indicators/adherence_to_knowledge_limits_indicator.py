from DataEvent import DataEvent
from EventsBroadcaster import broadcast_data
from indicators.CompositeIndicator import CompositeIndicator


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

        minimum_similarity_threshold = 0.9
        minimum_similarity_distance_threshold = 0.05

        for i in range(0, len(testing_labels)):
            predicted_label = predicted_labels[i]

            similarities = xdnn_classification_results.get("Similarities")
            scores = xdnn_classification_results.get("Scores")
            current_classification_similarity_score = scores[i][predicted_label]

            # b) are too uncertain (not meeting a confidence threshold
            if current_classification_similarity_score <= minimum_similarity_threshold:
                adherence_to_similarity_threshold_flags.append(0)
            else:
                adherence_to_similarity_threshold_flags.append(1)

            # TODO: remove unnecesary nesting of the similarities withing a cases' label ([0])
            in_label_similarities = list(similarities[i][predicted_label][0])

            training_parameters = xdnn_training_results.get("xDNNParms").get("Parameters")
            classes = list(training_parameters[predicted_label].get("Prototype").values())

            classes_with_similarities = []
            for i, c in enumerate(classes):
                classes_with_similarities.append([c, in_label_similarities[i]])

            sorted_cases_with_similarities = sorted(classes_with_similarities, key=lambda x: x[1], reverse=True)

            sorted_cases = [item[0] for item in sorted_cases_with_similarities]
            sorted_similarities = [item[1] for item in sorted_cases_with_similarities]

            # c) or being too close to the runner up classification).
            if sorted_similarities[0] - sorted_similarities[1] <= minimum_similarity_distance_threshold:
                adherence_to_similarity_distance_threshold_flags.append(0)
            else:
                adherence_to_similarity_distance_threshold_flags.append(1)

        for i in range(0, len(predicted_labels)):

            if correct_outside_knowledge_domain_flags[i] == 1 \
                    and adherence_to_similarity_threshold_flags[i] == 1 \
                    and adherence_to_similarity_distance_threshold_flags == 1:

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
