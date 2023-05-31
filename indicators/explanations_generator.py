import numpy as np

from DataEvent import DataEvent
from EventsBroadcaster import broadcast_data
from algorithms.faithfulness_algorithm_adapter import FaithfulnessAlgorithmAdapter
from algorithms.xdnn_algorithm_adapter import xDNNAlgorithmAdapter
from d2v import doc2vec
from indicators.CompositeIndicator import CompositeIndicator
from myutils import pre_process_text, sort_with_indices, label_to_story_point


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
        return {"testing_cases": [], "training_results": {}, "classification_results": {}}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        predicted_labels = kwargs.get("classification_results").get("EstLabs").flatten()
        predicted_labels = predicted_labels.astype(int)
        xdnn_training_results = kwargs.get("training_results")
        xdnn_classification_results = kwargs.get("classification_results")

        explanations = []

        for i, case in enumerate(kwargs["testing_cases"]):
            predicted_label = predicted_labels[i]

            similarities = xdnn_classification_results.get("Similarities")

            in_label_similarities = list(similarities[i][predicted_label])


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

            explanation = \
'User story is "{case}".' + \
f"""\nIF (User story is similar to "{sorted_cases[0]}") OR
IF (User story is similar to "{sorted_cases[1]}") OR
IF (User story is similar to "{sorted_cases[2]}")
Then '{label_to_story_point(predicted_label)}'"""
            explanations.append(explanation)

        broadcast_data({"explanations": explanations})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
