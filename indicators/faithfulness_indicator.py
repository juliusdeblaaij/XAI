import numpy as np

from DataEvent import DataEvent
from algorithms.faithfulness_algorithm_adapter import FaithfulnessAlgorithmAdapter
from algorithms.xdnn_algorithm_adapter import xDNNAlgorithmAdapter
from d2v import doc2vec
from indicators.CompositeIndicator import CompositeIndicator
from myutils import pre_process_text


class FaithfulnessIndicator(CompositeIndicator):

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

        cleaned_cases = []

        for case in kwargs["testing_cases"]:
            cleaned_text = pre_process_text(case)
            cleaned_cases.append(cleaned_text)

        faithfulness_algo = FaithfulnessAlgorithmAdapter()

        xdnn_training_results = kwargs.get("training_results")
        xdnn_classification_results = kwargs.get("classification_results")

        predicted_classes = xdnn_classification_results.get("EstLabs")


        kwargs = {
            "cases": cleaned_cases,
            "class_names": ["Not a user story", "1 SP", "2 SP", "3 SP", "5 SP", "8 SP"],
            "classifier_fn": self.classifier_fn,
            "predicted_classes": predicted_classes,
            "xdnn_training_results": xdnn_training_results,
            "xdnn_classification_results": xdnn_classification_results,
        }

        faithfulness_algo.run(callback=self.on_faithfulness_calculated, **kwargs)

    def classifier_fn(self, data):
        results = self.xdnn_classifier(data)
        return results
    def xdnn_classifier(self, data=None, **kwargs):

        if data is not None:
            if type(data) == type([]):

                xdnn_algo = xDNNAlgorithmAdapter()
                kwargs["mode"] = "Classify"

                case_embeddings = []

                for i, string in enumerate(data):
                    cleaned_text = pre_process_text(string)

                    case_embedding = doc2vec(cleaned_text, 'd2v_23k_dbow.model')
                    case_embedding = np.array(case_embedding)
                    case_embeddings.append(case_embedding)
                    print(f'Embedded string {i} out of {len(data)}.')

                kwargs["cases"] = np.array(case_embeddings)

                xdnn_algo.run(callback=self.xdnn_callback, **kwargs)

                while self.local_data().get("results") is None:
                    pass

                scores = self.local_data().get("results").get("Scores")
                return scores

    def xdnn_callback(self, results):
        self.local_data()["results"] = results

    def on_faithfulness_calculated(self, data):
        print(f'Faithfulness: {data}')

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
