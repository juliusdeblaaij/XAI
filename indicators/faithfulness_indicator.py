from collections import OrderedDict

import joblib
import numpy as np

from DataEvent import DataEvent
from EventsBroadcaster import broadcast_data
from algorithms.faithfulness_algorithm_adapter import FaithfulnessAlgorithmAdapter
from algorithms.tfidf_vectorizer_algorithm_adapter import TfidfVectorizerAlgorithmAdapter
from algorithms.xdnn_algorithm_adapter import xDNNAlgorithmAdapter
from d2v import doc2vec
from dataset_cleaner import filter_allowed_words
from indicators.CompositeIndicator import CompositeIndicator
from myutils import pre_process_text, pad_array


workers = OrderedDict()


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
        return {"testing_cases": [], "training_results": {}, "classification_results": {}, "explanations": [], "vectorizer_file_path": ""}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        all_cleaned_cases = []

        for case in kwargs["testing_cases"]:
            cleaned_text = filter_allowed_words(case)
            all_cleaned_cases.append(cleaned_text)

        xdnn_training_results = kwargs.get("training_results")
        xdnn_classification_results = kwargs.get("classification_results")

        all_predicted_labels = xdnn_classification_results.get("EstLabs")
        all_explanations = kwargs.get("explanations")


        all_label_names = ["Not a user story", "1 SP", "2 SP", "3 SP", "5 SP", "8 SP"]

        case_details = np.column_stack((all_cleaned_cases, all_predicted_labels, all_explanations))

        split_case_details = np.array_split(np.asarray(case_details), 8)
        for case_details in split_case_details:

            batch_cleaned_cases, batch_predicted_labels, batch_explanations = np.array_split(case_details, 3, 1)

            kwargs = {
                "cases": batch_cleaned_cases,
                "predicted_labels": batch_predicted_labels,
                "explanations": batch_explanations,
                "label_names": all_label_names,
                "xdnn_training_results": xdnn_training_results,
                "xdnn_classification_results": xdnn_classification_results,
            }

            faithfulness_algo = FaithfulnessAlgorithmAdapter()
            faithfulness_algo.run(callback=self.on_faithfulness_calculated, **kwargs)
            workers[str(faithfulness_algo.pid)] = None


    def on_faithfulness_calculated(self, kwargs):
        faithfulness_scores = kwargs.get("faithfulness_scores")
        pid = kwargs.get("pid")

        for worker in workers:
            if str(pid) == worker:
                workers[str(pid)] = faithfulness_scores

        if not None in workers.values():

            all_faithfulness_scores = []

            for faithfulness_scores in workers.values():
                all_faithfulness_scores.extend(faithfulness_scores)

            broadcast_data({"faithfulness_scores": all_faithfulness_scores})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())

