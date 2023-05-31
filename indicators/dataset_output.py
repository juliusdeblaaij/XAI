from DataEvent import DataEvent
from indicators.CompositeIndicator import CompositeIndicator
from EventsBroadcaster import broadcast_data
import pandas as pd
import numpy as np


class DatasetOutput(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"testing_cases": [], "testing_labels": [], "classification_results": {}, "explanations": [],
                "meaningfulness_scores": [],
                "explanation_accuracy_decisions": [], "explanation_accuracy_scores": [],
                "adherence_to_knowledge_limits": [],
                "correct_outside_knowledge_domain_flags": [], "adherence_to_similarity_threshold_flags": [],
                "adherence_to_similarity_distance_threshold_flags": [], "faithfulness_scores": [],
                "practitioner_acceptability_scores": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        cases = np.array(kwargs.get("testing_cases"))
        ids = []

        for i in range(0, len(cases)):
            ids.append(i)

        ids = np.asarray(ids)

        labels = np.array(kwargs.get("testing_labels"))
        predicted_labels = np.array(kwargs.get("classification_results").get("EstLabs").flatten()).astype(int)
        explanations = kwargs.get("explanations")

        meaningfulness = np.array(kwargs.get("meaningfulness_scores"))
        explanation_accuracy_decisions = np.array(kwargs.get("explanation_accuracy_decisions"))
        explanation_accuracy_scores = np.array(kwargs.get("explanation_accuracy_scores"))

        adherence_to_knowledge_limits = np.array(kwargs.get("adherence_to_knowledge_limits"))
        correct_outside_knowledge_domain_flags = np.array(kwargs.get("correct_outside_knowledge_domain_flags"))
        adherence_to_similarity_threshold_flags = np.array(kwargs.get("adherence_to_similarity_threshold_flags"))
        adherence_to_similarity_distance_threshold_flags = np.array(
            kwargs.get("adherence_to_similarity_distance_threshold_flags"))

        faithfulness_scores = kwargs.get("faithfulness_scores")
        practitioner_acceptability_scores = kwargs.get("practitioner_acceptability_scores")

        df = pd.DataFrame({"id": ids, "x_test": cases, "y_test": labels, "y_pred": predicted_labels,
                           "explanations": explanations,

                           "meaningfulness": meaningfulness,

                           "practitioner_acceptability": practitioner_acceptability_scores,
                           "faithfulness": faithfulness_scores,
                           "explanation_accuracy_scores": explanation_accuracy_scores,
                           "explanation_accuracy_decisions": explanation_accuracy_decisions,

                           "okd": correct_outside_knowledge_domain_flags,
                           "similarity_threshold": adherence_to_similarity_threshold_flags,
                           "similarity_dist_threshold": adherence_to_similarity_distance_threshold_flags,
                           "adherence_to_knowledge_limits": adherence_to_knowledge_limits})

        df.to_csv("xai_results.csv", index=False)

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())