import numpy as np

from DataEvent import DataEvent
from algorithms.faithfulness_algorithm_adapter import FaithfulnessAlgorithmAdapter
from algorithms.xdnn_algorithm_adapter import xDNNAlgorithmAdapter
from d2v import doc2vec
from indicators.CompositeIndicator import CompositeIndicator
from myutils import pre_process_text
from EventsBroadcaster import broadcast_data


class xDNNClassifier(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"training_results": {}, "testing_cases": [], "testing_features": [], "testing_labels": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        training_results = kwargs["training_results"]
        testing_cases = kwargs["testing_cases"]
        testing_features = kwargs["testing_features"]
        testing_labels = kwargs["testing_labels"]

        kwargs = {
            "training_results": training_results,
            "mode": "Classify",
            "cases": np.array(testing_cases),
            "features": np.array(testing_features),
            "labels": np.array(testing_labels),
        }

        xdnn_algo = xDNNAlgorithmAdapter()
        xdnn_algo.run(callback=self.on_xdnn_classified, **kwargs)

    def on_xdnn_classified(self, data):
        broadcast_data({"classification_results": data})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
