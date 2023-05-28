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
        return {"training_results": {}, "cases": [], "features": [], "labels": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        kwargs = {"mode": "Classify"}

        xdnn_algo = xDNNAlgorithmAdapter()
        xdnn_algo.run(callback=self.on_xdnn_trained, **kwargs)

    def on_xdnn_trained(self, data):
        broadcast_data(data)

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
