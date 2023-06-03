import numpy as np

from DataEvent import DataEvent
from algorithms.xdnn_algorithm_adapter import xDNNAlgorithmAdapter
from indicators.CompositeIndicator import CompositeIndicator
from EventsBroadcaster import broadcast_data


class xDNNTrainer(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"training_features": [], "training_cases": [], "training_labels": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        features = np.array(kwargs.get("training_features"))
        cases = np.array(kwargs.get("training_cases"))
        labels = np.array(kwargs.get("training_labels"))

        kwargs = {"mode": "Learning", "features": features, "cases": cases, "labels": labels}

        # TODO: kijk in xdnn sample code om te zien welk datatype de labels en images hebben
        xdnn_algo = xDNNAlgorithmAdapter()
        xdnn_algo.run(callback=self.on_xdnn_trained, **kwargs)
    
    def on_xdnn_trained(self, kwargs):
        broadcast_data({"training_results": kwargs.get("training_results")})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
