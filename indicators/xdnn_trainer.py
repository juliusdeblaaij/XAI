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
        return {"features": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        kwargs = {"mode": "Learning"}
        kwargs["features"] = np.array(kwargs.get("features"))

        # TODO: features is empty when run_algorithm is called!
        xdnn_algo = xDNNAlgorithmAdapter()
        xdnn_algo.run(callback=self.on_xdnn_trained, **kwargs)
    
    def on_xdnn_trained(self, data):
        broadcast_data({"training_results": data})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
