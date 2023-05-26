from DataEvent import DataEvent
from EventsBroadcaster import subscribe, broadcast, subscribe_to_data
from algorithms.AdditionAlgorithm import AdditionAlgorithm
from indicators.CompositeIndicator import CompositeIndicator

class IntegerCompositeIndicator(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    def input_signature(self) -> dict:
        return {"A": 0, "B": 0}

    def run_algorithm(self, data: dict):
        self.input_data().clear()
        addition = AdditionAlgorithm()
        addition.run(**data)

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
