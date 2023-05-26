from abc import ABC

from DataEvent import DataEvent
from EventsBroadcaster import subscribe_to_data
from indicators.CompositeIndicator import CompositeIndicator


class HippStringPrinter(CompositeIndicator):
    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    def input_signature(self) -> dict:
        return {"hip_string": ""}

    def run_algorithm(self, data: dict):
        self.input_data().clear()

        print(f"Hippe string:\n{data}")

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())