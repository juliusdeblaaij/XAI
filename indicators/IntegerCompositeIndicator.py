from DataEvent import DataEvent
from EventsBroadcaster import subscribe, broadcast
from indicators.CompositeIndicator import CompositeIndicator

class IntegerCompositeIndicator(CompositeIndicator):

    def __init__(self):
        subscribe('data_sent', self.on_event_happened)
        pass

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    def input_signature(self) -> dict:
        return {"A": 0, "B": 0}

    def run_algorithm(self, data: dict):
        self.input_data().clear()

        print(f"LIMEyo:\n{data}")
        broadcast('data_sent', {"hip_string": "hello world"})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
