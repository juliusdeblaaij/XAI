from sklearn.model_selection import train_test_split

from DataEvent import DataEvent
from algorithms.dox_algorithm_adapter import DoxAlgorithmAdapter
from indicators.CompositeIndicator import CompositeIndicator
from EventsBroadcaster import broadcast_data


class MeaningfulnessIndicator(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"practitioner_aspects": [], "explanations": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        explanandum_aspects = kwargs.get("practitioner_aspects")
        explanations = kwargs.get("explanations")

        kwargs = {"explanandum_aspects": explanandum_aspects,
                  "cases": explanations}

        dox_algo = DoxAlgorithmAdapter()
        dox_algo.run(callback=self.on_dox_calculated, **kwargs)

    def on_dox_calculated(self, data):
        broadcast_data({"meaningfulness_scores": data})
    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
