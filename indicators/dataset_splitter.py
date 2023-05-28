from sklearn.model_selection import train_test_split

from DataEvent import DataEvent
from indicators.CompositeIndicator import CompositeIndicator
from EventsBroadcaster import broadcast_data


class DatasetSplitter(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"features": [], "labels": [], "cases": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        training_features, testing_features, training_labels, testing_labels, training_cases, testing_cases = \
            train_test_split(kwargs["features"], kwargs["labels"], kwargs["cases"], test_size=0.2, random_state=42,
                             shuffle=True)

        broadcast_data({"training_features": training_features})
        broadcast_data({"testing_features": testing_features})

        broadcast_data({"training_labels": training_labels})
        broadcast_data({"testing_labels": testing_labels})

        broadcast_data({"training_cases": training_cases})
        broadcast_data({"testing_cases": testing_cases})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
