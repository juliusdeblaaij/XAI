import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from DataEvent import DataEvent
from algorithms.tfidf_vectorizer_algorithm_adapter import TfidfVectorizerAlgorithmAdapter
from indicators.CompositeIndicator import CompositeIndicator
from EventsBroadcaster import broadcast_data


class Embedder(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"vectorizer_file_path": "", "embeddings_file_path": "", "cases": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        kwargs["mode"] = "Embedding"

        vectorizer_algo = TfidfVectorizerAlgorithmAdapter()
        vectorizer_algo.run(callback=self.on_cases_embedded, **kwargs)

    def on_cases_embedded(self, kwargs):
        features = kwargs.get("embeddings")

        broadcast_data({"features": list(features)})


    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
