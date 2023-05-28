import numpy as np

from DataEvent import DataEvent
from algorithms.faithfulness_algorithm_adapter import FaithfulnessAlgorithmAdapter
from algorithms.tfidf_vectorizer_algorithm_adapter import TfidfVectorizerAlgorithmAdapter
from algorithms.xdnn_algorithm_adapter import xDNNAlgorithmAdapter
from d2v import doc2vec
from indicators.CompositeIndicator import CompositeIndicator
from myutils import pre_process_text
from EventsBroadcaster import broadcast_data


class CorpusTrainer(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"corpus_file_path": ""}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        kwargs["mode"] = "Learning"

        vectorizer_algo = TfidfVectorizerAlgorithmAdapter()
        vectorizer_algo.run(callback=self.on_vectorizer_trained, **kwargs)

    def on_vectorizer_trained(self, vectorizer_file_path):
        broadcast_data({"vectorizer_file_path": vectorizer_file_path})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
