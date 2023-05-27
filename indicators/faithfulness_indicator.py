import numpy as np

from DataEvent import DataEvent
from algorithms.faithfulness_algorithm_adapter import FaithfulnessAlgorithmAdapter
from algorithms.xdnn_algorithm_adapter import xDNNAlgorithmAdapter
from doc2vec import doc2vec
from indicators.CompositeIndicator import CompositeIndicator
from myutils import pre_process_text


class FaithfulnessIndicator(CompositeIndicator):


    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"cases": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        xdnn_algo = xDNNAlgorithmAdapter()
        kwargs["mode"] = "Classify"

        cleaned_embeddings = []
        cleaned_cases =[]

        for case in kwargs["cases"]:
            cleaned_text = pre_process_text(case)
            cleaned_cases.append(cleaned_text)

            case_embedding = doc2vec(cleaned_text, 'd2v_23k_dbow.model')
            case_embedding = np.array(case_embedding)
            cleaned_embeddings.append(case_embedding)

        self.local_data()["cleaned_cases"] = cleaned_cases

        kwargs["cases"] = np.array(cleaned_embeddings)

        xdnn_algo.run(callback=self.on_xDNN_classified, **kwargs)


    def on_xDNN_classified(self, data):
        print(data)
        cleaned_cases = self.local_data().get("cleaned_cases")

        faithfulness_algo = FaithfulnessAlgorithmAdapter()

        kwargs = {
            "text_instance": cleaned_cases[0],
            "class_names": ["Not a user story", "1 SP", "2 SP", "3 SP", "5 SP", "8 SP"],
            "probabilities": data["Scores"][0]
        }

        faithfulness_algo.run(callback=self.on_faithfulness_calculated, **kwargs)

    def on_faithfulness_calculated(self, data):
        print(f'Faithfulness: {data}')

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
