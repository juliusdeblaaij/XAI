import joblib
import numpy as np

from DataEvent import DataEvent
from EventsBroadcaster import broadcast_data
from algorithms.faithfulness_algorithm_adapter import FaithfulnessAlgorithmAdapter
from algorithms.tfidf_vectorizer_algorithm_adapter import TfidfVectorizerAlgorithmAdapter
from algorithms.xdnn_algorithm_adapter import xDNNAlgorithmAdapter
from d2v import doc2vec
from indicators.CompositeIndicator import CompositeIndicator
from myutils import pre_process_text, pad_array


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
        return {"testing_cases": [], "training_results": {}, "classification_results": {}, "explanations": [], "vectorizer_file_path": ""}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        cleaned_cases = []

        for case in kwargs["testing_cases"]:
            cleaned_text = pre_process_text(case)
            cleaned_cases.append(cleaned_text)

        faithfulness_algo = FaithfulnessAlgorithmAdapter()

        xdnn_training_results = kwargs.get("training_results")
        xdnn_classification_results = kwargs.get("classification_results")

        predicted_labels = xdnn_classification_results.get("EstLabs")
        explanations = kwargs.get("explanations")

        kwargs = {
            "cases": cleaned_cases,
            "class_names": ["Not a user story", "1 SP", "2 SP", "3 SP", "5 SP", "8 SP"],
            "classifier_fn": self.classifier_fn,
            "predicted_labels": predicted_labels,
            "explanations": explanations,
            "xdnn_training_results": xdnn_training_results,
            "xdnn_classification_results": xdnn_classification_results,
        }

        faithfulness_algo.run(callback=self.on_faithfulness_calculated, **kwargs)

    def classifier_fn(self, data):
        results = self.xdnn_classifier(data)
        return results
    def xdnn_classifier(self, data=None, **kwargs):

        if data is not None:
            if type(data) == type([]):

                case_embeddings = []

                vectorizer_file_path = r"C:\Users\SKIKK\PycharmProjects\XAI\vectorizer.pkl"

                vectorizer = joblib.load(vectorizer_file_path)
                vectorizer.input = 'content'

                for i, string in enumerate(data):
                    cleaned_text = pre_process_text(string)

                    case_embedding = vectorizer.transform([cleaned_text])
                    case_embedding = np.array(case_embedding.data)
                    case_embedding = pad_array(case_embedding)
                    case_embeddings.append(case_embedding)
                    # print(f'Embedded string {i} out of {len(data)}.')

                xdnn_algo = xDNNAlgorithmAdapter()
                kwargs["mode"] = "Classify"

                kwargs["features"] = np.asarray(case_embeddings)

                xdnn_algo.run(callback=self.xdnn_callback, **kwargs)

                while self.local_data().get("results") is None:
                    pass

                scores = self.local_data().get("results").get("Scores")
                return scores

    def xdnn_callback(self, results):
        self.local_data()["results"] = results

    def on_faithfulness_calculated(self, data):
        data_np = np.asarray(data)
        print(f'Faithfulness len: {len(data_np)}, min: {np.min(data_np)}, max: {np.max(data)}')
        broadcast_data({"faithfulness_scores": data})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
