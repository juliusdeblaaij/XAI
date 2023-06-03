import re

import joblib
import numpy as np

from algorithms.xdnn_algorithm_adapter import xDNNAlgorithmAdapter
from dataset_cleaner import filter_allowed_words
from myutils import pad_array
from non_blocking_process import AbstractNonBlockingProcess
from lime.lime_text import LimeTextExplainer
from multiprocessing import current_process

class FaithfulnessAlgorithm(AbstractNonBlockingProcess):

    def _do_work(self, cases=None, label_names=None, predicted_labels=None, xdnn_training_results=None, xdnn_classification_results=None, explanations=None):
        if cases is None:
            raise ValueError("Attempted to get faithfulness score without supplying 'cases'.")
        if label_names is None:
            raise ValueError("Attempted to get faithfulness score without supplying 'label_names'.")
        if predicted_labels is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'predicted_classes'.")
        if xdnn_training_results is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'xdnn_training_results'.")
        if xdnn_classification_results is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'xdnn_classification_results'.")
        if explanations is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'explanations'.")

        predicted_labels = predicted_labels.flatten().astype(float).astype(int).flatten()
        cases = cases.flatten()
        explanations = explanations.flatten()

        explainer = LimeTextExplainer(class_names=label_names)

        faithfulness_scores = []

        for i, case in enumerate(cases):
            predicted_label = predicted_labels[i]
            explanation = explanations[i]
            explanation_with_case = explanation.replace("{case}", case)

            case_explanation = explainer.explain_instance(text_instance=case,
                                             classifier_fn=self.classifier_fn,
                                             top_labels=10,
                                             num_samples=500) # 500



            case_contributing_words, case_word_contributions = zip(*case_explanation.as_list(label=predicted_label))

            case_contributing_words = list(case_contributing_words)
            case_word_contributions = list(case_word_contributions)

            if "user" in case_contributing_words: case_contributing_words.remove("user")
            if "story" in case_contributing_words: case_contributing_words.remove("story")

            postive_contributing_words = []

            for j, word in enumerate(case_contributing_words):
                if case_word_contributions[j] > 0:
                    postive_contributing_words.append(word)

            total_overlapping_words = 0

            if len(postive_contributing_words) != 0:

                for word in postive_contributing_words:
                    wordcount = self.count_word_occurrences(explanation, word)

                    if wordcount > 1:
                        wordcount = 1

                    total_overlapping_words += wordcount

                overlap_ratio = total_overlapping_words / len(postive_contributing_words)

                # print(f"{explanation_with_case}\nHas an overlap ratio of: {overlap_ratio}."
                #      f"\nPositively contributing words: {postive_contributing_words}")

                faithfulness_scores.append(overlap_ratio)


            else:
                # print(f"{explanation_with_case}\nHas an overlap ratio of: {0}")
                faithfulness_scores.append(0)

            print(f"Calculated {i+1} /{len(cases)} faithfulness.")

        result = {
            "faithfulness_scores": faithfulness_scores,
            "pid": current_process().pid
        }

        return result

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def count_word_occurrences(self, string: str, word: str) -> int:
        # Convert the string to lowercase to perform case-insensitive search
        string = string.lower()

        # Create a regular expression pattern to match the word as a whole word
        pattern = r'\b' + re.escape(word) + r'\b'

        # Count the occurrences of the word in the string using re.finditer
        count = sum(1 for _ in re.finditer(pattern, string))

        return count

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
                    cleaned_text = filter_allowed_words(string)

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

    def xdnn_callback(self, kwargs):
        self.local_data()["results"] = kwargs.get("classification_results")
