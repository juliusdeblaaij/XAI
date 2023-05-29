from matplotlib import pyplot as plt

from non_blocking_process import AbstractNonBlockingProcess
from lime.lime_text import LimeTextExplainer


class FaithfulnessAlgorithm(AbstractNonBlockingProcess):

    def _do_work(self, cases=None, class_names=None, classifier_fn=None, predicted_labels=None, xdnn_training_results=None, xdnn_classification_results=None, explanations=None):
        if cases is None:
            raise ValueError("Attempted to get faithfulness score without supplying 'cases'.")
        if class_names is None:
            raise ValueError("Attempted to get faithfulness score without supplying 'class_names'.")
        if classifier_fn is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'classifier_fn'.")
        if predicted_labels is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'predicted_classes'.")
        if xdnn_training_results is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'xdnn_training_results'.")
        if xdnn_classification_results is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'xdnn_classification_results'.")
        if explanations is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'explanations'.")

        predicted_labels = predicted_labels.flatten().astype(int)

        explainer = LimeTextExplainer(class_names=class_names)

        for i, case in enumerate(cases):
            predicted_label = predicted_labels[i]
            explanation = explanations[i]

            case_explanation = explainer.explain_instance(text_instance=case,
                                             classifier_fn=classifier_fn,
                                             top_labels=10,
                                             num_samples=100)

            explanation_explanation = explainer.explain_instance(text_instance=explanation,
                                                          classifier_fn=classifier_fn,
                                                          top_labels=10,
                                                          num_samples=100)

            case_contributing_words, case_word_contributions = zip(*case_explanation.as_list(label=predicted_label))
            explanation_contributing_words, explanation_word_contributions = zip(*explanation_explanation.as_list(label=predicted_label))

            print(f"{case_contributing_words}\n{explanation_contributing_words}")
