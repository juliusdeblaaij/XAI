from non_blocking_process import AbstractNonBlockingProcess
import lime
from lime.lime_text import LimeTextExplainer


class FaithfulnessAlgorithm(AbstractNonBlockingProcess):

    def _do_work(self, text_instance=None, class_names=None, probabilities=None, classifier_fn=None):
        if class_names is None:
            raise ValueError("Attempted to get faithfulness score without supplying 'text_instance'.")
        if class_names is None:
            raise ValueError("Attempted to get faithfulness score without supplying 'class_names'.")
        if probabilities is None:
            raise ValueError("Attempted to get faithfulness score without supplying 'probabilities'.")
        if classifier_fn is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'classifier_fn'.")

        explainer = LimeTextExplainer(class_names=class_names)

        exp = explainer.explain_instance(text_instance=text_instance,
                                         classifier_fn=classifier_fn,
                                         top_labels=3)

        return f"{exp}"
