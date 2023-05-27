from non_blocking_process import AbstractNonBlockingProcess
from lime.lime_text import LimeTextExplainer


class FaithfulnessAlgorithm(AbstractNonBlockingProcess):

    def _do_work(self, cases=None, class_names=None, classifier_fn=None, predicted_classes=None, actual_classes=None):
        if cases is None:
            raise ValueError("Attempted to get faithfulness score without supplying 'cases'.")
        if class_names is None:
            raise ValueError("Attempted to get faithfulness score without supplying 'class_names'.")
        if classifier_fn is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'classifier_fn'.")

        explainer = LimeTextExplainer(class_names=class_names)

        for i, case in enumerate(cases):
            exp = explainer.explain_instance(text_instance=case,
                                             classifier_fn=classifier_fn,
                                             top_labels=3,
                                             num_samples=100)

            print('Document id: %d' % i)
            print('Scores: ' % classifier_fn([case]))
            """print('Predicted class =', class_names[])
            print('True class: %s' % class_names[newsgroups_test.target[idx]])"""
