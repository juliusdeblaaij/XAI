from matplotlib import pyplot as plt

from non_blocking_process import AbstractNonBlockingProcess
from lime.lime_text import LimeTextExplainer


class FaithfulnessAlgorithm(AbstractNonBlockingProcess):

    def _do_work(self, cases=None, class_names=None, classifier_fn=None, predicted_classes=None, xdnn_training_results=None, xdnn_classification_results=None):
        if cases is None:
            raise ValueError("Attempted to get faithfulness score without supplying 'cases'.")
        if class_names is None:
            raise ValueError("Attempted to get faithfulness score without supplying 'class_names'.")
        if classifier_fn is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'classifier_fn'.")
        if predicted_classes is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'predicted_classes'.")
        if xdnn_training_results is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'xdnn_training_results'.")
        if xdnn_classification_results is None:
            raise ValueError("Attempted to get faithfulness score without specifying 'xdnn_classification_results'.")

        predicted_classes = predicted_classes.flatten().astype(int)

        for i, case in enumerate(cases):
            predicted_class = predicted_classes[i]

            closest_class_indices = xdnn_classification_results.get("ClosestClassIndices")
            most_similar_prototype_index = closest_class_indices[i][predicted_class] + 1
            training_parameters = xdnn_training_results.get("xDNNParms").get("Parameters")
            prototypes = training_parameters[predicted_class].get("Prototype")
            most_similar_prototype = prototypes[most_similar_prototype_index]

            print(f"Most similar prototype to:\n{case},\nis: {predicted_class} #{most_similar_prototype_index} {most_similar_prototype}")



        explainer = LimeTextExplainer(class_names=class_names)

        for i, case in enumerate(cases):
            exp = explainer.explain_instance(text_instance=case,
                                             classifier_fn=classifier_fn,
                                             top_labels=3,
                                             num_samples=100)

            data = exp.as_list(label=1)

            print('Explanation for class %s' % class_names[1])
            # print('\n'.join(map(str, data)))

            # Separate words and contributions
            words, contributions = zip(*data)
