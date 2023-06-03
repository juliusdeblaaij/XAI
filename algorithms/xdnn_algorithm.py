from multiprocessing import current_process

from non_blocking_process import AbstractNonBlockingProcess
from run_xdnn import RunxDNN


class xDNNAlgorithm(AbstractNonBlockingProcess):

    def _do_work(self, training_results=None, mode=None, cases=None, features=None, labels=None):
        if mode == None:
            raise ValueError('Tried to run xDNN without specifying a mode (Learning, Validation or Classify).')

        xDNN = RunxDNN()

        if mode == "Learning":
            if cases is None:
                raise ValueError('Tried to run xDNN classification without providing cases.')
            if features is None:
                raise ValueError('Tried to run xDNN classification without providing features.')
            if labels is None:
                raise ValueError('Tried to run xDNN classification without providing labels.')

            training_results = xDNN.train(cases=cases, features=features, labels=labels)

            result = {
                "training_results": training_results["xDNNParms"],
                "pid": current_process().pid
            }

            return result

        if mode == "Classify":
            training_results = xDNN.load_training_results()

            if training_results is None:
                raise ValueError("Tried to run xDNN classification without providing 'training_results'.")
            if features is None:
                raise ValueError('Tried to run xDNN classification without providing features.')
            classification_results = xDNN.classify(training_results=training_results, features=features)

            result = {
                "classification_results": {"EstLabs": classification_results["EstLabs"],
                                           "Scores": classification_results["Scores"],
                                           "Similarities": classification_results["Similarities"]},
                "pid": current_process().pid
            }

            return result
