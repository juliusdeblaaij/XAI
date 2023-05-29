from non_blocking_process import AbstractNonBlockingProcess
import numpy as np

from run_xdnn import RunxDNN
from d2v import doc2vec
from myutils import *


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

            return training_results

        if mode == "Classify":
            training_results = xDNN.load_training_results()

            if training_results is None:
                raise ValueError("Tried to run xDNN classification without providing 'training_results'.")
            if features is None:
                raise ValueError('Tried to run xDNN classification without providing features.')
            classification_results = xDNN.classify(training_results=training_results, features=features)

            return classification_results
