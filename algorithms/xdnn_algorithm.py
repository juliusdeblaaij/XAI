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

            training_results = xDNN.train(cases=cases, features=features, labels=labels)

            return training_results

        if mode == "Classify":
            if training_results is None or cases == []:
                raise ValueError("Tried to run xDNN classification without providing 'training_results'.")
            if cases is None or cases == []:
                raise ValueError('Tried to run xDNN classification without providing cases.')
            classification_results = xDNN.classify(training_results=training_results, features=features)

            return classification_results
