from non_blocking_process import AbstractNonBlockingProcess
import numpy as np

from run_xdnn import RunxDNN
from d2v import doc2vec
from myutils import *


class xDNNAlgorithm(AbstractNonBlockingProcess):

    def _do_work(self, cases=None, mode=None):
        if mode == None:
            raise ValueError('Tried to run xDNN without specifying a mode (Learning, Validation or Classify).')

        xDNN = RunxDNN()

        if mode == "Classify":
            if cases is None or cases == []:
                raise ValueError('Tried to run xDNN classification without providing cases.')
            classification_results = xDNN.classify(cases)

            cleaned_dict = {
                "EstLabs": np.concatenate(classification_results["EstLabs"]).ravel().astype(int),
                "Scores": classification_results["Scores"]
            }

            return cleaned_dict
