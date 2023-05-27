from non_blocking_process import AbstractNonBlockingProcess
import numpy as np

from run_xdnn import RunxDNN
from doc2vec import doc2vec
from myutils import *


class xDNNAlgorithm(AbstractNonBlockingProcess):

    def _do_work(self, cases=None, mode=None):
        if cases:
            print(cases)
        else:
            return None
        if mode:
            print(mode)
        else:
            return None

        xDNN = RunxDNN()
        training_results = xDNN.train()

        return training_results
