from multiprocessing import Queue

from algorithms.algorithm_adapter import AlgorithmAdapter
from algorithms.faithfulness_algorithm import FaithfulnessAlgorithm


class FaithfulnessAlgorithmAdapter(AlgorithmAdapter):

    _callback_queue = Queue()
    _external_callback = None

    def run(self, callback, daemon=False, **kwargs):
        self._external_callback = callback
        FaithfulnessAlgorithm(callback_queue=self._callback_queue,
                              daemon=daemon,
                              callback=self._internal_callback,
                              **kwargs)
