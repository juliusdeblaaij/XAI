from multiprocessing import Queue

from algorithms.algorithm_adapter import AlgorithmAdapter
from algorithms.dox_algorithm import DoXAlgorithm


class DoxAlgorithmAdapter(AlgorithmAdapter):

    _callback_queue = Queue()
    _external_callback = None

    def run(self, callback, daemon=False, **kwargs):
        self._external_callback = callback
        DoXAlgorithm(callback_queue=self._callback_queue,
                              daemon=daemon,
                              callback=self._internal_callback,
                              **kwargs)
