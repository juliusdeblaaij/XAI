from multiprocessing import Queue

from algorithms.algorithm_adapter import AlgorithmAdapter
from algorithms.xdnn_algorithm import xDNNAlgorithm


class xDNNAlgorithmAdapter(AlgorithmAdapter):

    def run(self, callback, daemon=False, **kwargs):
        self.external_callback = callback
        callback_queue = Queue()
        algo = xDNNAlgorithm(callback_queue=callback_queue,
                            daemon=daemon,
                            callback=self._internal_callback,
                            **kwargs)
        self.pid = algo.pid
        self.callback_queues[str(self.pid)] = callback_queue
