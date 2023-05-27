import multiprocessing
from abc import abstractmethod, ABC
from multiprocessing import Queue


class AlgorithmAdapter(ABC):
    _callback_queue = Queue()
    _external_callback = None

    def _internal_callback(self, result):
        self._callback_queue.put(None)
        self._external_callback(result)

    @abstractmethod
    def run(self, callback, daemon=False, **kwargs):
        pass
