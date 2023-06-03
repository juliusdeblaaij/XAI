import multiprocessing
from abc import abstractmethod, ABC
from collections import OrderedDict
from multiprocessing import Queue
from typing import Callable


class AlgorithmAdapter(ABC):

    callback_queues = OrderedDict()
    pid = 0
    external_callback: Callable

    def _internal_callback(self, kwargs):
        self.callback_queues[str(kwargs.get("pid"))].put(None)
        self.external_callback(kwargs)

    @abstractmethod
    def run(self, callback, daemon=False, **kwargs):
        pass
