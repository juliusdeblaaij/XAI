from abc import abstractmethod, ABC
from multiprocessing import Process, Queue, current_process
from threading import Thread
from time import sleep


class AbstractNonBlockingProcess(ABC):

    def __init__(self, callback_queue: Queue, callback, daemon=False, **kwargs):
        """
        Creates a non-blocking child process with callback.

        :param data: Any object that will be used as input by the process.
        :param callback_queue: A Queue object that used to get data from the process,
        close the process by calling Queue().put(None)
        :param callback: A callback function that will be fed an object which is the process' output.
        :param daemon: Whether the process is a daemon.
        """

        p = Process(target=self._main_process, daemon=daemon,
                    args=(callback_queue, self._callback_thread, callback), kwargs=kwargs)
        p.start()

        t = Thread(target=self._callback_caller, args=(callback_queue,))
        t.start()


    def _main_process(self, callback_queue, callback_thread, callback, **kwargs):
        work = self._do_work(**kwargs)
        callback_queue.put((callback_thread, work, callback))

    def _callback_thread(self, data, callback):
        callback(data)

    def _callback_caller(self, cb_queue):
        for func, *args in iter(cb_queue.get, None):  # pass None to exit thread
            func(*args)

    @abstractmethod
    def _do_work(self, **kwargs):
        pass
