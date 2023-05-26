from multiprocessing import Process, Queue, current_process
from threading import Thread
from time import sleep

class NonBlockingProcessor():

    process = None
    thread = None

    def __init__(self, data, callback):
        callback_queue = Queue()

        p = Process(target=self.main_process, args=(data, callback_queue, self.callback_thread))
        p.start()

        t = Thread(target=self.callback_caller, args=(callback_queue, ))
        t.start()

    def main_process(self, data, callback_queue, cb):

        work = self.do_work(data)
        callback_queue.put((cb, work))

    def do_work(self, data):
        sleep(1)
        return data

    def callback_thread(self, data):
        print(f"data from cb_thread in process [{current_process().pid}]: {data}")



    def callback_caller(self, cb_queue):
        for func, *args in iter(cb_queue.get, None):  # pass None to exit thread
            func(*args)