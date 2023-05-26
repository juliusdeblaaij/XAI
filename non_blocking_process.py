from multiprocessing import Process, Queue, current_process
from threading import Thread
from time import sleep

class NonBlockingProcess():


    def __init__(self, data, callback_queue: Queue, callback):
        p = Process(target=self.main_process, args=(data, callback_queue, self.callback_thread, callback))
        p.start()

        t = Thread(target=self.callback_caller, args=(callback_queue, ))
        t.start()

    def main_process(self, data, callback_queue, callback_thread, callback):

        work = self.do_work(data)
        callback_queue.put((callback_thread, work, callback))

    def do_work(self, data):
        sleep(1)
        return data

    def callback_thread(self, data, callback):
        callback(data)




    def callback_caller(self, cb_queue):
        for func, *args in iter(cb_queue.get, None):  # pass None to exit thread
            func(*args)
