from multiprocessing import current_process, Queue
from time import sleep

from EventsBroadcaster import broadcast_data
from algorithms.faithfulness_algorithm import FaithfulnessAlgorithm
from indicators.faithfulness_indicator import FaithfulnessIndicator

callback_queue = Queue()

def some_func(data):
    print(f"some_func {data}")
    callback_queue.put(None)

if __name__ == "__main__":
    print(f"main pid: [{current_process().pid}]")

    faithfulness_indicator = FaithfulnessIndicator()

    broadcast_data({"mode": "Learning", "cases": [
        "as a performance tester id like to investigate why theres high cpu startup time for both admin and container "
        "servers perhaps profiling would assist isolating the bottlenecks scope identify the bottlenecks document "
        "reasons list proscons"],
        "predicted_classes": [5],
        "actual_classes": [5]
    })