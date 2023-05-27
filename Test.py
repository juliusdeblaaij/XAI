import time
from multiprocessing import freeze_support

from EventsBroadcaster import broadcast_data, broadcast, subscribe
from indicators.HippStringPrinter import HippStringPrinter
from indicators.IntegerCompositeIndicator import IntegerCompositeIndicator
from indicators.faithfulness_indicator import FaithfulnessIndicator

faithfulness_indicator = FaithfulnessIndicator()


if __name__ == "__main__":
    freeze_support()

    """broadcast_data({"cases": ["as a performance tester id like to investigate why theres high cpu startup time for both "
                             "admin and container servers perhaps profiling would assist isolating the bottlenecks scope"
                             " identify the bottlenecks document reasons list proscons"], "mode": "Learning"})"""