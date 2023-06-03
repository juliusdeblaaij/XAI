from collections import OrderedDict

import numpy as np
from sklearn.model_selection import train_test_split

from DataEvent import DataEvent
from algorithms.dox_algorithm_adapter import DoxAlgorithmAdapter
from indicators.CompositeIndicator import CompositeIndicator
from EventsBroadcaster import broadcast_data

workers = OrderedDict()


class MeaningfulnessIndicator(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"practitioner_aspects": [], "explanations": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        explanandum_aspects = kwargs.get("practitioner_aspects")
        all_explanations = kwargs.get("explanations")

        self.local_data()["workers"] = OrderedDict()

        split_explanations = np.array_split(np.asarray(all_explanations), 4)
        for explanations in split_explanations:
            kwargs = {"explanandum_aspects": explanandum_aspects,
                      "cases": explanations}

            dox_algo = DoxAlgorithmAdapter()
            dox_algo.run(callback=self.on_dox_calculated, **kwargs)
            workers[str(dox_algo.pid)] = None


    def on_dox_calculated(self, kwargs):
        faithfulness_scores = kwargs.get("average_dox_scores")
        pid = kwargs.get("pid")

        for worker in workers:
            if str(pid) == worker:
                workers[str(pid)] = faithfulness_scores

        if not None in workers:

            all_meaningfulness_scores = []

            for key, value in workers.items():
                all_meaningfulness_scores.extend(value)

            broadcast_data({"meaningfulness_scores": all_meaningfulness_scores})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
