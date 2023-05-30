from DataEvent import DataEvent
from EventsBroadcaster import broadcast_data
from algorithms.audience_acceptability_algorithm_adapter import AudienceAcceptabilityAlgorithmAdapter
from indicators.CompositeIndicator import CompositeIndicator
from fuzzy_expert.inference import DecompositionalInference
from fuzzy_expert.variable import FuzzyVariable
from fuzzy_expert.rule import FuzzyRule
from myutils import *
from get_aspects import get_aspects


class ExplanationAccuracyIndicator(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"outsider_acceptability_scores": [], "practitioner_acceptability_scores": [], "expert_acceptability_scores": [], "faithfulness_scores": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        outsider_acceptability_scores = kwargs.get("outsider_acceptability_scores")
        practitioner_acceptability_scores = kwargs.get("practitioner_acceptability_scores")
        expert_acceptability_scores = kwargs.get("expert_acceptability_scores")
        faithfulness_scores = kwargs.get("explanations")

        for i, faithfulness_score in faithfulness_scores:
            practitioner_acceptability_score = practitioner_acceptability_scores[i]



    def on_audience_acceptability_calculated(self, data):
        outsider_acceptability_scores = data.get("outsider_acceptability_scores")
        practitioner_acceptability_scores = data.get("practitioner_acceptability_scores")
        expert_acceptability_scores = data.get("expert_acceptability_scores")

        broadcast_data({"outsider_acceptability_scores": outsider_acceptability_scores,
                        "practitioner_acceptability_scores": practitioner_acceptability_scores,
                        "expert_acceptability_scores": expert_acceptability_scores})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
