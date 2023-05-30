from DataEvent import DataEvent
from EventsBroadcaster import broadcast_data
from algorithms.audience_acceptability_algorithm_adapter import AudienceAcceptabilityAlgorithmAdapter
from indicators.CompositeIndicator import CompositeIndicator
from fuzzy_expert.inference import DecompositionalInference
from fuzzy_expert.variable import FuzzyVariable
from fuzzy_expert.rule import FuzzyRule
from myutils import *
from get_aspects import get_aspects


class AudienceAcceptabilityIndicator(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"outsider_aspects": [], "practitioner_aspects": [], "expert_aspects": [], "explanations": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        outsider_aspects_amount = len(kwargs.get("outsider_aspects"))
        practitioner_aspects_amount = len(kwargs.get("practitioner_aspects"))
        experts_aspects_amount = len(kwargs.get("expert_aspects"))
        explanations = kwargs.get("explanations")

        kwargs = {
            "outsider_aspects_amount": outsider_aspects_amount,
            "practitioner_aspects_amount": practitioner_aspects_amount,
            "experts_aspects_amount": experts_aspects_amount,
            "explanations": explanations
        }

        audience_acceptability_algo = AudienceAcceptabilityAlgorithmAdapter()
        audience_acceptability_algo.run(callback=self.on_audience_acceptability_calculated, **kwargs)

    def on_audience_acceptability_calculated(self, data):
        outsider_acceptability_scores = data.get("outsider_acceptability_scores")
        practitioner_acceptability_scores = data.get("practitioner_acceptability_scores")
        expert_acceptability_scores = data.get("expert_acceptability_scores")

        broadcast_data({"outsider_acceptability_scores": outsider_acceptability_scores,
                        "practitioner_acceptability_scores": practitioner_acceptability_scores,
                        "expert_acceptability_scores": expert_acceptability_scores})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())