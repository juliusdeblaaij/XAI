from DataEvent import DataEvent
from EventsBroadcaster import broadcast_data
from algorithms.audience_acceptability_algorithm_adapter import AudienceAcceptabilityAlgorithmAdapter
from indicators.CompositeIndicator import CompositeIndicator


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
        expert_aspects_amount = len(kwargs.get("expert_aspects"))
        explanations = kwargs.get("explanations")

        kwargs = {
            "outsider_aspects_amount": outsider_aspects_amount,
            "practitioner_aspects_amount": practitioner_aspects_amount,
            "expert_aspects_amount": expert_aspects_amount,
            "explanations": explanations
        }

        audience_acceptability_algo = AudienceAcceptabilityAlgorithmAdapter()
        audience_acceptability_algo.run(callback=self.on_audience_acceptability_calculated, **kwargs)

    def on_audience_acceptability_calculated(self, acceptability_scores):
        broadcast_data({"acceptability_scores": acceptability_scores})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
