from DataEvent import DataEvent
from EventsBroadcaster import broadcast_data

from doxpy.models.knowledge_extraction.knowledge_graph_extractor import KnowledgeGraphExtractor
from indicators.CompositeIndicator import CompositeIndicator
import confuse
from get_aspects import get_aspects
from myutils import *

class AudienceAspectsExtractor(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"outsider_questions": "", "practitioner_questions": "", "expert_questions": ""}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        outsider_questions = kwargs.get("outsider_questions")
        practitioner_questions = kwargs.get("practitioner_questions")
        expert_questions = kwargs.get("expert_questions")

        outsider_knowledge_graph = extract_knowledge_graph(text=outsider_questions)

        outsider_aspects = get_aspects(outsider_knowledge_graph)

        practitioner_knowledge_graph = extract_knowledge_graph(text=practitioner_questions)

        practitioner_aspects = get_aspects(practitioner_knowledge_graph)

        expert_knowledge_graph = extract_knowledge_graph(text=expert_questions)

        expert_aspects = get_aspects(expert_knowledge_graph)

        broadcast_data({"outsider_aspects": outsider_aspects, "practitioner_aspects": practitioner_aspects, "expert_aspects": expert_aspects})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
