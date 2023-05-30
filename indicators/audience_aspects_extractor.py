from DataEvent import DataEvent
from EventsBroadcaster import broadcast_data

from doxpy.models.knowledge_extraction.knowledge_graph_extractor import KnowledgeGraphExtractor
from indicators.CompositeIndicator import CompositeIndicator
import confuse
from get_aspects import get_aspects


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
        return {"outsider_questions": "", "practicioner_questions": "", "expert_questions": ""}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        config = confuse.Configuration('XAI', __name__)
        graph_builder_options = config["graph_builder_options"].get(dict)
        graph_cleaning_options = config["graph_cleaning_options"].get(dict)
        graph_extraction_options = config["graph_extraction_options"].get(dict)

        outsider_questions = kwargs.get("outsider_questions")
        practicioner_questions = kwargs.get("practicioner_questions")
        expert_questions = kwargs.get("expert_questions")

        outsider_knowledge_graph = self.extract_knowledge_graph(text=outsider_questions,
                                                                graph_builder_options=graph_builder_options,
                                                                graph_cleaning_options=graph_cleaning_options,
                                                                graph_extraction_options=graph_extraction_options)

        outsider_aspects = get_aspects(outsider_knowledge_graph)

        practicioner_knowledge_graph = self.extract_knowledge_graph(text=practicioner_questions,
                                                                graph_builder_options=graph_builder_options,
                                                                graph_cleaning_options=graph_cleaning_options,
                                                                graph_extraction_options=graph_extraction_options)

        practicioner_aspects = get_aspects(practicioner_knowledge_graph)

        expert_knowledge_graph = self.extract_knowledge_graph(text=expert_questions,
                                                                graph_builder_options=graph_builder_options,
                                                                graph_cleaning_options=graph_cleaning_options,
                                                                graph_extraction_options=graph_extraction_options)

        expert_aspects = get_aspects(expert_knowledge_graph)

        broadcast_data({"outsider_aspects": outsider_aspects, "practicioner_aspects": practicioner_aspects, "expert_aspects": expert_aspects})

    def extract_knowledge_graph(self, text: str, graph_builder_options: dict, graph_cleaning_options: dict, graph_extraction_options: dict):
        knowledge_graph = KnowledgeGraphExtractor(graph_builder_options).set_content_list(
            content_list=[text],
            **graph_cleaning_options).build(
            **graph_extraction_options)

        return knowledge_graph

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
