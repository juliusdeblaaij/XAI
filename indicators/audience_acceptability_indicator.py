from DataEvent import DataEvent
from EventsBroadcaster import broadcast_data
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
        return {"outsider_aspects": [], "practicioner_aspects": [], "expert_aspects": [], "explanations": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        outsider_aspects_amount = len(kwargs.get("outsider_aspects"))
        practitioner_aspects_amount = len(kwargs.get("practicioner_aspects"))
        experts_aspects_amount = len(kwargs.get("expert_aspects"))
        explanations = kwargs.get("explanations")

        aspects_variable = FuzzyVariable(
            universe_range=(0, 20),
            terms={
                "SPARSE": ('zmf', outsider_aspects_amount, outsider_aspects_amount + 2),
                "PERTINENT": ("gbellmf", practitioner_aspects_amount, practitioner_aspects_amount / 2, 4),
                "EXTENSIVE": ("smf", experts_aspects_amount, experts_aspects_amount + 2)
            }
        )

        acceptability_variable = FuzzyVariable(
            universe_range=(0, 10),
            terms={
                "TOTALLY_UNACCEPTABLE": ('trapmf', 0, 0, 1, 1.5),
                "UNACCEPTABLE": ('trapmf', 1, 1.5, 2.5, 3),
                "SLIGHTLY_UNACCEPTABLE": ('trapmf', 2.5, 3, 4, 4.5),
                "NEUTRAL": ('trapmf', 4, 4.5, 5.5, 6),
                "SLIGHTLY_ACCEPTABLE": ('trapmf', 5.5, 6, 7, 7.5),
                "ACCEPTABLE": ('trapmf', 7, 7.5, 8.5, 9),
                "TOTALLY_ACCEPTABLE": ('trapmf', 8.5, 9, 10, 10)
            })

        fuzzy_variables = {
            "aspects": aspects_variable,
            "outsider_acceptability": acceptability_variable,
            "practitioner_acceptability": acceptability_variable,
            "expert_acceptability": acceptability_variable,
        }

        outsider_rules = [
            FuzzyRule(
                premise=[
                    ("aspects", "SPARSE"),
                ],
                consequence=[("outsider_acceptability", "TOTALLY_ACCEPTABLE")],
            ),
            FuzzyRule(
                premise=[
                    ("aspects", "PERTINENT"),
                ],
                consequence=[("outsider_acceptability", "SLIGHTLY_UNACCEPTABLE")],
            ),
            FuzzyRule(
                premise=[
                    ("aspects", "EXTENSIVE"),
                ],
                consequence=[("outsider_acceptability", "TOTALLY_UNACCEPTABLE")],
            ),
        ]

        practitioner_rules = [
            FuzzyRule(
                premise=[
                    ("aspects", "SPARSE"),
                ],
                consequence=[("practitioner_acceptability", "UNACCEPTABLE")],
            ),
            FuzzyRule(
                premise=[
                    ("aspects", "PERTINENT"),
                ],
                consequence=[("practitioner_acceptability", "TOTALLY_ACCEPTABLE")],
            ),
            FuzzyRule(
                premise=[
                    ("aspects", "EXTENSIVE"),
                ],
                consequence=[("practitioner_acceptability", "UNACCEPTABLE")],
            ),
        ]

        expert_rules = [
            FuzzyRule(
                premise=[
                    ("aspects", "SPARSE"),
                ],
                consequence=[("expert_acceptability", "TOTALLY_UNACCEPTABLE")],
            ),
            FuzzyRule(
                premise=[
                    ("aspects", "PERTINENT"),
                ],
                consequence=[("expert_acceptability", "NEUTRAL")],
            ),
            FuzzyRule(
                premise=[
                    ("aspects", "EXTENSIVE"),
                ],
                consequence=[("expert_acceptability", "TOTALLY_ACCEPTABLE")],
            ),
        ]

        fuzzy_rules = []
        fuzzy_rules.extend(outsider_rules)
        fuzzy_rules.extend(practitioner_rules)
        fuzzy_rules.extend(expert_rules)

        model = DecompositionalInference(
            and_operator="min",
            or_operator="max",
            implication_operator="Rc",
            composition_operator="max-min",
            production_link="max",
            defuzzification_operator="cog",
        )

        outsider_acceptability_scores = []
        practicioner_acceptability_scores = []
        expert_acceptability_scores = []

        for explanation in explanations:
            expert_knowledge_graph = extract_knowledge_graph(text=explanation)

            explanation_aspects = get_aspects(expert_knowledge_graph)
            explanation_aspects_amount = len(explanation_aspects)

            model(
                variables=fuzzy_variables,
                rules=fuzzy_rules,
                aspects=explanation_aspects_amount
            )

            acceptability_scores = model.defuzzificated_infered_memberships

            outsider_acceptability_scores.append(acceptability_scores.get("outsider_acceptability"))
            practicioner_acceptability_scores.append(acceptability_scores.get("practitioner_acceptability"))
            expert_acceptability_scores.append(acceptability_scores.get("expert_acceptability"))

        broadcast_data({"outsider_acceptability_scores": outsider_acceptability_scores,
                        "practicioner_acceptability_scores": practicioner_acceptability_scores,
                        "expert_acceptability_scores": expert_acceptability_scores})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
