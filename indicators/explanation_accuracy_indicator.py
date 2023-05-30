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

        faithfulness_variable = FuzzyVariable(
            universe_range=(0, 1),
            terms={
                "LOW": ('trapmf', 0, 0, 0.7, 0.75),
                "MEDIUM": ("trapmf", 0.7, 0.75, 0.85, 0.9),
                "HIGH": ("trapmf", 0.85, 0.9, 1.0, 1.0)
            }
        )

        acceptability_variable = FuzzyVariable(
            universe_range=(0, 10),
            terms={
                "LOW": ("trapmf", 0, 0, 4, 4.5),
                "MEDIUM": ("trapmf", 4.0, 4.5, 7, 7.5),
                "HIGH": ("trapmf", 7, 7.5, 10, 10),
        })

        explanation_accuracy_variable = FuzzyVariable(
            universe_range=(0, 10),
            terms={
                "FAIL": ("trapmf", 0, 0, 0, 5),
                "PASS": ("trapmf", 5, 10, 10, 10),
            })

        fuzzy_variables = {
            "faithfulness": faithfulness_variable,
            # "outsider_acceptability": acceptability_variable,
            "practitioner_acceptability": acceptability_variable,
            # "expert_acceptability": acceptability_variable,
            "explanation_accuracy": explanation_accuracy_variable,
        }

        practitioner_rules = [
            FuzzyRule(
                premise=[
                    ("faithfulness", "HIGH"),
                    ("AND", "practitioner_acceptability", "HIGH"),
                ],
                consequence=[("explanation_accuracy", "PASS")],
            ),
            FuzzyRule(
                premise=[
                    ("faithfulness", "HIGH"),
                    ("AND", "practitioner_acceptability", "MEDIUM"),
                ],
                consequence=[("explanation_accuracy", "PASS")],
            ),
            FuzzyRule(
                premise=[
                    ("faithfulness", "MEDIUM"),
                    ("AND", "practitioner_acceptability", "HIGH"),
                ],
                consequence=[("explanation_accuracy", "PASS")],
            ),
            FuzzyRule(
                premise=[
                    ("faithfulness", "MEDIUM"),
                    ("AND", "practitioner_acceptability", "MEDIUM"),
                ],
                consequence=[("explanation_accuracy", "FAIL")],
            ),
            FuzzyRule(
                premise=[
                    ("faithfulness", "LOW"),
                    ("OR", "practitioner_acceptability", "LOW"),
                ],
                consequence=[("explanation_accuracy", "FAIL")],
            ),
        ]

        model = DecompositionalInference(
            and_operator="min",
            or_operator="max",
            implication_operator="Rc",
            composition_operator="max-min",
            production_link="max",
            defuzzification_operator="cog",
        )

        explanation_accuracy_decisions = []

        for i, faithfulness_score in faithfulness_scores:
            practitioner_acceptability_score = practitioner_acceptability_scores[i]

            model(
                variables=fuzzy_variables,
                rules=practitioner_rules,
                faithfulness=faithfulness_score,
                practitioner_acceptability=practitioner_acceptability_score
            )

            explanation_accuracy = model.defuzzificated_infered_memberships
            if explanation_accuracy >= 5:
                print("PASS")
                explanation_accuracy_decisions.append(1)
            else:
                print("FAIL")
                explanation_accuracy_decisions.append(0)

        broadcast_data({"explanation_accuracy_decisions": explanation_accuracy_decisions})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
