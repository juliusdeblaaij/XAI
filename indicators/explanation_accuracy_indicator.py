from DataEvent import DataEvent
from EventsBroadcaster import broadcast_data
from indicators.CompositeIndicator import CompositeIndicator
from fuzzy_expert.inference import DecompositionalInference
from fuzzy_expert.variable import FuzzyVariable
from fuzzy_expert.rule import FuzzyRule

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
        return {"acceptability_scores": [], "faithfulness_scores": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        acceptability_scores = kwargs.get("acceptability_scores")
        faithfulness_scores = kwargs.get("faithfulness_scores")

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
            "acceptability": acceptability_variable,
            # "expert_acceptability": acceptability_variable,
            "explanation_accuracy": explanation_accuracy_variable,
        }

        practitioner_rules = [
            FuzzyRule(
                premise=[
                    ("faithfulness", "HIGH"),
                    ("AND", "acceptability", "HIGH"),
                ],
                consequence=[("explanation_accuracy", "PASS")],
            ),
            FuzzyRule(
                premise=[
                    ("faithfulness", "HIGH"),
                    ("AND", "acceptability", "MEDIUM"),
                ],
                consequence=[("explanation_accuracy", "PASS")],
            ),
            FuzzyRule(
                premise=[
                    ("faithfulness", "MEDIUM"),
                    ("AND", "acceptability", "HIGH"),
                ],
                consequence=[("explanation_accuracy", "PASS")],
            ),
            FuzzyRule(
                premise=[
                    ("faithfulness", "MEDIUM"),
                    ("AND", "acceptability", "MEDIUM"),
                ],
                consequence=[("explanation_accuracy", "FAIL")],
            ),
            FuzzyRule(
                premise=[
                    ("faithfulness", "LOW"),
                    ("OR", "acceptability", "LOW"),
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

        explanation_accuracy_scores = []
        explanation_accuracy_decisions = []

        for i, faithfulness_score in enumerate(faithfulness_scores):
            acceptability_score = acceptability_scores[i]

            model(
                variables=fuzzy_variables,
                rules=practitioner_rules,
                faithfulness=faithfulness_score,
                acceptability=acceptability_score
            )

            explanation_accuracy_score = model.defuzzificated_infered_memberships.get("explanation_accuracy")
            if explanation_accuracy_score >= 5:
                explanation_accuracy_decisions.append(1)
            else:
                explanation_accuracy_decisions.append(0)

            explanation_accuracy_scores.append(explanation_accuracy_score)

        broadcast_data({"explanation_accuracy_decisions": explanation_accuracy_decisions,
                        "explanation_accuracy_scores": explanation_accuracy_scores,
                        "acceptability_scores": acceptability_scores})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
