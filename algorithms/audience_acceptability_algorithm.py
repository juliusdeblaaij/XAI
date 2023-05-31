import re

from fuzzy_expert.inference import DecompositionalInference
from fuzzy_expert.rule import FuzzyRule
from fuzzy_expert.variable import FuzzyVariable

from get_aspects import get_aspects
from myutils import extract_knowledge_graph
from non_blocking_process import AbstractNonBlockingProcess


class AudienceAcceptabilityAlgorithm(AbstractNonBlockingProcess):

    def _do_work(self, outsider_aspects_amount=None, practitioner_aspects_amount=None, experts_aspects_amount=None, explanations=None):
        if outsider_aspects_amount is None:
            raise ValueError("Attempted to calculate audience acceptability without providing 'outsider_aspects_amount'.")
        if practitioner_aspects_amount is None:
            raise ValueError("Attempted to calculate audience acceptability without providing 'practitioner_aspects_amount'.")
        if experts_aspects_amount is None:
            raise ValueError("Attempted to calculate audience acceptability without providing 'experts_aspects_amount'.")
        if explanations is None:
            raise ValueError("Attempted to calculate audience acceptability without providing 'explanations'.")

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
            "acceptability": acceptability_variable,
        }

        practitioner_rules = [
            FuzzyRule(
                premise=[
                    ("aspects", "SPARSE"),
                ],
                consequence=[("acceptability", "UNACCEPTABLE")],
            ),
            FuzzyRule(
                premise=[
                    ("aspects", "PERTINENT"),
                ],
                consequence=[("acceptability", "TOTALLY_ACCEPTABLE")],
            ),
            FuzzyRule(
                premise=[
                    ("aspects", "EXTENSIVE"),
                ],
                consequence=[("acceptability", "UNACCEPTABLE")],
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

        acceptability_scores = []

        for i, explanation in enumerate(explanations):
            knowledge_graph = extract_knowledge_graph(text=explanation)

            explanation_aspects = get_aspects(knowledge_graph)
            explanation_aspects_amount = len(explanation_aspects)

            model(
                variables=fuzzy_variables,
                rules=practitioner_rules,
                aspects=explanation_aspects_amount
            )

            model_results = model.defuzzificated_infered_memberships

            acceptability_scores.append(model_results.get("acceptability"))

            if i > 0:
                if 100 % i == 0:
                    print(f"Calculated acceptability scores {i} / {len(explanations)}")

        return acceptability_scores
