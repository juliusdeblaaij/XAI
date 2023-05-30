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
        practitioner_acceptability_scores = []
        expert_acceptability_scores = []

        for i, explanation in enumerate(explanations):
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
            practitioner_acceptability_scores.append(acceptability_scores.get("practitioner_acceptability"))
            expert_acceptability_scores.append(acceptability_scores.get("expert_acceptability"))

            if i > 0:
                if 100 % i == 0:
                    print(f"Calculated acceptability scores {i} / {len(explanations)}")

        return {"outsider_acceptability_scores": outsider_acceptability_scores,
                "practitioner_acceptability_scores": practitioner_acceptability_scores,
                "expert_acceptability_scores": expert_acceptability_scores}
