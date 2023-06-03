import re
from multiprocessing import current_process

from fuzzy_expert.inference import DecompositionalInference
from fuzzy_expert.rule import FuzzyRule
from fuzzy_expert.variable import FuzzyVariable

from dataset_cleaner import filter_allowed_words
from dox_utils import get_details_from, question_template_list
from myutils import extract_knowledge_graph
from non_blocking_process import AbstractNonBlockingProcess


class AudienceAcceptabilityAlgorithm(AbstractNonBlockingProcess):

    def _do_work(self, outsider_aspects_amount=None, practitioner_aspects_amount=None, expert_aspects_amount=None, explanations=None):
        if outsider_aspects_amount is None:
            raise ValueError("Attempted to calculate audience acceptability without providing 'outsider_aspects_amount'.")
        if practitioner_aspects_amount is None:
            raise ValueError("Attempted to calculate audience acceptability without providing 'practitioner_aspects_amount'.")
        if expert_aspects_amount is None:
            raise ValueError("Attempted to calculate audience acceptability without providing 'expert_aspects_amount'.")
        if explanations is None:
            raise ValueError("Attempted to calculate audience acceptability without providing 'explanations'.")

        outsider_aspects_amount = outsider_aspects_amount * len(question_template_list)
        practitioner_aspects_amount = practitioner_aspects_amount * len(question_template_list)
        expert_aspects_amount = expert_aspects_amount * len(question_template_list)

        details_amount_universal_limit = expert_aspects_amount * 1.5

        if outsider_aspects_amount > details_amount_universal_limit:
            outsider_aspects_amount = details_amount_universal_limit
        if practitioner_aspects_amount > details_amount_universal_limit:
            practitioner_aspects_amount = details_amount_universal_limit
        if expert_aspects_amount > details_amount_universal_limit:
            expert_aspects_amount = details_amount_universal_limit


        scaled_outsider_details_amount = (outsider_aspects_amount / details_amount_universal_limit) * 10
        scaled_practitioner_details_amount = (practitioner_aspects_amount / details_amount_universal_limit) * 10
        scaled_experts_details_amount = (expert_aspects_amount / details_amount_universal_limit) * 10

        details_variable = FuzzyVariable(
            universe_range=(0, details_amount_universal_limit),
            terms={
                "SPARSE": ('zmf', scaled_outsider_details_amount, scaled_outsider_details_amount + 2),
                "PERTINENT": ("gbellmf", scaled_practitioner_details_amount, scaled_practitioner_details_amount / 2, 4),
                "EXTENSIVE": ("smf", scaled_experts_details_amount, scaled_experts_details_amount + 2)
            }
        )

        """acceptability_variable = FuzzyVariable(
            universe_range=(0, details_amount_universal_limit),
            terms={
                "TOTALLY_UNACCEPTABLE": ('zmf', 0.1 * details_amount_universal_limit, 0.15 * details_amount_universal_limit),
                "UNACCEPTABLE": ('trapmf', 0.1 * details_amount_universal_limit, 0.15 * details_amount_universal_limit, 0.25 * details_amount_universal_limit, 0.3 * details_amount_universal_limit),
                "SLIGHTLY_UNACCEPTABLE": ('trapmf', 0.25 * details_amount_universal_limit, 0.3 * details_amount_universal_limit, 0.4 * details_amount_universal_limit, 0.45 * details_amount_universal_limit),
                "NEUTRAL": ('trapmf', 0.4 * details_amount_universal_limit, 0.45 * details_amount_universal_limit, 0.55 * details_amount_universal_limit, 0.6 * details_amount_universal_limit),
                "SLIGHTLY_ACCEPTABLE": ('trapmf', 0.55 * details_amount_universal_limit, 0.6 * details_amount_universal_limit, 0.7 * details_amount_universal_limit, 0.75 * details_amount_universal_limit),
                "ACCEPTABLE": ('trapmf', 0.7 * details_amount_universal_limit, 0.75 * details_amount_universal_limit, 0.85 * details_amount_universal_limit, 0.9 * details_amount_universal_limit),
                "TOTALLY_ACCEPTABLE": ('sigmf', 0.85 * details_amount_universal_limit, 0.9 * details_amount_universal_limit)
            })"""

        acceptability_variable = FuzzyVariable(
            universe_range=(0, 10),
            terms={
                "TOTALLY_UNACCEPTABLE": ('zmf', 1, 1.5),
                "UNACCEPTABLE": ('trapmf', 1, 1.5, 2.5, 3),
                "SLIGHTLY_UNACCEPTABLE": ('trapmf', 2.5, 3, 4, 4.5),
                "NEUTRAL": ('trapmf', 4, 4.5, 5.5, 6),
                "SLIGHTLY_ACCEPTABLE": ('trapmf', 5.5, 6, 7, 7.5),
                "ACCEPTABLE": ('trapmf', 7, 7.5, 8.5, 9),
                "TOTALLY_ACCEPTABLE": ('sigmf', 8.5, 9)
            })

        fuzzy_variables = {
            "details": details_variable,
            "acceptability": acceptability_variable,
        }

        practitioner_rules = [
            FuzzyRule(
                premise=[
                    ("details", "SPARSE"),
                ],
                consequence=[("acceptability", "UNACCEPTABLE")],
            ),
            FuzzyRule(
                premise=[
                    ("details", "PERTINENT"),
                ],
                consequence=[("acceptability", "TOTALLY_ACCEPTABLE")],
            ),
            FuzzyRule(
                premise=[
                    ("details", "EXTENSIVE"),
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
            knowledge_graph = extract_knowledge_graph(text=filter_allowed_words(explanation))

            explanation_details = get_details_from(knowledge_graph)
            explanation_details_amount = len(explanation_details)

            if explanation_details_amount > details_amount_universal_limit:
                explanation_details_amount = details_amount_universal_limit

            details_scaled = (explanation_details_amount / details_amount_universal_limit) * 10

            model(
                variables=fuzzy_variables,
                rules=practitioner_rules,
                details=details_scaled
            )

            model_results = model.defuzzificated_infered_memberships

            acceptability_scores.append(model_results.get("acceptability"))

            if i > 0:
                if 100 % i == 0:
                    print(f"Calculated acceptability scores {i} / {len(explanations)}")

        result = {
            "acceptability_scores": acceptability_scores,
            "pid": current_process().pid
        }

        return result
