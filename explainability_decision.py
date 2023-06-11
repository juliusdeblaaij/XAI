import matplotlib.pyplot as plt
import numpy as np

from fuzzy_expert.variable import FuzzyVariable

def get_keys_with_highest_value(dictionary):
    # Find the maximum value in the dictionary
    max_value = max(dictionary.values())

    # Create a list to store the keys with the highest value
    highest_value_keys = []

    # Iterate over the dictionary and check if the value is equal to the maximum value
    for key, value in dictionary.items():
        if value == max_value:
            highest_value_keys.append(key)

    return highest_value_keys

variables = {
    "meaningfulness": FuzzyVariable(
        universe_range=(0.01, 0.11),
        terms={
            "High": ('smf', 0.02, 0.04),
            "Neutral": ('trimf', 0.01, 0.02, 0.03),
            "Low": ('zmf', 0.01, 0.02),
        },
    ),
    "acceptability": FuzzyVariable(
        universe_range=(2, 9),
        terms={
            "Acceptable": ('smf', 5.5, 9),
            "Unacceptable": ('zmf', 3, 6),
        },
    ),
    "faithfulness": FuzzyVariable(
        universe_range=(0.0, 1.0),
        terms={
            "Faithful": ("smf", 0.45, 1.0),
            "Unfaithful": ("zmf", 0, 0.55),
        },
    ),
    "okd": FuzzyVariable(
            universe_range=(0.0, 1.0),
            terms={
                "Adhering": ("smf", 0.45, 1.0),
                "Violating": ("zmf", 0, 0.55),
            },
        ),
    "similarity": FuzzyVariable(
            universe_range=(0.0, 1.0),
            terms={
                "Adhering": ("smf", 0.45, 1.0),
                "Violating": ("zmf", 0, 0.55),
            },
        ),
    "decision": FuzzyVariable(
        universe_range=(0.0, 1.0),
        terms={
            "EXPLAINABLE": ("smf", 0.45, 1.0),
            "NOT_EXPLAINABLE": ("zmf", 0, 0.55),
        },
    ),
}

plt.clf()
variables["meaningfulness"].plot()
plt.savefig("./Figures/meaningfulness_membership.pdf")
plt.clf()
variables["acceptability"].plot()
plt.savefig("./Figures/acceptability_membership.pdf")
plt.clf()
variables["faithfulness"].plot()
plt.savefig("./Figures/faithfulness_membership.pdf")
plt.clf()
variables["okd"].plot()
plt.savefig("./Figures/okd_membership.pdf")
plt.clf()
variables["similarity"].plot()
plt.savefig("./Figures/similarity_membership.pdf")
plt.clf()
variables["decision"].plot()
plt.savefig("./Figures/decision_membership.pdf")

from fuzzy_expert.rule import FuzzyRule

rules = [
    FuzzyRule(
        premise=[
            ("meaningfulness", "High"),
            ("AND", "acceptability", "Acceptable"),
            ("AND", "faithfulness", "Faithful"),
            ("AND", "okd", "Adhering"),
            ("AND", "similarity", "Adhering"),
        ],
        consequence=[("decision", "EXPLAINABLE")],
    ),
    FuzzyRule(
        premise=[
            ("meaningfulness", "Low"),
            ("OR", "meaningfulness", "Neutral"),
            ("OR", "acceptability", "Acceptable"),
            ("OR", "faithfulness", "Unfaithful"),
            ("OR", "okd", "Violating"),
            ("OR", "similarity", "Violating"),
        ],
        consequence=[("decision", "NOT_EXPLAINABLE")],
    )
]

from fuzzy_expert.inference import DecompositionalInference

model = DecompositionalInference(
    and_operator="min",
    or_operator="max",
    implication_operator="Rc",
    composition_operator="max-min",
    production_link="max",
    defuzzification_operator="cog",
)

model(
    variables=variables,
    rules=rules,
    meaningfulness=0.03,
    acceptability=5.64,
    faithfulness=1.0,
    okd=1.0,
    similarity=1.0,
)

result = model.defuzzificated_infered_memberships
for output in result:
    print(f'{output}')

print(f'Highest score acceptability:')
for key in get_keys_with_highest_value(result):
    print(f'{key} : {result[key]}')
